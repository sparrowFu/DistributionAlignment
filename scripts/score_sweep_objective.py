#!/usr/bin/env python
"""Objective-adapted per-caption scoring sweep for MCDisp-Align (zero training).

Under the unified multi-caption protocol (N images vs N*K captions; any-hit
I2T / per-caption T2I), MCDisp-Align may use its own scorer while baselines
keep theirs. All variants here score an (image, single caption) pair WITHOUT
reading the caption set at inference:

  V0    cos(mu_v, mu_t)                                  reference
  LL    -(q + logdet_v)/D   log N(mu_t; mu_v, Sigma_v)   coverage/likelihood
        q = Mahalanobis^2/D of mu_t under Sigma_v (diag + U U^T);
        logdet via matrix-determinant lemma. Row-wise (=I2T) logdet is
        constant, so LL only differs from -q for T2I ranking.
  SYM   symmetric two-Gaussian: -(q_vt + logdet_v + q_tv + logdet_t)/D
  W2    V0 - lam * mean_d (sigma_v - sigma_t)^2          diag-W2 var mismatch
  SMAX  max_s cos(z_s, mu_t), z_s ~ N(mu_v, Sigma_v)     coverage-style
        multi-point probing (L_cover pushes caption means INSIDE the image
        ellipsoid, so a hit by ANY sample mirrors the any-hit metric)
  STOPm mean of top-m sample similarities (softened SMAX)
  CSLS  2*C - r_img - r_cap (k-reciprocal hubness fix; transductive)

Hyperparameters selected on the dev half (even images), reported on the test
half (odd images), matching scripts/score_variant_sweep.py conventions.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.eval_common import build_eval_dataloader
from utils.logger import get_logger
from utils.mcdisp_coverage_score import prepare_image_covariance, _quad_per_dim
from utils.seed import set_seed

from scripts.score_variant_sweep import extract, recalls_from_scores

logger = get_logger("score_sweep_obj", config.LOG_DIR / "score_sweep_objective.log")

CSLS_KS = [5, 10, 20]
W2_LAMS = [0.5, 2.0]
SAMPLES = [16, 64]
TOPM = 3


def variance_signal_diag(f):
    """Why variance-discount scorers were flat: spread of per-sample v-bar."""
    v_img = f["img_logvar"].exp().mean(-1)          # (N,)
    v_cap = f["text_logvars"].exp().mean(-1).mean(-1)  # (N,) avg over K, D
    out = {
        "img_vbar_mean": v_img.mean().item(),
        "img_vbar_std": v_img.std().item(),
        "img_vbar_cv": (v_img.std() / v_img.mean().clamp_min(1e-12)).item(),
        "cap_vbar_mean": v_cap.mean().item(),
        "cap_vbar_std": v_cap.std().item(),
        "cap_vbar_cv": (v_cap.std() / v_cap.mean().clamp_min(1e-12)).item(),
    }
    if f.get("img_U") is not None:
        # mean top singular value of U per image, relative to its Frobenius norm
        U = f["img_U"].float()
        sv = torch.linalg.svdvals(U)                 # (N, r)
        out["U_s1_mean"] = sv[:, 0].mean().item()
        out["U_fro_mean"] = sv.norm(dim=-1).mean().item()
    return out


def logdet_full(state, img_logvar):
    """log|diag(v)+UU^T|/D per image, via |A|*|I+U^T A^-1 U|.

    G = U^T A^-1 U = chol @ chol^T (per prepare_image_covariance), so
    log|G| = 2 * sum(log diag(chol)) without an explicit logdet call.
    """
    D = img_logvar.shape[-1]
    ld = img_logvar.sum(-1)                          # log|A|
    if state["chol"] is not None:
        ld = ld + 2.0 * torch.log(torch.diagonal(
            state["chol"], dim1=-2, dim2=-1)).sum(-1)
    return ld / D


@torch.no_grad()
def score_variants(f, device):
    """Build per-(image, caption) score matrices from a feature dict."""
    img_mu = f["img_mu"].to(device).float()
    cap_mus = f["text_mus"].to(device).float()
    t_lv = f["text_logvars"].to(device).float()
    i_lv = f["img_logvar"].to(device).float()
    i_U = None if f.get("img_U") is None else f["img_U"].to(device).float()
    N, K, D = cap_mus.shape
    cap_mu = cap_mus.reshape(N * K, D)

    img_n = F.normalize(img_mu, dim=-1)
    cap_n = F.normalize(cap_mu, dim=-1)
    C = img_n @ cap_n.T                               # (N, M)

    out = {"V0": C.clone()}

    # ---- CSLS (hubness fix, transductive over this gallery) -------------
    for k in CSLS_KS:
        r_img = C.topk(k, dim=1).values.mean(1, keepdim=True)   # (N,1)
        r_cap = C.topk(k, dim=0).values.mean(0, keepdim=True)   # (1,M)
        out[f"CSLS:k={k}"] = 2 * C - r_img - r_cap

    # ---- Mahalanobis quad (chunked) + logdet ----------------------------
    state = prepare_image_covariance(i_lv, i_U)
    q = torch.empty(N, N * K, device=device)
    for i0 in range(0, N, 256):
        i1 = min(i0 + 256, N)
        for c0 in range(0, N * K, 4096):
            c1 = min(c0 + 4096, N * K)
            q[i0:i1, c0:c1] = _quad_per_dim(
                cap_mu[c0:c1].unsqueeze(0) - img_mu[i0:i1].unsqueeze(1),
                state["inv_var"][i0:i1].unsqueeze(1),
                None if state["U_scaled"] is None else state["U_scaled"][i0:i1].unsqueeze(1),
                state["chol"][i0:i1].unsqueeze(1) if state["chol"] is not None else None)
    ld_v = logdet_full(state, i_lv)                   # (N,)
    out["LL"] = -(q + ld_v.unsqueeze(1))
    out["LLmix:g=0.15"] = C - 0.15 * (q + ld_v.unsqueeze(1))
    out["LLmix:g=0.5"] = C - 0.5 * (q + ld_v.unsqueeze(1))

    # ---- symmetric two-Gaussian -----------------------------------------
    cap_var = t_lv.reshape(N * K, D).exp()
    ld_t = t_lv.reshape(N * K, D).sum(-1) / D         # log|Sigma_t|/D
    q_tv = torch.empty(N, N * K, device=device)
    for i0 in range(0, N, 256):
        i1 = min(i0 + 256, N)
        for c0 in range(0, N * K, 4096):
            c1 = min(c0 + 4096, N * K)
            delta = cap_mu[c0:c1].unsqueeze(0) - img_mu[i0:i1].unsqueeze(1)
            q_tv[i0:i1, c0:c1] = ((delta ** 2) / cap_var[c0:c1].unsqueeze(0)).sum(-1) / D
    out["SYM"] = -(q + ld_v.unsqueeze(1) + q_tv + ld_t.unsqueeze(0))
    del q_tv

    # ---- W2 variance mismatch -------------------------------------------
    sig_v = i_lv.exp().sqrt().mean(-1, keepdim=True)          # (N,1) mean_d sigma_v
    sig_t = cap_var.sqrt().mean(-1, keepdim=True)             # (M,1)
    w2var = (sig_v - sig_t.t()) ** 2                          # (N,M) mean-level
    for lam in W2_LAMS:
        out[f"W2:lam={lam}"] = C - lam * w2var

    # ---- sample-based coverage probing (SMAX / STOPm) --------------------
    for S in SAMPLES:
        g = torch.Generator(device="cpu").manual_seed(1234)
        eps = torch.randn(N, S, D, generator=g).to(device)
        samples = img_mu.unsqueeze(1) + eps * i_lv.exp().sqrt().unsqueeze(1)
        if i_U is not None:
            eps_lr = torch.randn(N, S, i_U.shape[-1], generator=g).to(device)
            samples = samples + torch.einsum("ndr,nsr->nsd", i_U, eps_lr)
        sn = F.normalize(samples, dim=-1)                     # (N,S,D)
        smax = torch.empty(N, N * K, device=device)
        stop = torch.empty(N, N * K, device=device)
        cap_chunk = max(1024, 8192 * 16 // S)                 # keep sims < ~3GB
        for c0 in range(0, N * K, cap_chunk):
            c1 = min(c0 + cap_chunk, N * K)
            sims = torch.einsum("nsd,md->nsm", sn, cap_n[c0:c1])  # (N,S,m)
            smax[:, c0:c1] = sims.max(1).values
            stop[:, c0:c1] = sims.topk(min(TOPM, S), dim=1).values.mean(1)
            del sims
        out[f"SMAX:S={S}"] = smax
        out[f"STOP{TOPM}:S={S}"] = stop
        del samples, sn, eps
        torch.cuda.empty_cache()

    return out


def eval_half(f, device, K):
    N = f["img_mu"].shape[0]
    variants = score_variants(f, device)

    def _cols(img_idx):
        return (img_idx * K).repeat_interleave(K) + torch.arange(K, device=device).repeat(len(img_idx))

    dev_idx = torch.arange(0, N, 2, device=device)
    test_idx = torch.arange(1, N, 2, device=device)
    report, best = {}, None
    while variants:  # pop so only one score matrix is alive at a time
        name, S = variants.popitem()
        r_dev = recalls_from_scores(S[dev_idx][:, _cols(dev_idx)], len(dev_idx), K, device)
        r_test = recalls_from_scores(S[test_idx][:, _cols(test_idx)], len(test_idx), K, device)
        report[name] = {"dev_mr": r_dev["mr"], "test": r_test}
        if best is None or r_dev["mr"] > best[1]:
            best = (name, r_dev["mr"], r_test)
        del S
        torch.cuda.empty_cache()
    return report, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["coco", "flickr"], default="coco")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    set_seed(config.SEED)

    from models.mcdisp_align_model import MCDispAlignModel
    loader, _ = build_eval_dataloader(args.dataset, batch_size=args.batch_size,
                                      num_workers=4, num_samples=args.num_samples)

    results = {}
    for seed in args.seeds:
        ckpt = f"checkpoints/unfrozen_seed{seed}/mcdisp_align_kl_{args.dataset}_best.pt"
        key = f"mcdisp_kl_unfrozen_s{seed}_{args.dataset}"
        model = MCDispAlignModel()
        model.load(ckpt)
        model = model.to(args.device).eval()
        f = extract(model, "mcdisp", loader, args.device, args.num_samples)
        del model
        torch.cuda.empty_cache()

        K = f["text_mus"].shape[1]
        results[key] = {"variance_signal": variance_signal_diag(f)}
        report, best = eval_half(f, args.device, K)
        results[key]["variants"] = report
        results[key]["dev_selected"] = {"name": best[0], "test": best[2]}
        v0 = report["V0"]["test"]
        logger.info(
            f"{key}: V0 test mr={v0['mr']:.4f} i2t@1={v0['i2t@1']:.4f} t2i@1={v0['t2i@1']:.4f} | "
            f"dev-selected {best[0]} -> test mr={best[2]['mr']:.4f} "
            f"i2t@1={best[2]['i2t@1']:.4f} t2i@1={best[2]['t2i@1']:.4f}")
        del f
        torch.cuda.empty_cache()

    out = Path("outputs/score_sweep")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"results_objective_{args.dataset}.json").write_text(json.dumps(results, indent=2))
    logger.info("saved outputs/score_sweep/results_objective_%s.json", args.dataset)


if __name__ == "__main__":
    main()
