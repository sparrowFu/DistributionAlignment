#!/usr/bin/env python
"""Path-1 scoring-metric adaptation sweep (zero training).

For each model (MCDisp unfrozen/frozen KL, CLIP-FT fairness control), extract
features once on coco, then sweep scoring variants for the per-caption
unified protocol, SELECT hyperparameters on a dev half and report on the
held-out test half:

  V0  baseline           cos(img, cap)
  V1  set fusion         cos + beta * cos(img, merged_set_of_cap)     [CLIP gets it too]
  V2  uncertainty fusion V1 - eta * sigma_bar_text(cap)               [MCDisp only]
  V3  MC expected cosine E_{v~N(mu_v, Sigma_v)}[cos(v, cap)]          [MCDisp only; uses sigma AND U]
  V4  raw Mahalanobis    cos - gamma * q(img, cap)                    [MCDisp only]

Both directions (I2T / T2I) share the same S(i, t) definition per plan §4.2
legacy; T2I fusion uses the candidate image's caption-set center.
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
from utils.seed import set_seed
from utils.mcdisp_coverage_score import prepare_image_covariance, _quad_per_dim

logger = get_logger("score_sweep", config.LOG_DIR / "score_sweep.log")

BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
ETAS = [0.0, 0.5, 1.0]
GAMMAS = [0.0, 0.05, 0.15, 0.5, 1.5]
MC_SAMPLES = [4, 16]


@torch.no_grad()
def extract(model, kind, loader, device, n):
    imgs, caps, cap_lvs, img_lvs, img_Us = [], [], [], [], []
    got = 0
    for batch in loader:
        if batch is None:
            continue
        pil, cl = batch["image"], batch["captions"]
        B, K = len(pil), len(cl[0])
        flat = [c for l in cl for c in l]
        ti = model.process_text(flat)
        if kind == "mcdisp":
            out = model(model.process_images(pil).to(device),
                        ti["input_ids"].to(device).view(B, K, -1),
                        ti["attention_mask"].to(device).view(B, K, -1))
            imgs.append(out["img_mu"].float())
            caps.append(out["text_mus"].float())
            cap_lvs.append(out["text_logvars"].float())
            img_lvs.append(out["img_logvar"].float())
            if out.get("img_U") is not None:
                img_Us.append(out["img_U"].float())
        else:
            im, tx = model(images=model.process_images(pil).to(device),
                           input_ids=ti["input_ids"].to(device),
                           attention_mask=ti["attention_mask"].to(device),
                           normalize=True)
            imgs.append(im.float())
            caps.append(tx.float().view(B, K, -1))
        got += B
        if got >= n:
            break
    f = {"img_mu": torch.cat(imgs).cpu(), "text_mus": torch.cat(caps).cpu()}
    if cap_lvs:
        f["text_logvars"] = torch.cat(cap_lvs).cpu()
        f["img_logvar"] = torch.cat(img_lvs).cpu()
        f["img_U"] = torch.cat(img_Us).cpu() if img_Us else None
    return f


def score_variants(f, kind, dev, device):
    """Return dict variant-> (S or per-candidate columns, needs_set) with
    hyperparameter grids folded in as separate variants 'name:param'."""
    img_mu, cap_mus = f["img_mu"].to(device), f["text_mus"].to(device)
    N, K, D = cap_mus.shape
    cap_mu = cap_mus.reshape(N * K, D)
    img_n = F.normalize(img_mu, dim=-1)
    cap_n = F.normalize(cap_mu, dim=-1)
    C = img_n @ cap_n.T                                   # (N, M)
    cap_n3 = F.normalize(cap_mus, dim=-1)
    merged = F.normalize(cap_n3.mean(1), dim=-1)          # (N, D)
    Cset = img_n @ merged.T                               # (N, N) set-level
    Cset_exp = Cset.repeat_interleave(K, dim=1)           # (N, M)

    out = {"V0": C}
    for b in BETAS:
        if b > 0:
            out[f"V1:beta={b}"] = C + b * Cset_exp
    if kind == "mcdisp":
        sig_t = torch.exp(f["text_logvars"].to(device)).mean(-1).reshape(N * K)
        for b in BETAS[1:]:
            for e in ETAS[1:]:
                out[f"V2:beta={b},eta={e}"] = C + b * Cset_exp - e * sig_t.unsqueeze(0)
        state = prepare_image_covariance(f["img_logvar"].to(device),
                                         None if f["img_U"] is None else f["img_U"].to(device))
        # chunked per-dim Mahalanobis: (img_block, cap_block, D) intermediates stay ~3G
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
        for g in GAMMAS[1:]:
            out[f"V4:gamma={g}"] = C - g * q
        # V3: Monte-Carlo expected cosine under the image Gaussian
        for S in MC_SAMPLES:
            torch.manual_seed(0)
            g_cpu = torch.Generator(device="cpu").manual_seed(1234)
            eps = torch.randn(N, S, D, generator=g_cpu).to(device)
            Lc = None
            if f["img_U"] is not None:
                U = f["img_U"].to(device)                 # (N, D, r)
                eps_lr = torch.randn(N, S, U.shape[-1], generator=g_cpu).to(device)
                Lc = torch.einsum("ndr,nsr->nsd", U, eps_lr)
            std = torch.exp(0.5 * f["img_logvar"].to(device))          # (N, D)
            samples = img_mu.unsqueeze(1) + eps * std.unsqueeze(1)     # (N, S, D)
            if Lc is not None:
                samples = samples + Lc
            sn = F.normalize(samples, dim=-1)                          # (N, S, D)
            EC = torch.empty(N, N * K, device=device)
            for c0 in range(0, N * K, 8192):                           # chunk captions
                c1 = min(c0 + 8192, N * K)
                EC[:, c0:c1] = torch.einsum("nsd,md->nsm", sn, cap_n[c0:c1]).mean(1)
            out[f"V3:S={S}"] = EC
            for a in (0.5,):
                out[f"V3mix:S={S},a={a}"] = a * EC + (1 - a) * C
    return out


def recalls_from_scores(S, N, K, device):
    """Six recalls + mR from a full score matrix (row=image, col=caption)."""
    cids = torch.arange(N, device=device).repeat_interleave(K)
    top_i = torch.argsort(-S, dim=1, stable=True)[:, :10]
    top_c = torch.argsort(-S, dim=0, stable=True)[:10].T
    out = {}
    for k in (1, 5, 10):
        sub = top_i[:, :k]
        gi = torch.arange(N, device=device).unsqueeze(1)
        out[f"i2t@{k}"] = ((sub >= gi * K) & (sub < gi * K + K)).any(1).float().mean().item()
        out[f"t2i@{k}"] = (top_c[:, :k] == cids.unsqueeze(1)).any(1).float().mean().item()
    out["mr"] = sum(v for k, v in out.items()) / 6
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    set_seed(config.SEED)

    jobs = [
        ("mcdisp_kl_unfrozen_s42", "mcdisp", "checkpoints/unfrozen_seed42/mcdisp_align_kl_coco_best.pt"),
        ("mcdisp_kl_frozen_s42", "mcdisp", "checkpoints/seed42/mcdisp_align_kl_coco_best.pt"),
        ("clip_ft_s42", "clip", "checkpoints/seed42/clip_baseline_coco_best.pt"),
    ]
    from models.mcdisp_align_model import MCDispAlignModel
    from models.clip_baseline import CLIPFineTuneBaseline

    loader, _ = build_eval_dataloader("coco", batch_size=args.batch_size,
                                      num_workers=4, num_samples=args.num_samples)
    results = {}
    for key, kind, ckpt in jobs:
        model = (MCDispAlignModel() if kind == "mcdisp" else CLIPFineTuneBaseline())
        model.load(ckpt)
        model = model.to(args.device).eval()
        f = extract(model, kind, loader, args.device, args.num_samples)
        del model
        torch.cuda.empty_cache()

        N = f["img_mu"].shape[0]
        K = f["text_mus"].shape[1]
        variants = score_variants(f, kind, None, args.device)
        # even images = dev, odd images = test (each against its own half's
        # gallery; captions travel with their images)
        report = {}
        best = None
        dev_idx = torch.arange(0, N, 2, device=args.device)
        test_idx = torch.arange(1, N, 2, device=args.device)
        def _cols(img_idx):        # caption columns i*K + k of the given images
            return (img_idx * K).repeat_interleave(K) + torch.arange(K, device=args.device).repeat(len(img_idx))
        for name, S in variants.items():
            r_dev = recalls_from_scores(S[dev_idx][:, _cols(dev_idx)], len(dev_idx), K, args.device)
            r_test = recalls_from_scores(S[test_idx][:, _cols(test_idx)], len(test_idx), K, args.device)
            report[name] = {"dev_mr": r_dev["mr"], "test": r_test}
            if best is None or r_dev["mr"] > best[1]:
                best = (name, r_dev["mr"], r_test)
        results[key] = {"variants": report, "dev_selected": {"name": best[0], "test": best[2]}}
        logger.info(f"{key}: dev-selected {best[0]} -> test mr={best[2]['mr']:.4f} "
                    f"i2t@1={best[2]['i2t@1']:.4f} t2i@1={best[2]['t2i@1']:.4f} "
                    f"(V0 test mr={report['V0']['test']['mr']:.4f})")

    out = Path("outputs/score_sweep")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
