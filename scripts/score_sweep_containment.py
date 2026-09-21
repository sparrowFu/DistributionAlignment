#!/usr/bin/env python
"""Containment-based scoring sweep (user proposal).

Training aligns the IMAGE distribution with the caption-SET distribution and
(L_cover) pushes the image ellipsoid to CONTAIN its captions' mean points.
This sweep turns that containment relation into per-(image, caption) scores:

  C0   -||mu_v - mu_t||^2                  raw proximity (containment with
                                           constant isotropic sigma)
  C1   -quad(i,t)                          Mahalanobis containment (= LL)
  C2   column-z containment                quad(i,t) standardized across
                                           IMAGES for each caption: how much
                                           MORE image i contains caption t
                                           than a random image does
  C3   row-z containment                   quad standardized across captions
                                           for each image (image-side hub fix)
  C6   C2 + C3                             two-sided standardized containment
  C4   Gaussian likelihood ratio           log N(mu_t; mu_v, Sigma_v)
                                           - log N(mu_t; pooled cap Gaussian)
  C3b  cos + lam * C2                      containment re-weights cosine
  C5   RRF(cos, C2)                        reciprocal-rank fusion
  CH   CSLS(cos) + lam * C2                containment ON TOP of CSLS

Half-protocol (dev=even/test=odd images) hyperparameter selection, same as
score_variant_sweep.py. Features cached to outputs/feat_cache/.
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

logger = get_logger("score_sweep_cont", config.LOG_DIR / "score_sweep_containment.log")

LAMBDAS = [0.1, 0.3, 1.0]
RRF_K = 60


def get_features(model, loader, device, n, cache_key):
    cache = Path("outputs/feat_cache") / f"{cache_key}.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return torch.load(cache, weights_only=False)
    f = extract(model, "mcdisp", loader, device, n)
    torch.save(f, cache)
    return f


@torch.no_grad()
def containment_variants(f, device):
    img_mu = f["img_mu"].to(device).float()
    cap_mus = f["text_mus"].to(device).float()
    t_lv = f["text_logvars"].to(device).float()
    i_lv = f["img_logvar"].to(device).float()
    i_U = None if f.get("img_U") is None else f["img_U"].to(device).float()
    N, K, D = cap_mus.shape
    M = N * K
    cap_mu = cap_mus.reshape(M, D)

    img_n = F.normalize(img_mu, dim=-1)
    cap_n = F.normalize(cap_mu, dim=-1)
    C = img_n @ cap_n.T

    # quad(i,t): Mahalanobis^2/D of mu_t under Sigma_v (chunked)
    state = prepare_image_covariance(i_lv, i_U)
    q = torch.empty(N, M, device=device)
    d2 = torch.empty(N, M, device=device)        # plain L2^2 for C0
    for i0 in range(0, N, 256):
        i1 = min(i0 + 256, N)
        for c0 in range(0, M, 4096):
            c1 = min(c0 + 4096, M)
            delta = cap_mu[c0:c1].unsqueeze(0) - img_mu[i0:i1].unsqueeze(1)
            q[i0:i1, c0:c1] = _quad_per_dim(
                delta,
                state["inv_var"][i0:i1].unsqueeze(1),
                None if state["U_scaled"] is None else state["U_scaled"][i0:i1].unsqueeze(1),
                state["chol"][i0:i1].unsqueeze(1) if state["chol"] is not None else None)
            d2[i0:i1, c0:c1] = (delta ** 2).sum(-1)

    out = {"C0": -d2, "C1": -q}
    del d2

    # column-z: standardize q across images per caption (caption-side null)
    cm, cs = q.mean(0, keepdim=True), q.std(0, keepdim=True).clamp_min(1e-6)
    zc = (cm - q) / cs
    # row-z: standardize across captions per image (image-side hubness)
    rm, rs = q.mean(1, keepdim=True), q.std(1, keepdim=True).clamp_min(1e-6)
    zr = (rm - q) / rs
    out["C2"] = zc
    out["C3"] = zr
    out["C6"] = zc + zr
    del q

    # C4: Gaussian LR vs pooled caption background (diag)
    bg_mean = cap_mu.mean(0)
    bg_var = (cap_mu - bg_mean).var(0).clamp_min(1e-6)
    quad_bg = ((cap_mu - bg_mean) ** 2 / bg_var).sum(-1) / D        # (M,)
    ld_v = i_lv.sum(-1) / D                                         # (N,)
    out["C4"] = -0.5 * (-out["C1"] * D) - 0.5 * ld_v.unsqueeze(1) + 0.5 * quad_bg.unsqueeze(0)
    # (C4 row-wise is -0.5*quad*D; column-wise adds the background quad)

    for lam in LAMBDAS:
        out[f"C3b:lam={lam}"] = C + lam * zc
    r_c = C.topk(RRF_K, dim=1).indices
    r_z = zc.topk(RRF_K, dim=1).indices
    fused = torch.zeros_like(C)
    fused.scatter_add_(1, r_c, 1.0 / (RRF_K + 1 + r_c).float())
    fused.scatter_add_(1, r_z, 1.0 / (RRF_K + 1 + r_z).float())
    out["C5"] = fused
    del r_c, r_z, fused

    # CSLS baseline + containment hybrid
    k = 5
    r_img = C.topk(k, dim=1).values.mean(1, keepdim=True)
    r_cap = C.topk(k, dim=0).values.mean(0, keepdim=True)
    csls = 2 * C - r_img - r_cap
    out["CH:csls"] = csls
    for lam in [0.1, 0.3]:
        out[f"CH:lam={lam}"] = csls + lam * zc
    out["V0"] = C
    return out


def eval_half(f, device, K):
    N = f["img_mu"].shape[0]
    variants = containment_variants(f, device)

    def _cols(img_idx):
        return (img_idx * K).repeat_interleave(K) + torch.arange(K, device=device).repeat(len(img_idx))

    dev_idx = torch.arange(0, N, 2, device=device)
    test_idx = torch.arange(1, N, 2, device=device)
    report, best = {}, None
    while variants:
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
        f = get_features(model, loader, args.device, args.num_samples, key)
        del model
        torch.cuda.empty_cache()

        K = f["text_mus"].shape[1]
        report, best = eval_half(f, args.device, K)
        results[key] = {"variants": report, "dev_selected": {"name": best[0], "test": best[2]}}
        v0 = report["V0"]["test"]
        logger.info(
            f"{key}: V0 test mr={v0['mr']:.4f} i2t@1={v0['i2t@1']:.4f} t2i@1={v0['t2i@1']:.4f} | "
            f"dev-selected {best[0]} -> test mr={best[2]['mr']:.4f} "
            f"i2t@1={best[2]['i2t@1']:.4f} t2i@1={best[2]['t2i@1']:.4f}")
        del f
        torch.cuda.empty_cache()

    out = Path("outputs/score_sweep")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"results_containment_{args.dataset}.json").write_text(json.dumps(results, indent=2))
    logger.info("saved outputs/score_sweep/results_containment_%s.json", args.dataset)


if __name__ == "__main__":
    main()
