#!/usr/bin/env python
"""Analytical experiment: do text representations fall INSIDE the aligned
image distribution?

For every image, MCDisp-Align learns a Gaussian N(mu_v, Sigma_v),
Sigma_v = diag(sigma^2) + U U^T, whose variance is SUPERVISED toward the
multi-caption semantic spread. This script tests that claim geometrically:
compute the squared Mahalanobis distance of each caption center mu_k w.r.t.
its own image Gaussian and check containment against the chi-square(D)
quantiles (a point actually sampled from the Gaussian falls inside the 95%
ellipsoid with probability 0.95, so a well-aligned image distribution should
contain its captions at a comparable rate).

Reported per model (MCDisp-Align std / KL: full diag+low-rank covariance via
Woodbury; ProLIP fine-tuned / zero-shot: diagonal Gaussian, variance NOT
supervised by caption spread -- the control):

  contain@{50,95,99}   fraction of OWN captions inside the chi2_D quantile
  foreign@{...}        same for captions of OTHER images (control: a
                       meaningful distribution should contain own >> foreign)
  mean d2 own/foreign  vs E[chi2_D] = D reference
  set-center d2        distance of the merged caption-set center (mcdisp only)

Usage:
  python scripts/eval_text_containment.py [--datasets coco flickr]
      [--num-images 100] [--ckpt-root checkpoints/seed42]
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.logger import get_logger, log_exception
from utils.seed import set_seed
from utils.eval_common import build_eval_dataloader

from utils.cpu_affinity import apply_cpu_affinity
apply_cpu_affinity()

logger = get_logger("containment", config.LOG_DIR / "text_containment.log")

MODEL_SPECS = [
    ("mcdisp_align", "MCDisp-Align (std)"),
    ("mcdisp_align_kl", "MCDisp-Align (KL)"),
    ("prolip", "ProLIP fine-tuned"),
    ("prolip_zero_shot", "ProLIP zero-shot"),
]


def chi2_quantiles(df: int, ps=(0.5, 0.95, 0.99)):
    try:
        from scipy.stats import chi2
        return {p: float(chi2.ppf(p, df)) for p in ps}
    except ImportError:
        import math
        # Wilson-Hilferty approximation (accurate for large df)
        z = {0.5: 0.0, 0.95: 1.6449, 0.99: 2.3263}
        return {p: df * (1 - 2 / (9 * df) + z[p] * math.sqrt(2 / (9 * df))) ** 3
                for p in ps}


def mahalanobis_diag(d, inv_var):
    """d: (..., D), inv_var: broadcastable diag(1/sigma^2)."""
    return (d * d * inv_var).sum(-1)


def mahalanobis_lowrank(d, var, U):
    """d: (..., D); var: (..., D) diagonal; U: (..., D, r) low-rank factor.

    d^T (diag(var) + U U^T)^-1 d via Woodbury, vectorized over the leading
    (batch/pair) dims: var and U broadcast against d.
    """
    inv_var = 1.0 / var                              # (..., D)
    t1 = (U * inv_var.unsqueeze(-1) * d.unsqueeze(-1)).sum(-2)   # (..., r)
    A = torch.einsum('...dr,...ds->...rs', U * inv_var.unsqueeze(-1), U)  # (...,r,r)
    A = A + torch.eye(A.shape[-1], device=A.device)
    y = torch.linalg.solve(A, t1.unsqueeze(-1)).squeeze(-1)      # (..., r)
    quad = (d * d * inv_var).sum(-1) - (t1 * y).sum(-1)
    return quad


@torch.no_grad()
def extract(model_key, model, batches, device):
    """-> dict with img_mu/img_var (N,D), text_mus (N,K,D), img_U or None."""
    img_mu_l, img_var_l, cap_l, u_l = [], [], [], []
    has_U = None
    for pil_images, caption_lists in tqdm(batches, desc=f"encode[{model_key}]", leave=False):
        B, K = len(caption_lists), len(caption_lists[0])
        flat = [c for cl in caption_lists for c in cl]
        pixel_values = model.process_images(pil_images).to(device)
        ti = model.process_text(flat)
        input_ids = ti["input_ids"].to(device)
        if model_key.startswith("mcdisp"):
            am = ti["attention_mask"].to(device)
            out = model(pixel_values, input_ids.view(B, K, -1), am.view(B, K, -1))
            img_mu_l.append(out["img_mu"].cpu())
            img_var_l.append(out["img_sigma"].pow(2).cpu())
            cap_l.append(out["text_mus"].cpu())          # (B, K, D)
            U = out.get("img_U")
            if U is not None:
                u_l.append(U.cpu())
                has_U = True
        else:  # prolip / prolip_zero_shot
            img = model.encode_images(pixel_values, normalize=False)
            txt = model.encode_texts(input_ids, normalize=False)
            img_mu_l.append(img["mean"].cpu())
            img_var_l.append(torch.exp(img["std"]).cpu())
            cap_l.append(txt["mean"].cpu().view(B, K, -1))
            has_U = False
    feats = {
        "img_mu": torch.cat(img_mu_l), "img_var": torch.cat(img_var_l),
        "text_mus": torch.cat(cap_l),
        "img_U": torch.cat(u_l) if has_U else None,
    }
    return feats


def containment_stats(feats, qs, shift_half=True):
    N, K, D = feats["text_mus"].shape
    img_mu, img_var, U = feats["img_mu"], feats["img_var"], feats["img_U"]

    def d2_for(cap_mus):  # cap_mus (N, K, D) -> (N, K) squared Mahalanobis
        d = cap_mus - img_mu.unsqueeze(1)
        if U is not None:
            return mahalanobis_lowrank(d, img_var.unsqueeze(1), U.unsqueeze(1))
        return mahalanobis_diag(d, 1.0 / img_var.unsqueeze(1))

    own = d2_for(feats["text_mus"])
    shift = (torch.arange(N) + N // 2) % N
    foreign = d2_for(feats["text_mus"][shift])

    out = {
        "d2_own_mean": own.mean().item(), "d2_own_std": own.std(unbiased=True).item(),
        "d2_foreign_mean": foreign.mean().item(),
        "E_chi2": float(D),
    }
    for p, q in qs.items():
        out[f"contain@{int(p*100)}"] = (own <= q).float().mean().item()
        out[f"foreign@{int(p*100)}"] = (foreign <= q).float().mean().item()
    return out, own


def build_model(key, ckpt_root, dataset):
    if key in ("mcdisp_align", "mcdisp_align_kl"):
        from models.mcdisp_align_model import MCDispAlignModel
        m = MCDispAlignModel()
        m.load(str(Path(ckpt_root) / f"{key}_{dataset}_best.pt"))
        return m
    if key == "prolip":
        from models.prolip_model import ProLIPModel
        m = ProLIPModel()
        m.load(str(Path(ckpt_root) / f"prolip_{dataset}_best.pt"))
        return m
    if key == "prolip_zero_shot":
        from models.prolip_model import ProLIPModel
        return ProLIPModel(freeze=True)
    raise KeyError(key)


def run_dataset(dataset, args, out_dir):
    set_seed(config.SEED)
    loader, _ = build_eval_dataloader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        num_samples=args.num_images if dataset == "coco" else None)
    batches, n = [], 0
    for batch in loader:
        if batch is None:
            continue
        batches.append((batch["image"], batch["captions"]))
        n += len(batch["image"])
        if n >= args.num_images:
            break
    logger.info(f"[{dataset}] collected {n} images")

    results = {}
    for key, disp in MODEL_SPECS:
        if args.models and key not in args.models:
            continue
        model = build_model(key, args.ckpt_root, dataset).to(args.device).eval()
        feats = extract(key, model, batches, args.device)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
        D = feats["text_mus"].shape[-1]
        qs = chi2_quantiles(D)
        stats, own_d2 = containment_stats(feats, qs)
        stats["display"] = disp
        stats["dim"] = D
        stats["quantiles"] = {f"q{int(p*100)}": q for p, q in qs.items()}
        stats["d2_own_raw"] = own_d2.flatten().tolist()
        results[key] = stats
        logger.info(f"[{dataset}] {disp}: contain@95={stats['contain@95']:.3f} "
                    f"(foreign {stats['foreign@95']:.3f}) mean d2={stats['d2_own_mean']:.1f} "
                    f"(E[chi2]={D})")

    ds_dir = out_dir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    with open(ds_dir / "results.json", "w") as f:
        json.dump({"dataset": dataset, "num_images": n, "models": results}, f, indent=2)
    write_table(dataset, results, ds_dir / "summary.md")
    return results


def write_table(dataset, results, path):
    lines = [
        f"# 文本表示是否落在图像分布内（{dataset}）",
        "",
        "马氏距离 d²(μ_k; N(μ_v, Σ_v))，Σ_v=diag(σ²)+UUᵀ（MCDisp）/ 对角（ProLIP）；",
        "contain@95 = 自身 caption 落在 χ²_D 95% 椭圆内的比例（从分布采样的点理论值 0.95）；",
        "foreign = 外来 caption 的同比例（对照，越低说明分布越有选择性）。",
        "",
        "| 模型 | contain@50 | contain@95 | contain@99 | foreign@95 | mean d²(自身) | E[χ²_D] |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, r in results.items():
        lines.append(
            f"| {r['display']} | {r['contain@50']:.3f} | {r['contain@95']:.3f} "
            f"| {r['contain@99']:.3f} | {r['foreign@95']:.3f} "
            f"| {r['d2_own_mean']:.1f} | {r['E_chi2']} |")
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"table written: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["coco", "flickr"])
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--ckpt-root", default="checkpoints/seed42")
    p.add_argument("--models", nargs="+", default=None,
                   help="Subset of model keys to analyse (default: all)")
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "cuda":
            free, _ = torch.cuda.mem_get_info()
            if free < 8 << 30:
                logger.info(f"GPU free {free >> 30}G < 8G -> cpu")
                args.device = "cpu"
    out_dir = Path(args.output_dir) if args.output_dir else (
        config.OUTPUT_DIR / "text_containment")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"device={args.device} ckpt_root={args.ckpt_root}")

    for ds in args.datasets:
        run_dataset(ds, args, out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "containment analysis failed")
        raise
