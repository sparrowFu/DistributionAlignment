"""
σ variance-head collapse diagnostic.

QUESTION
    Does σ² track the L_var supervision target (caption_spread)? If not, is it
    because the variance head collapsed (Case A, fixable) or because the target
    itself is ~constant (Case B, premise weak)?

WHY THIS EXISTS
    eval_sigma_analysis.py already showed Pearson(σ², caption_diversity) ≈ -0.02.
    But that alone doesn't say WHY. This script pins it down by comparing the
    *variation* of σ² vs the *variation* of the exact L_var target, quantifying
    the scale mismatch, and testing whether matching in log-space (the likely
    fix) would recover the correlation.

COMPUTES
    - img_var  = exp(img_logvar)                     # the predicted σ² (N,D)
    - caption_spread = mean_k (μ_k - μ̄)²            # the EXACT L_var target (N,D)
      (matches losses/dist_align_losses.py:514)
    - coefficient of variation (CV = std/mean) of each → is each signal varying?
    - scale ratio img_var / caption_spread
    - Pearson/Spearman in RAW and LOG space (log-space test = fix hypothesis)
    - does σ² track any image property at all? (corr with ||μ_img||)
    - scatter plot img_var vs caption_spread (the visual smoking gun)
    - prints a Case A/B verdict + root cause + fix direction

USAGE
    python scripts/eval_sigma_diagnostic.py
    python scripts/eval_sigma_diagnostic.py --num-samples 5000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.caption_dataset import ImageCaptionDataset, filter_none_collate
from models.dist_align_model import DistributionAlignmentModel
from utils.logger import get_logger, log_exception
from utils.seed import set_seed

try:
    from scipy.stats import spearmanr
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


logger = get_logger("sigma_diagnostic", config.SIGMA_ANALYSIS_LOG_PATH)


def parse_args():
    p = argparse.ArgumentParser(description="σ variance-head collapse diagnostic")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--no-plot", action="store_true", help="skip matplotlib scatter")
    return p.parse_args()


@torch.no_grad()
def extract_features(model, dataloader, device, num_samples):
    """Extract img_logvar, img_mu, per-caption text_mus. (mirrors eval_sigma_analysis)"""
    model.eval()
    all_img_logvar, all_img_mu, all_text_mus = [], [], []
    n = 0
    for batch in tqdm(dataloader, desc="Extracting"):
        if batch is None:
            continue
        pil_images = batch["image"]
        caption_lists = batch["captions"]
        B = len(pil_images)
        K = len(caption_lists[0])
        pixel_values = model.process_images(pil_images).to(device)
        all_captions = [c for cl in caption_lists for c in cl]
        ti = model.process_text(all_captions)
        input_ids = ti["input_ids"].view(B, K, -1).to(device)
        attn = ti["attention_mask"].view(B, K, -1).to(device)
        out = model(pixel_values, input_ids, attn)
        all_img_logvar.append(out["img_logvar"].cpu())
        all_img_mu.append(out["img_mu"].cpu())
        all_text_mus.append(out["text_mus"].cpu())
        n += B
        if n >= num_samples:
            break
    return (
        torch.cat(all_img_logvar, dim=0)[:num_samples],
        torch.cat(all_img_mu, dim=0)[:num_samples],
        torch.cat(all_text_mus, dim=0)[:num_samples],
    )


def diagnose(img_logvar: torch.Tensor, img_mu: torch.Tensor, text_mus: torch.Tensor) -> dict:
    """Core diagnostic. All math matches the L_var definition exactly."""
    eps = 1e-12
    img_var = torch.exp(img_logvar)                               # (N,D) predicted σ²
    text_mu_bar = text_mus.mean(dim=1)                            # (N,D) caption center
    # EXACT L_var target: losses/dist_align_losses.py:514
    caption_spread = ((text_mus - text_mu_bar.unsqueeze(1)) ** 2).mean(dim=1)  # (N,D)

    # Per-image scalars (mean over D)
    iv = img_var.mean(dim=-1).numpy()               # (N,)
    cs = caption_spread.mean(dim=-1).numpy()        # (N,)
    img_mu_norm = img_mu.norm(dim=-1).numpy()       # (N,) does σ track the mean at all?

    def cv(x):
        return float(x.std() / (abs(x.mean()) + eps))

    def pear(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum()) + eps))

    res = {
        "n": int(img_var.shape[0]),
        "img_var": {
            "mean": float(iv.mean()), "std": float(iv.std()), "cv": cv(iv),
            "min": float(iv.min()), "max": float(iv.max()),
        },
        "caption_spread": {
            "mean": float(cs.mean()), "std": float(cs.std()), "cv": cv(cs),
            "min": float(cs.min()), "max": float(cs.max()),
        },
        "scale_ratio_img_over_caption": float(iv.mean() / (cs.mean() + eps)),
        "correlation_raw": {
            "pearson_img_vs_caption_spread": pear(iv, cs),
            "pearson_img_var_vs_img_mu_norm": pear(iv, img_mu_norm),
        },
        "correlation_log": {
            # Fix hypothesis: match in log-space so tiny caption_spread values
            # produce comparable-magnitude gradients.
            "pearson_log_img_var_vs_log_caption_spread": pear(np.log(iv + eps), np.log(cs + eps)),
        },
    }
    if _HAVE_SCIPY:
        rho_raw, p_raw = spearmanr(iv, cs)
        rho_log, p_log = spearmanr(np.log(iv + eps), np.log(cs + eps))
        res["correlation_raw"]["spearman"] = float(rho_raw)
        res["correlation_raw"]["spearman_p"] = float(p_raw)
        res["correlation_log"]["spearman_log"] = float(rho_log)
        res["correlation_log"]["spearman_log_p"] = float(p_log)

    # Per-dim correlation (averaged) — does σ track caption_spread along any axis?
    iv_pd = img_var.numpy()                # (N,D)
    cs_pd = caption_spread.numpy()         # (N,D)
    dim_r = []
    for d in range(iv_pd.shape[1]):
        a = iv_pd[:, d]; b = cs_pd[:, d]
        if a.std() < eps or b.std() < eps:
            continue
        dim_r.append(pear(a, b))
    res["mean_per_dim_pearson"] = float(np.mean(dim_r)) if dim_r else 0.0

    # ---- Verdict ----
    cv_cap = res["caption_spread"]["cv"]
    cv_img = res["img_var"]["cv"]
    r_raw = res["correlation_raw"]["pearson_img_vs_caption_spread"]
    r_log = res["correlation_log"]["pearson_log_img_var_vs_log_caption_spread"]

    if cv_cap < 0.15:
        case = "B"
        msg = (
            f"target (caption_spread) itself near-constant (CV={cv_cap:.1%}); "
            "MSCOCO's 5 captions are too similar to give a varying signal. "
            "Premise weak — needs richer diversity (pseudo/hierarchical captions) "
            "or reposition the methodology."
        )
    elif cv_img < 0.15 and abs(r_raw) < 0.15:
        case = "A"
        msg = (
            f"target VARIES (CV={cv_cap:.1%}) but σ² COLLAPSED to a constant "
            f"(CV={cv_img:.1%}, Pearson={r_raw:.3f}). Variance head is not tracking "
            "the supervision. FIXABLE. Likely cause: scale mismatch "
            f"(σ² is {res['scale_ratio_img_over_caption']:.1f}x caption_spread) makes "
            "the L_var MSE gradient tiny vs L_reg pulling σ² toward a constant; "
            "the head takes the easy constant shortcut. "
            "Fix: match in log-space, raise λ_var, cut λ_reg."
        )
        if abs(r_log) > abs(r_raw) + 0.1:
            msg += (f" [log-space correlation {r_log:.3f} >> raw {r_raw:.3f} → "
                    "log-space L_var is a promising fix.]")
    else:
        case = "?"
        msg = (f"neither clean collapse nor clean target; inspect scatter. "
               f"cv_cap={cv_cap:.1%}, cv_img={cv_img:.1%}, r_raw={r_raw:.3f}.")
    res["verdict"] = {"case": case, "explanation": msg}
    return res


def save_scatter(iv, cs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(cs, iv, s=4, alpha=0.2)
    ax.set_xlabel("caption_spread  (L_var target, per-image)")
    ax.set_ylabel("img σ²  (per-image)")
    ax.set_title("σ² vs caption_spread  (flat = variance head collapsed)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(config.SEED)

    out_dir = Path(args.output_dir) if args.output_dir else config.SIGMA_ANALYSIS_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = args.checkpoint or str(config.DIST_ALIGN_BEST_CKPT)
    logger.info(f"Loading model from {ckpt}")
    model = DistributionAlignmentModel(
        freeze_clip=config.DIST_ALIGN_FREEZE_CLIP,
        distribution_merging=config.DIST_ALIGN_DISTRIBUTION_MERGING,
        cov_rank=config.MSDA_COV_RANK,
    )
    if Path(ckpt).exists():
        model.load(ckpt)
    else:
        logger.warning(f"Checkpoint not found: {ckpt} — running with random weights (diagnostic only)")
    model = model.to(args.device)

    dataset = ImageCaptionDataset(
        captions_path=config.CAPTIONS_PATH, images_dir=config.IMAGES_DIR, num_captions=5,
    )
    if args.num_samples < len(dataset):
        gen = torch.Generator().manual_seed(config.SEED)
        idx = torch.randperm(len(dataset), generator=gen)[:args.num_samples].tolist()
        dataset = Subset(dataset, idx)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=filter_none_collate)

    logger.info("Extracting features...")
    img_logvar, img_mu, text_mus = extract_features(model, loader, args.device, args.num_samples)
    logger.info(f"Features: {img_logvar.shape[0]} samples")

    res = diagnose(img_logvar, img_mu, text_mus)

    # Save scatter
    if not args.no_plot:
        iv = torch.exp(img_logvar).mean(dim=-1).numpy()
        cs = ((text_mus - text_mus.mean(dim=1, keepdim=True)) ** 2).mean(dim=(1, 2)).numpy()
        scatter_path = out_dir / "sigma_vs_caption_spread.png"
        try:
            save_scatter(iv, cs, scatter_path)
            res["scatter_plot"] = str(scatter_path)
            logger.info(f"Scatter saved to {scatter_path}")
        except Exception as e:
            logger.warning(f"Plot failed ({e}); skipping")

    out_path = out_dir / "sigma_diagnostic_results.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)

    # ---- Print verdict ----
    print("\n" + "=" * 64)
    print("σ VARIANCE-HEAD COLLAPSE DIAGNOSTIC")
    print("=" * 64)
    print(f"samples: {res['n']}")
    print(f"\nimg σ²        : mean={res['img_var']['mean']:.4f}  std={res['img_var']['std']:.4f}  "
          f"CV={res['img_var']['cv']:.1%}  [{res['img_var']['min']:.4f}, {res['img_var']['max']:.4f}]")
    print(f"caption_spread: mean={res['caption_spread']['mean']:.4f}  std={res['caption_spread']['std']:.4f}  "
          f"CV={res['caption_spread']['cv']:.1%}  [{res['caption_spread']['min']:.4f}, {res['caption_spread']['max']:.4f}]")
    print(f"scale ratio σ²/caption_spread = {res['scale_ratio_img_over_caption']:.1f}x")
    print(f"\nraw   Pearson(σ², caption_spread) = {res['correlation_raw']['pearson_img_vs_caption_spread']:+.4f}"
          + (f"   Spearman={res['correlation_raw'].get('spearman', float('nan')):+.4f}" if _HAVE_SCIPY else ""))
    print(f"log   Pearson(σ², caption_spread) = {res['correlation_log']['pearson_log_img_var_vs_log_caption_spread']:+.4f}"
          + (f"   Spearman={res['correlation_log'].get('spearman_log', float('nan')):+.4f}" if _HAVE_SCIPY else ""))
    print(f"raw   Pearson(σ², ||μ_img||)      = {res['correlation_raw']['pearson_img_var_vs_img_mu_norm']:+.4f}  "
          f"(does σ track ANYTHING?)")
    print(f"mean per-dim Pearson             = {res['mean_per_dim_pearson']:+.4f}")
    print(f"\n>>> VERDICT: Case {res['verdict']['case']}")
    print(f"    {res['verdict']['explanation']}")
    print("=" * 64)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "σ diagnostic failed")
        raise
