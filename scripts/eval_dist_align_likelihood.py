"""
Distribution-aware retrieval evaluation for dist_align (MSDA).

This is the "make the distribution actually participate in inference" diagnostic.
Unlike ``evaluate_dist_align.py`` (whose primary score is the uncertainty-
discounted *cosine* -- only a scalar sigma^2 discount, the per-D variance and the
covariance factor U never enter retrieval), this script scores retrieval with the
FULL image Gaussian

    Sigma_v = diag(sigma_v^2) + U_v U_v^T

via the log-likelihood of each text mean under each image distribution

    score(image_n, text_m) = log N(text_m ; img_mean_n, Sigma_n)   [higher = better]

and reports it SIDE BY SIDE with the two mean-based scorers, so the marginal
contribution of sigma^2 / U is directly visible:

    - cos_recall  : cosine-on-means           (mu-only)
    - msda_recall : uncertainty-discounted cos (mu + SCALAR sigma^2 discount)
    - lik_recall  : full-Gaussian likelihood  (mu + per-D sigma^2 + U)   <-- NEW

The likelihood score is asymmetric (text point scored under image Gaussian; text
is diagonal-only in v1 so there is no text covariance). One (N, N) score matrix
yields both directions: I2T ranks each row (image -> texts), T2I ranks each
column (text -> images).

IMPORTANT -- space/scale caveat: sigma^2 was supervised by L_var in the
UN-normalized head space, while L_set / L_cov normalize the means. By default we
L2-normalize the means before scoring (matching the cos/msda scorers and the
likelihood function's contract) and pass sigma^2 / U as the model produced them;
mean(sigma^2) and ||U|| diagnostics are printed so any scale mismatch is
visible. Use --no-normalize-means to score in raw head space, and
--per-dim-normalize / --no-logdet to ablate the likelihood's two scale knobs.

NO RETRAINING is required: this only loads an existing checkpoint and swaps the
retrieval scorer. Use --which both to compare the mu-selected "best" checkpoint
against the "last" one (the best was selected by msda_recall@1, which is mu-based,
so it is not guaranteed to be the best under the likelihood score).

Usage:
    python scripts/eval_dist_align_likelihood.py --dataset coco --which best
    python scripts/eval_dist_align_likelihood.py --dataset flickr --which both
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow `python scripts/eval_dist_align_likelihood.py` (repo root on sys.path).
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from models.dist_align_model import DistributionAlignmentModel
from utils.distribution_score import image_text_loglik_matrix
from utils.eval_common import build_eval_dataloader, resolve_checkpoint, VALID_DATASETS
from utils.eval_results import append_eval_results, groups_to_flat, print_recall_groups
from utils.logger import get_logger, log_exception
from utils.retrieval import compute_recall_bidirectional, compute_recall_msda_chunked
from utils.seed import set_seed


_LOG_PATH = config.LOG_DIR / "eval_dist_align_likelihood.log"
_RESULTS_PATH = config.OUTPUT_DIR / "dist_align_likelihood_eval_results.json"

logger = get_logger("eval_dist_align_likelihood", _LOG_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Distribution-aware (likelihood) retrieval eval for dist_align")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit checkpoint path. Overrides --dataset/--which.")
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=list(VALID_DATASETS),
                        help="Dataset to evaluate on (also auto-selects checkpoint). Default: coco")
    parser.add_argument("--which", type=str, default="best", choices=["best", "last", "both"],
                        help="Which checkpoint(s) to evaluate: best (mu-selected), last, or both. Default: best")
    parser.add_argument("--captions-path", type=str, default=None, help="coco only; overrides config captions file")
    parser.add_argument("--images-dir", type=str, default=None, help="coco only; overrides config images dir")
    parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    parser.add_argument("--recall-at-k", type=int, nargs="+", default=config.RECALL_AT_K)
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="Number of images to evaluate (coco subset; flickr uses full test). Default: 5000")
    parser.add_argument("--tau", type=float, default=config.MSDA_TAU,
                        help="Temperature for the MSDA uncertainty-discounted similarity")
    # Likelihood knobs (the two deprecated-but-relevant scale controls).
    parser.add_argument("--per-dim-normalize", action="store_true", default=True,
                        help="Divide the Mahalanobis term by D (default: True).")
    parser.add_argument("--no-per-dim-normalize", dest="per_dim_normalize", action="store_false")
    parser.add_argument("--use-logdet", action="store_true", default=True,
                        help="Include the -0.5*log|Sigma| normalization term (default: True).")
    parser.add_argument("--no-logdet", dest="use_logdet", action="store_false")
    parser.add_argument("--normalize-means", action="store_true", default=True,
                        help="L2-normalize means before likelihood scoring (matches cos/msda; default: True).")
    parser.add_argument("--no-normalize-means", dest="normalize_means", action="store_false")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def extract_features(model, dataloader, device, num_samples=None):
    """Extract distribution features, INCLUDING the image covariance factor U.

    Mirrors evaluate_dist_align.extract_features but also collects img_U (which
    that script drops). Returns CPU tensors; the caller moves what it needs to GPU.
    """
    model.eval()

    all_img_mu, all_text_mu = [], []
    all_img_logvar, all_text_logvar = [], []
    all_img_U = []          # (B, D, r) per batch, or skipped when cov_rank == 0
    sample_count = 0
    has_U = None

    logger.info("Extracting features (incl. covariance factor U)...")
    for batch in tqdm(dataloader, desc="Extract"):
        if batch is None:
            continue

        pil_images = batch["image"]
        caption_lists = batch["captions"]

        pixel_values = model.process_images(pil_images).to(device)
        batch_size = len(pil_images)
        num_captions = len(caption_lists[0])

        all_captions = []
        for cl in caption_lists:
            all_captions.extend(cl)
        text_inputs = model.process_text(all_captions)
        input_ids = text_inputs["input_ids"].view(batch_size, num_captions, -1).to(device)
        attention_mask = text_inputs["attention_mask"].view(batch_size, num_captions, -1).to(device)

        outputs = model(pixel_values, input_ids, attention_mask)

        all_img_mu.append(outputs['img_mu'].cpu())
        all_text_mu.append(outputs['text_mu'].cpu())
        all_img_logvar.append(outputs['img_logvar'].cpu())
        all_text_logvar.append(outputs['text_logvar'].cpu())

        u = outputs['img_U']
        if has_U is None:
            has_U = u is not None
        if u is not None:
            all_img_U.append(u.cpu())

        sample_count += batch_size
        if num_samples and sample_count >= num_samples:
            break

    def _cat(lst):
        t = torch.cat(lst, dim=0)
        return t[:num_samples] if num_samples else t

    img_mu = _cat(all_img_mu)
    text_mu = _cat(all_text_mu)
    img_logvar = _cat(all_img_logvar)
    text_logvar = _cat(all_text_logvar)
    img_U = _cat(all_img_U) if (has_U and all_img_U) else None

    logger.info(f"Features: img_mu {tuple(img_mu.shape)}, img_U "
                f"{tuple(img_U.shape) if img_U is not None else 'None (diagonal-only)'}")
    return img_mu, text_mu, img_logvar, text_logvar, img_U


@torch.no_grad()
def recall_from_score_matrix(sim, k_values):
    """Bidirectional Recall@K from a precomputed (N, N) score matrix.

    Higher score = better match; the diagonal is the positive pair (the eval
    loaders use shuffle=False, so sample i stays aligned with its own captions).

    Returns (i2t, t2i) dicts mapping k -> recall.
    """
    n = sim.shape[0]
    gt = torch.arange(n, device=sim.device)

    # I2T: for each image (row), rank texts.
    ranked_rows = torch.argsort(sim, dim=1, descending=True)          # (N, N)
    i2t = {k: (ranked_rows[:, :k] == gt.unsqueeze(1)).any(dim=1).sum().item() / n
           for k in k_values}

    # T2I: for each text (column), rank images.
    ranked_cols = torch.argsort(sim, dim=0, descending=True)          # (N, N)
    t2i = {k: (ranked_cols[:k, :] == gt.unsqueeze(0)).any(dim=0).sum().item() / n
           for k in k_values}
    return i2t, t2i


def _group(family, label, i2t, t2i, k_values):
    """Build a recall group in the shape print_recall_groups / groups_to_flat expect."""
    return {
        "family": family,
        "label": label,
        "per_k": {
            k: {"i2t": i2t[k], "t2i": t2i[k], "mean": (i2t[k] + t2i[k]) / 2}
            for k in k_values
        },
    }


@torch.no_grad()
def evaluate_one(args, checkpoint_path):
    """Run cos / msda / lik retrieval on one checkpoint. Returns (groups, extras)."""
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {checkpoint_path}")

    # Build model with cov_rank matching the checkpoint (load() rebuilds the cov
    # head to match, so the constructor default is just a starting point).
    model = DistributionAlignmentModel(
        freeze_clip=config.DIST_ALIGN_FREEZE_CLIP,
        distribution_merging=config.DIST_ALIGN_DISTRIBUTION_MERGING,
        cov_rank=config.MSDA_COV_RANK,
    )
    model.load(checkpoint_path)
    logger.info(f"Loaded with cov_rank={model.cov_rank} (U is {'present' if model.cov_rank > 0 else 'absent (diagonal-only)'})")
    model = model.to(args.device)

    dataloader, num_eval_samples = build_eval_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
        num_samples=args.num_samples,
        captions_path=args.captions_path,
        images_dir=args.images_dir,
    )
    logger.info(f"Dataset ({args.dataset}): {num_eval_samples} samples")

    img_mu, text_mu, img_logvar, text_logvar, img_U = extract_features(
        model, dataloader, args.device, args.num_samples)

    img_mu_d = img_mu.to(args.device)
    text_mu_d = text_mu.to(args.device)
    img_lv_d = img_logvar.to(args.device)
    text_lv_d = text_logvar.to(args.device)
    img_U_d = img_U.to(args.device) if img_U is not None else None

    # --- Diagnostics: sigma^2 / U scale (so likelihood numbers are interpretable) ---
    img_var = torch.exp(img_logvar)
    mu_norm_sq = (img_mu.float() ** 2).sum(dim=-1).mean().item()
    extras = {
        "cov_rank": int(model.cov_rank),
        "mean_sigma2_img": float(img_var.mean().item()),
        "mean_sigma2_text": float(torch.exp(text_logvar).mean().item()),
        "mean_mu_norm_sq_img": mu_norm_sq,
        # sigma^2 rescaled into normalized-mean space (divide by ||mu||^2).
        "mean_sigma2_img_normalized_space": float((img_var / (img_mu.float() ** 2).sum(dim=-1, keepdim=True)).mean().item()),
        "mean_U_fro_per_image": (float((img_U.float() ** 2).sum(dim=(1, 2)).mean().item()) if img_U is not None else None),
    }
    logger.info(
        "Diagnostics: mean sigma^2_img=%.4f (=%.4f in normalized-mean space, "
        "since mean||mu_img||^2=%.4f), mean sigma^2_text=%.4f, "
        "mean||U||_F/img=%s",
        extras["mean_sigma2_img"], extras["mean_sigma2_img_normalized_space"],
        mu_norm_sq, extras["mean_sigma2_text"],
        f"{extras['mean_U_fro_per_image']:.4f}" if extras["mean_U_fro_per_image"] is not None else "N/A",
    )

    # --- Scorer 1: cosine-on-means (mu-only baseline) ---
    cos = compute_recall_bidirectional(img_mu_d, text_mu_d, args.recall_at_k, normalize=True)
    cos_i2t = {k: cos[f"recall_i2t@{k}"] for k in args.recall_at_k}
    cos_t2i = {k: cos[f"recall_t2i@{k}"] for k in args.recall_at_k}

    # --- Scorer 2: MSDA uncertainty-discounted cosine (mu + scalar sigma^2 discount) ---
    msda = compute_recall_msda_chunked(img_mu_d, img_lv_d, text_mu_d, text_lv_d,
                                       args.recall_at_k, tau=args.tau)
    msda_i2t = {k: msda[f"msda_recall_i2t@{k}"] for k in args.recall_at_k}
    msda_t2i = {k: msda[f"msda_recall_t2i@{k}"] for k in args.recall_at_k}

    # --- Scorer 3: full-Gaussian likelihood (mu + per-D sigma^2 + U) ---
    lik_img_mean = F.normalize(img_mu_d, dim=-1) if args.normalize_means else img_mu_d
    lik_text_mean = F.normalize(text_mu_d, dim=-1) if args.normalize_means else text_mu_d
    sim_lik = image_text_loglik_matrix(
        lik_img_mean, torch.exp(img_lv_d), img_U_d, lik_text_mean,
        eps=config.MSDA_COV_EPS,
        per_dim_normalize=args.per_dim_normalize,
        use_logdet=args.use_logdet,
    )
    lik_i2t, lik_t2i = recall_from_score_matrix(sim_lik, args.recall_at_k)

    groups = [
        _group("cos_recall", "Cosine-on-means       (mu-only)", cos_i2t, cos_t2i, args.recall_at_k),
        _group("msda_recall", "MSDA discounted cos   (mu + SCALAR sigma^2)", msda_i2t, msda_t2i, args.recall_at_k),
        _group("lik_recall", "Full-Gaussian lik     (mu + per-D sigma^2 + U)", lik_i2t, lik_t2i, args.recall_at_k),
    ]
    print_recall_groups(groups, logger)
    return groups, extras


def main():
    args = parse_args()
    set_seed(config.SEED)

    if args.checkpoint:
        ckpts = [args.checkpoint]
    else:
        ckpts = [str(resolve_checkpoint("dist_align", args.dataset, w))
                 for w in (("best", "last") if args.which == "both" else (args.which,))]

    logger.info(f"Dataset={args.dataset} | which={args.which} | normalize_means={args.normalize_means} | "
                f"per_dim_normalize={args.per_dim_normalize} | use_logdet={args.use_logdet} | tau={args.tau}")

    all_results = []
    for ckpt in ckpts:
        groups, extras = evaluate_one(args, ckpt)
        all_results.append({
            "checkpoint": ckpt,
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "tau": args.tau,
            "normalize_means": args.normalize_means,
            "per_dim_normalize": args.per_dim_normalize,
            "use_logdet": args.use_logdet,
            "diagnostics": extras,
            "metrics": groups_to_flat(groups),
        })

    # --- Compact side-by-side summary across all checkpoints ---
    logger.info("=" * 70)
    logger.info("Side-by-side R@1 (mean of I2T+T2I):")
    header = f"  {'checkpoint':<48} {'cos':>7} {'msda':>7} {'lik':>7}"
    logger.info(header)
    for r in all_results:
        m = r["metrics"]
        name = Path(r["checkpoint"]).name
        logger.info(f"  {name:<48} "
                    f"{m.get('cos_recall@1', float('nan')):>7.3f} "
                    f"{m.get('msda_recall@1', float('nan')):>7.3f} "
                    f"{m.get('lik_recall@1', float('nan')):>7.3f}")
    logger.info("=" * 70)

    append_eval_results(str(_RESULTS_PATH), {"runs": all_results}, logger)
    logger.info(f"Results appended to {_RESULTS_PATH}")
    logger.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Likelihood evaluation failed")
        raise
