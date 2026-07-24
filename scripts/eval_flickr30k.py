"""
GaussianImageDistribution - Flickr30K Cross-Dataset Generalization Evaluation (Exp6)

Evaluates models trained on MSCOCO on the Flickr30K dataset to test cross-dataset
generalization. Computes image-text retrieval R@K metrics.

Usage:
    python scripts/eval_flickr30k.py --model-type dist_align
    python main.py --task eval_flickr30k --model-type dist_align
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
from data.flickr30k_dataset import get_flickr30k_test_loader
from models.clip_baseline import CLIPFineTuneBaseline
from models.dist_align_model import DistributionAlignmentModel
from models.prolip_model import ProLIPModel
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


logger = get_logger("eval_flickr30k", config.LOG_DIR / "flickr30k.log")


# =============================================================================
# Model Configuration
# =============================================================================

MODEL_CONFIGS = {
    "dist_align": {
        "model_fn": lambda: DistributionAlignmentModel(
            freeze_clip=True,
            distribution_merging=config.DIST_ALIGN_DISTRIBUTION_MERGING,
        ),
        "default_ckpt": config.DIST_ALIGN_BEST_CKPT,
        "output_path": config.OUTPUT_DIR / "flickr30k_dist_align_results.json",
        "has_distribution": True,
    },
    "clip_baseline": {
        "model_fn": lambda: CLIPFineTuneBaseline(),
        "default_ckpt": config.CLIP_BASELINE_BEST_CKPT,
        "output_path": config.OUTPUT_DIR / "flickr30k_clip_baseline_results.json",
        "has_distribution": False,
    },
    "prolip": {
        "model_fn": lambda: ProLIPModel(freeze=True),
        "default_ckpt": config.PROLIP_BEST_CKPT,
        "output_path": config.OUTPUT_DIR / "flickr30k_prolip_results.json",
        "has_distribution": True,
    },
    "clip_zero_shot": {
        "model_fn": lambda: CLIPFineTuneBaseline(
            freeze_image=True,
            freeze_text=True,
        ),
        "default_ckpt": None,  # Frozen CLIP, no checkpoint needed
        "output_path": config.OUTPUT_DIR / "flickr30k_clip_zero_shot_results.json",
        "has_distribution": False,
    },
}


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Exp6: Flickr30K Cross-Dataset Generalization Evaluation"
    )

    parser.add_argument(
        "--model-type", type=str, required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model type to evaluate",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (uses default if None)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.EVAL_BATCH_SIZE,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--recall-at-k", type=int, nargs="+",
        default=config.RECALL_AT_K,
        help="Recall@K values to compute",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Number of samples to evaluate (default: use full test set)",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--mscoco-results", type=str, default=None,
        help="Path to MSCOCO results JSON for comparison",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: config.OUTPUT_DIR / flickr30k)",
    )

    return parser.parse_args()


# =============================================================================
# Feature Extraction Functions
# =============================================================================

@torch.no_grad()
def extract_features_clip(model, dataloader, device, num_samples=None):
    """Extract features for CLIP-based models (zero-shot and fine-tune)."""
    model.eval()
    all_img, all_text = [], []
    sample_count = 0

    for batch in tqdm(dataloader, desc="Extracting features"):
        if batch is None:
            continue

        pil_images = batch["image"]
        caption_lists = batch["captions"]

        pixel_values = model.process_images(pil_images).to(device)

        # Use first caption per image
        selected_captions = [caps[0] for caps in caption_lists]
        text_inputs = model.process_text(selected_captions)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        img_feat, text_feat = model(
            images=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            normalize=True,
        )

        all_img.append(img_feat.cpu())
        all_text.append(text_feat.cpu())

        sample_count += len(pil_images)
        if num_samples and sample_count >= num_samples:
            break

    img_features = torch.cat(all_img, dim=0)
    text_features = torch.cat(all_text, dim=0)

    if num_samples:
        img_features = img_features[:num_samples]
        text_features = text_features[:num_samples]

    return img_features, text_features


@torch.no_grad()
def extract_features_distribution(model, dataloader, device, num_samples=None):
    """Extract features for distribution-based models (dist_align, prolip)."""
    model.eval()
    all_img_mu, all_text_mu = [], []
    all_img_logvar, all_text_logvar = [], []
    all_img_U = []
    sample_count = 0

    for batch in tqdm(dataloader, desc="Extracting features"):
        if batch is None:
            continue

        pil_images = batch["image"]
        caption_lists = batch["captions"]

        pixel_values = model.process_images(pil_images).to(device)

        # Flatten all captions for processing
        batch_size = len(pil_images)
        num_captions = len(caption_lists[0])

        all_captions = []
        for caption_list in caption_lists:
            all_captions.extend(caption_list)

        text_inputs = model.process_text(all_captions)
        input_ids = text_inputs["input_ids"].view(batch_size, num_captions, -1).to(device)
        attention_mask = text_inputs["attention_mask"].view(batch_size, num_captions, -1).to(device)

        outputs = model(pixel_values, input_ids, attention_mask)

        all_img_mu.append(outputs["img_mu"].cpu())
        all_text_mu.append(outputs["text_mu"].cpu())
        all_img_logvar.append(outputs["img_logvar"].cpu())
        all_text_logvar.append(outputs["text_logvar"].cpu())
        img_U_batch = outputs["img_U"]
        all_img_U.append(img_U_batch.cpu() if img_U_batch is not None else None)

        sample_count += batch_size
        if num_samples and sample_count >= num_samples:
            break

    img_mu = torch.cat(all_img_mu, dim=0)
    text_mu = torch.cat(all_text_mu, dim=0)
    img_logvar = torch.cat(all_img_logvar, dim=0)
    text_logvar = torch.cat(all_text_logvar, dim=0)
    if any(u is None for u in all_img_U):
        img_U = None
    else:
        img_U = torch.cat(all_img_U, dim=0)

    if num_samples:
        img_mu = img_mu[:num_samples]
        text_mu = text_mu[:num_samples]
        img_logvar = img_logvar[:num_samples]
        text_logvar = text_logvar[:num_samples]
        if img_U is not None:
            img_U = img_U[:num_samples]

    return img_mu, text_mu, img_logvar, text_logvar, img_U


# =============================================================================
# Recall Computation
# =============================================================================

def compute_recall_chunked(
    img_features: torch.Tensor,
    text_features: torch.Tensor,
    k_values: list,
    chunk_size: int = 1000,
) -> dict:
    """Compute Recall@K in chunks to avoid OOM on large matrices."""
    n = img_features.shape[0]

    # L2 normalize features for cosine similarity
    img_features = F.normalize(img_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    hits = {k: 0 for k in k_values}

    logger.info(f"Computing Recall@K with chunk_size={chunk_size}...")

    for start in tqdm(range(0, n, chunk_size), desc="Recall chunks"):
        end = min(start + chunk_size, n)
        sim_chunk = torch.matmul(img_features[start:end], text_features.T)
        ranked_indices = torch.argsort(sim_chunk, dim=1, descending=True)

        for k in k_values:
            top_k = ranked_indices[:, :k]
            gt = torch.arange(start, end).unsqueeze(1)
            is_in_top_k = (top_k == gt).any(dim=1)
            hits[k] += is_in_top_k.sum().item()

    recall_metrics = {}
    for k in k_values:
        recall_metrics[f"recall@{k}"] = hits[k] / n
        logger.info(f"  Recall@{k}: {recall_metrics[f'recall@{k}']:.4f}")

    return recall_metrics


def compute_recall_uc_chunked(
    img_mu: torch.Tensor,
    img_logvar: torch.Tensor,
    text_mu: torch.Tensor,
    text_logvar: torch.Tensor,
    k_values: list,
    temperature: float = 0.07,
    chunk_size: int = 1000,
) -> dict:
    """
    Compute Recall@K using uncertainty-calibrated similarity.

    sim(x,y) = mu_x . mu_y / (tau * sqrt(1 + mean(sigma_x^2)) * sqrt(1 + mean(sigma_y^2)))
    """
    n = img_mu.shape[0]

    # Precompute per-sample scaling factors (mean, not sum, for dimension-independence)
    img_var_avg = torch.exp(img_logvar).mean(dim=-1)  # (n,)
    text_var_avg = torch.exp(text_logvar).mean(dim=-1)  # (n,)
    img_scale = torch.sqrt(1.0 + img_var_avg)  # (n,)
    text_scale = torch.sqrt(1.0 + text_var_avg)  # (n,)

    # Normalize means
    img_mu_norm = F.normalize(img_mu, dim=-1)
    text_mu_norm = F.normalize(text_mu, dim=-1)

    hits = {k: 0 for k in k_values}

    logger.info(f"Computing UC-Recall@K (tau={temperature}) with chunk_size={chunk_size}...")

    for start in tqdm(range(0, n, chunk_size), desc="UC Recall chunks"):
        end = min(start + chunk_size, n)

        # Mean similarity: [chunk, n]
        sim_chunk = torch.matmul(img_mu_norm[start:end], text_mu_norm.T)

        # Apply uncertainty calibration
        scale_matrix = img_scale[start:end].unsqueeze(1) * text_scale.unsqueeze(0)
        sim_chunk = sim_chunk / (temperature * scale_matrix)

        ranked_indices = torch.argsort(sim_chunk, dim=1, descending=True)

        for k in k_values:
            top_k = ranked_indices[:, :k]
            gt = torch.arange(start, end).unsqueeze(1)
            is_in_top_k = (top_k == gt).any(dim=1)
            hits[k] += is_in_top_k.sum().item()

    recall_metrics = {}
    for k in k_values:
        recall_metrics[f"uc_recall@{k}"] = hits[k] / n
        logger.info(f"  UC-Recall@{k}: {recall_metrics[f'uc_recall@{k}']:.4f}")

    return recall_metrics


def compute_recall_loglik_chunked(
    img_mu, img_logvar, img_U, text_mu, k_values,
    per_dim_normalize=True, use_logdet=True, chunk_size=256,
):
    """Recall@K under the distribution-likelihood score.

    S[n, m] = log N(text_m ; img_n, Sigma_n) in chunks. i2t ranks rows
    (image query -> texts); t2i ranks columns (text query -> images).
    Ground truth is the diagonal (image i matches text i).
    """
    import torch.nn.functional as F
    from utils.distribution_score import image_text_loglik_matrix

    n = img_mu.shape[0]
    img_mu_n = F.normalize(img_mu, dim=-1)
    text_mu_n = F.normalize(text_mu, dim=-1)
    img_var = torch.exp(img_logvar)
    device = img_mu.device

    i2t_hits = {k: 0 for k in k_values}
    t2i_hits = {k: 0 for k in k_values}
    for start in tqdm(range(0, n, chunk_size), desc="loglik i2t chunks"):
        end = min(start + chunk_size, n)
        img_U_chunk = img_U[start:end] if img_U is not None else None
        S = image_text_loglik_matrix(
            img_mu_n[start:end], img_var[start:end], img_U_chunk, text_mu_n,
            per_dim_normalize=per_dim_normalize, use_logdet=use_logdet,
            chunk_size=end - start,
        )  # (C, n)
        ranked = torch.argsort(S, dim=1, descending=True)
        gt = torch.arange(start, end, device=device).unsqueeze(1)
        for k in k_values:
            i2t_hits[k] += (ranked[:, :k] == gt).any(dim=1).sum().item()
    for start in tqdm(range(0, n, chunk_size), desc="loglik t2i chunks"):
        end = min(start + chunk_size, n)
        S = image_text_loglik_matrix(
            img_mu_n, img_var, img_U, text_mu_n[start:end],
            per_dim_normalize=per_dim_normalize, use_logdet=use_logdet,
            chunk_size=256,
        )  # (n, C): rows=images, cols=texts[start:end]
        ranked = torch.argsort(S, dim=0, descending=True)  # (n, C)
        gt = torch.arange(start, end, device=device).unsqueeze(0)  # (1, C)
        for k in k_values:
            t2i_hits[k] += (ranked[:k] == gt).any(dim=0).sum().item()

    return {
        **{f"loglik_i2t_recall@{k}": i2t_hits[k] / n for k in k_values},
        **{f"loglik_t2i_recall@{k}": t2i_hits[k] / n for k in k_values},
    }


def compute_recall_multicaption(
    img_features: torch.Tensor,
    text_features_all: torch.Tensor,
    k_values: list,
    num_captions: int = 5,
    chunk_size: int = 1000,
) -> dict:
    """
    Compute Recall@K for multi-caption setting (Flickr30K has 5 captions per image).

    A retrieval is considered successful if ANY of the K captions for the target
    image appears in the top-K results.

    Args:
        img_features: (N, D) image features
        text_features_all: (N * num_captions, D) all text features
        num_captions: number of captions per image
        k_values: list of K values for Recall@K
        chunk_size: chunk size for memory-efficient computation
    """
    n = img_features.shape[0]

    img_features = F.normalize(img_features, dim=-1)
    text_features_all = F.normalize(text_features_all, dim=-1)

    hits = {k: 0 for k in k_values}

    logger.info(f"Computing multi-caption Recall@K (num_captions={num_captions})...")

    for start in tqdm(range(0, n, chunk_size), desc="Multi-caption Recall chunks"):
        end = min(start + chunk_size, n)
        sim_chunk = torch.matmul(img_features[start:end], text_features_all.T)
        ranked_indices = torch.argsort(sim_chunk, dim=1, descending=True)

        for k in k_values:
            top_k = ranked_indices[:, :k]
            for i in range(start, end):
                # Ground truth: all caption indices for this image
                gt_indices = torch.arange(
                    i * num_captions, (i + 1) * num_captions
                )
                is_match = any(
                    idx.item() in gt_indices.tolist()
                    for idx in top_k[i - start]
                )
                if is_match:
                    hits[k] += 1

    recall_metrics = {}
    for k in k_values:
        recall_metrics[f"recall@{k}"] = hits[k] / n
        logger.info(f"  Multi-caption Recall@{k}: {recall_metrics[f'recall@{k}']:.4f}")

    return recall_metrics


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    """Main evaluation function."""
    args = parse_args()
    set_seed(config.SEED)

    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR / "flickr30k"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = MODEL_CONFIGS[args.model_type]

    logger.info("=" * 60)
    logger.info("Exp6: Flickr30K Cross-Dataset Generalization Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("=" * 60)

    # Load Flickr30K test set
    logger.info("Loading Flickr30K test set...")
    flickr_data = get_flickr30k_test_loader(
        batch_size=args.batch_size,
        num_workers=0,
        num_captions=config.FLICKR30K_NUM_CAPTIONS,
    )

    if flickr_data is None:
        logger.error("Flickr30K dataset not available. Aborting.")
        return

    dataloader = flickr_data["dataloader"]
    dataset = flickr_data["dataset"]
    logger.info(f"Flickr30K test set: {len(dataset)} images loaded")

    # Determine checkpoint path
    checkpoint_path = args.checkpoint or (
        str(model_cfg["default_ckpt"]) if model_cfg["default_ckpt"] else None
    )

    # Load model
    logger.info(f"Loading model: {args.model_type}")
    model = model_cfg["model_fn"]()

    if checkpoint_path and Path(checkpoint_path).exists():
        model.load(checkpoint_path)
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    elif checkpoint_path:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.warning("Proceeding with randomly initialized weights.")

    model = model.to(args.device)
    model.eval()

    # Extract features and compute recall based on model type
    has_distribution = model_cfg["has_distribution"]

    if has_distribution:
        # Distribution-based models: extract mu and logvar
        img_mu, text_mu, img_logvar, text_logvar, img_U = extract_features_distribution(
            model, dataloader, args.device, args.num_samples,
        )

        # Standard cosine recall on mu
        recall_metrics = compute_recall_chunked(
            img_mu, text_mu, args.recall_at_k, chunk_size=1000,
        )

        # UC-style recall with uncertainty calibration
        uc_recall = compute_recall_uc_chunked(
            img_mu, img_logvar, text_mu, text_logvar,
            args.recall_at_k, temperature=0.07, chunk_size=1000,
        )
        recall_metrics.update(uc_recall)

        # Distribution-likelihood recall (uses image covariance as the metric)
        if img_U is not None:
            loglik_recall = compute_recall_loglik_chunked(
                img_mu.to(args.device), img_logvar.to(args.device), img_U.to(args.device),
                text_mu.to(args.device), args.recall_at_k,
                per_dim_normalize=config.MSDA_PER_DIM_NORMALIZE,
                use_logdet=config.MSDA_USE_LOGDET,
            )
            recall_metrics.update(loglik_recall)

    else:
        # CLIP-based models (zero-shot and fine-tune): simple point features
        img_features, text_features = extract_features_clip(
            model, dataloader, args.device, args.num_samples,
        )

        recall_metrics = compute_recall_chunked(
            img_features, text_features, args.recall_at_k, chunk_size=1000,
        )

    # Load MSCOCO results for comparison if provided
    mscoco_metrics = None
    if args.mscoco_results and Path(args.mscoco_results).exists():
        with open(args.mscoco_results, "r") as f:
            mscoco_data = json.load(f)
            mscoco_metrics = mscoco_data.get("metrics", mscoco_data)
        logger.info(f"Loaded MSCOCO results from: {args.mscoco_results}")

    # Save results
    results = {
        "model_type": args.model_type,
        "checkpoint": checkpoint_path,
        "dataset": "flickr30k_test",
        "num_images": len(dataset),
        "recall_at_k": args.recall_at_k,
        "metrics": recall_metrics,
    }

    if mscoco_metrics:
        results["mscoco_metrics"] = mscoco_metrics

    output_path = output_dir / f"flickr30k_{args.model_type}_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"Exp6: Flickr30K Cross-Dataset Evaluation ({args.model_type})")
    print("=" * 60)
    print(f"{'Metric':<25} {'Flickr30K':>12}", end="")
    if mscoco_metrics:
        print(f" {'MSCOCO':>12} {'Delta':>12}")
    else:
        print()
    print("-" * 60)

    for metric_name, value in recall_metrics.items():
        print(f"{metric_name:<25} {value:>12.4f}", end="")
        if mscoco_metrics and metric_name in mscoco_metrics:
            delta = value - mscoco_metrics[metric_name]
            print(f" {mscoco_metrics[metric_name]:>12.4f} {delta:>+12.4f}")
        else:
            print()

    print("=" * 60)
    logger.info("Flickr30K evaluation complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Flickr30K evaluation failed")
        raise
