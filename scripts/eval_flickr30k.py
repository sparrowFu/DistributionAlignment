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
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.flickr30k_dataset import get_flickr30k_test_loader
from models.clip_baseline import CLIPFineTuneBaseline
from models.dist_align_model import DistributionAlignmentModel
from models.prolip_model import ProLIPModel
from models.grove_model import GroVEModel
from models.icpe_model import ICPEModel
from models.d2p_model import D2PModel
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
        "model_fn": lambda: ProLIPModel(
            freeze_clip=True,
            dropout_rate=config.DIST_ALIGN_DROPOUT_RATE,
        ),
        "default_ckpt": config.PROLIP_BEST_CKPT,
        "output_path": config.OUTPUT_DIR / "flickr30k_prolip_results.json",
        "has_distribution": True,
    },
    "grove": {
        "model_fn": lambda: GroVEModel(
            num_inducing=config.GROVE_NUM_INDUCING,
            freeze_clip=True,
        ),
        "default_ckpt": config.GROVE_BEST_CKPT,
        "output_path": config.OUTPUT_DIR / "flickr30k_grove_results.json",
        "has_distribution": True,
    },
    "icpe": {
        "model_fn": lambda: ICPEModel(
            num_neighbors=config.ICPE_NUM_NEIGHBORS,
            regularization=config.ICPE_REGULARIZATION,
        ),
        "default_ckpt": None,  # ICPE is training-free
        "output_path": config.OUTPUT_DIR / "flickr30k_icpe_results.json",
        "has_distribution": True,
    },
    "d2p": {
        "model_fn": lambda: D2PModel(
            freeze_clip=True,
            dropout_rate=config.D2P_DROPOUT_RATE,
        ),
        "default_ckpt": config.D2P_BEST_CKPT,
        "output_path": config.OUTPUT_DIR / "flickr30k_d2p_results.json",
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
    """Extract features for distribution-based models (dist_align, prolip, grove, d2p)."""
    model.eval()
    all_img_mu, all_text_mu = [], []
    all_img_logvar, all_text_logvar = [], []
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

        sample_count += batch_size
        if num_samples and sample_count >= num_samples:
            break

    img_mu = torch.cat(all_img_mu, dim=0)
    text_mu = torch.cat(all_text_mu, dim=0)
    img_logvar = torch.cat(all_img_logvar, dim=0)
    text_logvar = torch.cat(all_text_logvar, dim=0)

    if num_samples:
        img_mu = img_mu[:num_samples]
        text_mu = text_mu[:num_samples]
        img_logvar = img_logvar[:num_samples]
        text_logvar = text_logvar[:num_samples]

    return img_mu, text_mu, img_logvar, text_logvar


@torch.no_grad()
def extract_features_icpe(model, dataloader, device, num_samples=None):
    """
    Extract features for ICPE (training-free).

    ICPE first collects all CLIP features, then computes k-NN covariance
    as distributional representation, and finally does retrieval.
    """
    model.eval()
    all_img, all_text = [], []
    sample_count = 0

    # Phase 1: Collect raw CLIP features
    for batch in tqdm(dataloader, desc="Collecting CLIP features"):
        if batch is None:
            continue

        pil_images = batch["image"]
        caption_lists = batch["captions"]

        pixel_values = model.process_images(pil_images).to(device)

        selected_captions = [caps[0] for caps in caption_lists]
        text_inputs = model.process_text(selected_captions)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # Get CLIP features directly
        img_feat = model.encode_image(pixel_values)
        text_feat = model.encode_text(input_ids, attention_mask)

        all_img.append(F.normalize(img_feat, dim=-1).cpu())
        all_text.append(F.normalize(text_feat, dim=-1).cpu())

        sample_count += len(pil_images)
        if num_samples and sample_count >= num_samples:
            break

    img_features = torch.cat(all_img, dim=0)
    text_features = torch.cat(all_text, dim=0)

    if num_samples:
        img_features = img_features[:num_samples]
        text_features = text_features[:num_samples]

    # Phase 2: Compute ICPE covariance (k-NN based)
    logger.info("Computing ICPE k-NN covariance for image features...")
    img_logvar = model.compute_icpe_covariance(img_features)
    img_logvar = torch.log(img_logvar)

    logger.info("Computing ICPE k-NN covariance for text features...")
    text_logvar = model.compute_icpe_covariance(text_features)
    text_logvar = torch.log(text_logvar)

    return img_features, text_features, img_logvar, text_logvar


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

    sim(x,y) = mu_x . mu_y / (tau * sqrt(1 + ||sigma_x||^2) * sqrt(1 + ||sigma_y||^2))
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

    if args.model_type == "icpe":
        # ICPE: collect features, compute k-NN covariance, then retrieve
        img_mu, text_mu, img_logvar, text_logvar = extract_features_icpe(
            model, dataloader, args.device, args.num_samples,
        )

        # Standard cosine recall on raw CLIP features (ICPE mu = CLIP features)
        recall_metrics = compute_recall_chunked(
            img_mu, text_mu, args.recall_at_k, chunk_size=1000,
        )

        # Also compute UC-style recall with ICPE variance
        uc_recall = compute_recall_uc_chunked(
            img_mu, img_logvar, text_mu, text_logvar,
            args.recall_at_k, temperature=0.07, chunk_size=1000,
        )
        recall_metrics.update(uc_recall)

    elif has_distribution:
        # Distribution-based models: extract mu and logvar
        img_mu, text_mu, img_logvar, text_logvar = extract_features_distribution(
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
