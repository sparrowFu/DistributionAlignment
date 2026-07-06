"""
GaussianImageDistribution - MSDA Distribution Alignment Evaluation Script

Evaluates the MSDA (Multi-caption Semantic Distribution Alignment) model using
image-text retrieval. The forward output keys (img_mu / text_mu / img_logvar /
text_logvar) are unchanged, so retrieval logic is identical to before; the
optional uncertainty-calibrated similarity corresponds to MSDA's
uncertainty-discounted similarity.

Usage:
    python scripts/evaluate_dist_align.py
    python main.py --task eval_dist_align
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.caption_dataset import ImageCaptionDataset, filter_none_collate
from models.dist_align_model import DistributionAlignmentModel
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


logger = get_logger("eval_dist_align", config.EVAL_DIST_ALIGN_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Distribution Alignment Model")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (uses best checkpoint if None)")
    parser.add_argument("--captions-path", type=str, default=None,
                        help="Path to captions file (uses config default if None)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Path to images directory (uses config default if None)")
    parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE,
                        help="Evaluation batch size")
    parser.add_argument("--recall-at-k", type=int, nargs="+", default=config.RECALL_AT_K,
                        help="Recall@K values to compute")
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="Number of samples to evaluate (default: 5000)")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output JSON path (uses config default if None)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--use-uncertainty-sim", action="store_true",
                        default=True,
                        help="Also compute Recall@K with MSDA uncertainty-discounted similarity")
    parser.add_argument("--tau", type=float, default=config.MSDA_TAU,
                        help="Temperature for uncertainty-discounted similarity")

    return parser.parse_args()


@torch.no_grad()
def extract_features(
    model: DistributionAlignmentModel,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = None
):
    """Extract image and text distribution features (mu and logvar)."""
    model.eval()

    all_img_mu = []
    all_text_mu = []
    all_img_logvar = []
    all_text_logvar = []
    sample_count = 0

    logger.info("Extracting features...")
    for batch in tqdm(dataloader):
        if batch is None:
            continue

        # Get data - PIL images and text lists
        pil_images = batch["image"]
        caption_lists = batch["captions"]

        # Process images with CLIP processor
        pixel_values = model.process_images(pil_images).to(device)

        # Process text captions
        batch_size = len(pil_images)
        num_captions = len(caption_lists[0])

        # Flatten all captions: [B*K]
        all_captions = []
        for caption_list in caption_lists:
            all_captions.extend(caption_list)

        # Process with CLIP processor
        text_inputs = model.process_text(all_captions)

        # Reshape to [B, K, max_len]
        input_ids = text_inputs["input_ids"].view(batch_size, num_captions, -1).to(device)
        attention_mask = text_inputs["attention_mask"].view(batch_size, num_captions, -1).to(device)

        # Forward pass
        outputs = model(pixel_values, input_ids, attention_mask)

        all_img_mu.append(outputs['img_mu'].cpu())
        all_text_mu.append(outputs['text_mu'].cpu())
        all_img_logvar.append(outputs['img_logvar'].cpu())
        all_text_logvar.append(outputs['text_logvar'].cpu())

        sample_count += batch_size
        if num_samples and sample_count >= num_samples:
            break

    # Concatenate features
    img_mu = torch.cat(all_img_mu, dim=0)
    text_mu = torch.cat(all_text_mu, dim=0)
    img_logvar = torch.cat(all_img_logvar, dim=0)
    text_logvar = torch.cat(all_text_logvar, dim=0)

    if num_samples:
        img_mu = img_mu[:num_samples]
        text_mu = text_mu[:num_samples]
        img_logvar = img_logvar[:num_samples]
        text_logvar = text_logvar[:num_samples]

    logger.info(f"Features shape: Images {img_mu.shape}, Texts {text_mu.shape}")

    return img_mu, text_mu, img_logvar, text_logvar


def compute_recall_chunked(
    img_features: torch.Tensor,
    text_features: torch.Tensor,
    k_values: list,
    chunk_size: int = 1000
) -> dict:
    """Compute Recall@K in chunks to avoid OOM on large matrices."""
    import torch.nn.functional as F

    n = img_features.shape[0]
    max_k = max(k_values)

    # L2 normalize features for cosine similarity
    img_features = F.normalize(img_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Track hits for each k
    hits = {k: 0 for k in k_values}

    logger.info(f"Computing Recall@K with chunk_size={chunk_size}...")

    for start in tqdm(range(0, n, chunk_size), desc="Recall chunks"):
        end = min(start + chunk_size, n)

        # Compute partial similarity matrix: [chunk, n]
        sim_chunk = torch.matmul(img_features[start:end], text_features.T)

        # Get rankings
        ranked_indices = torch.argsort(sim_chunk, dim=1, descending=True)

        for k in k_values:
            top_k = ranked_indices[:, :k]
            gt = torch.arange(start, end).unsqueeze(1)
            is_in_top_k = (top_k == gt).any(dim=1)
            hits[k] += is_in_top_k.sum().item()

    recall_metrics = {}
    for k in k_values:
        recall_metrics[f'recall@{k}'] = hits[k] / n
        logger.info(f"Recall@{k}: {recall_metrics[f'recall@{k}']:.4f}")

    return recall_metrics


def compute_recall_uc_chunked(
    img_mu: torch.Tensor,
    img_logvar: torch.Tensor,
    text_mu: torch.Tensor,
    text_logvar: torch.Tensor,
    k_values: list,
    temperature: float = 0.07,
    chunk_size: int = 1000
) -> dict:
    """
    Compute Recall@K using uncertainty-calibrated similarity.

    sim(x,y) = μ_x · μ_y / (τ · √(1 + ‖σ_x‖²) · √(1 + ‖σ_y‖²))
    """
    import torch.nn.functional as F

    n = img_mu.shape[0]
    max_k = max(k_values)

    # Precompute per-sample scaling factors (mean, not sum, for dimension-independence)
    img_var_avg = torch.exp(img_logvar).mean(dim=-1)  # (n,)
    text_var_avg = torch.exp(text_logvar).mean(dim=-1)  # (n,)
    img_scale = torch.sqrt(1.0 + img_var_avg)  # (n,)
    text_scale = torch.sqrt(1.0 + text_var_avg)  # (n,)

    # Normalize means
    img_mu_norm = F.normalize(img_mu, dim=-1)
    text_mu_norm = F.normalize(text_mu, dim=-1)

    # Track hits for each k
    hits = {k: 0 for k in k_values}

    logger.info(f"Computing Recall@K (UC similarity, τ={temperature}) with chunk_size={chunk_size}...")

    for start in tqdm(range(0, n, chunk_size), desc="UC Recall chunks"):
        end = min(start + chunk_size, n)

        # Mean similarity: [chunk, n]
        sim_chunk = torch.matmul(img_mu_norm[start:end], text_mu_norm.T)

        # Apply uncertainty calibration
        scale_matrix = img_scale[start:end].unsqueeze(1) * text_scale.unsqueeze(0)
        sim_chunk = sim_chunk / (temperature * scale_matrix)

        # Get rankings
        ranked_indices = torch.argsort(sim_chunk, dim=1, descending=True)

        for k in k_values:
            top_k = ranked_indices[:, :k]
            gt = torch.arange(start, end).unsqueeze(1)
            is_in_top_k = (top_k == gt).any(dim=1)
            hits[k] += is_in_top_k.sum().item()

    recall_metrics = {}
    for k in k_values:
        recall_metrics[f'uc_recall@{k}'] = hits[k] / n
        logger.info(f"UC-Recall@{k}: {recall_metrics[f'uc_recall@{k}']:.4f}")

    return recall_metrics


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set random seed
    set_seed(config.SEED)

    # Load model
    checkpoint_path = args.checkpoint or str(config.DIST_ALIGN_BEST_CKPT)
    logger.info(f"Loading model from {checkpoint_path}")

    model = DistributionAlignmentModel(
        freeze_clip=config.DIST_ALIGN_FREEZE_CLIP,
        distribution_merging=config.DIST_ALIGN_DISTRIBUTION_MERGING,
        cov_rank=config.MSDA_COV_RANK,
    )

    model.load(checkpoint_path)
    model = model.to(args.device)

    # Load dataset
    captions_path = args.captions_path or config.CAPTIONS_PATH
    images_dir = args.images_dir or config.IMAGES_DIR

    dataset = ImageCaptionDataset(
        captions_path=captions_path,
        images_dir=images_dir,
        num_captions=config.NUM_CAPTIONS
    )

    # Use subset for evaluation
    num_samples = args.num_samples
    if num_samples and num_samples < len(dataset):
        # Use fixed seed for reproducible subset
        generator = torch.Generator().manual_seed(config.SEED)
        indices = torch.randperm(len(dataset), generator=generator)[:num_samples].tolist()
        dataset = Subset(dataset, indices)
        logger.info(f"Using {num_samples} samples (random subset)")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=filter_none_collate
    )

    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Extract features (mu and logvar)
    img_mu, text_mu, img_logvar, text_logvar = extract_features(
        model, dataloader, args.device, args.num_samples
    )

    # Compute Recall@K using standard cosine similarity on mu
    recall_metrics = compute_recall_chunked(
        img_mu, text_mu, args.recall_at_k, chunk_size=1000
    )

    # Optionally also compute Recall@K using uncertainty-discounted similarity
    if args.use_uncertainty_sim:
        uc_recall_metrics = compute_recall_uc_chunked(
            img_mu, img_logvar, text_mu, text_logvar,
            args.recall_at_k, temperature=args.tau, chunk_size=1000
        )
        recall_metrics.update(uc_recall_metrics)

    # Save results
    output_path = args.output_path or str(config.DIST_ALIGN_EVAL_RESULTS_PATH)
    results = {
        'checkpoint': str(checkpoint_path),
        'num_samples': len(dataset) if not args.num_samples else args.num_samples,
        'metrics': recall_metrics
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Evaluation failed")
        raise
