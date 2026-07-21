"""
GaussianImageDistribution - MSDA Distribution Alignment Evaluation Script

Evaluates the MSDA (Multi-caption Semantic Distribution Alignment) model using
image-text retrieval under two scorers:

  - MSDA score (primary): the uncertainty-discounted cosine
        sim = (mu_v . mu_t) / (tau * sqrt(1+mean sigma_v^2) * sqrt(1+mean sigma_t^2))
    i.e. the same score the L_set contrastive loss optimizes.
  - Cosine-on-means (secondary): the methodology's "mean-only retrieval" mode.

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
from utils.retrieval import (
    compute_recall_bidirectional,
    compute_recall_msda_chunked,
)
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
    parser.add_argument("--tau", type=float, default=config.MSDA_TAU,
                        help="Temperature for the MSDA uncertainty-discounted similarity")

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

    img_mu_d = img_mu.to(args.device)
    text_mu_d = text_mu.to(args.device)
    img_lv_d = img_logvar.to(args.device)
    text_lv_d = text_logvar.to(args.device)

    # Primary: MSDA uncertainty-discounted cosine (= L_set score)
    msda_metrics = compute_recall_msda_chunked(
        img_mu_d, img_lv_d, text_mu_d, text_lv_d, args.recall_at_k, tau=args.tau)
    logger.info("MSDA-score Recall@K (primary):")
    for k in args.recall_at_k:
        logger.info(f"  R@{k}: i2t={msda_metrics[f'msda_recall_i2t@{k}']:.4f} "
                    f"t2i={msda_metrics[f'msda_recall_t2i@{k}']:.4f} "
                    f"mean={msda_metrics[f'msda_recall@{k}']:.4f}")

    # Secondary: cosine-on-means (methodology's mean-only retrieval mode)
    cos = compute_recall_bidirectional(img_mu_d, text_mu_d, args.recall_at_k, normalize=True)
    cos_metrics = {}
    for k in args.recall_at_k:
        cos_metrics[f"cos_recall_i2t@{k}"] = cos[f"recall_i2t@{k}"]
        cos_metrics[f"cos_recall_t2i@{k}"] = cos[f"recall_t2i@{k}"]
        cos_metrics[f"cos_recall@{k}"] = (cos[f"recall_i2t@{k}"] + cos[f"recall_t2i@{k}"]) / 2
    logger.info("Cosine-on-means Recall@K (mean-only mode):")
    for k in args.recall_at_k:
        logger.info(f"  R@{k}: i2t={cos_metrics[f'cos_recall_i2t@{k}']:.4f} "
                    f"t2i={cos_metrics[f'cos_recall_t2i@{k}']:.4f} "
                    f"mean={cos_metrics[f'cos_recall@{k}']:.4f}")

    recall_metrics = {**msda_metrics, **cos_metrics}

    # Save results
    output_path = args.output_path or str(config.DIST_ALIGN_EVAL_RESULTS_PATH)
    results = {
        'checkpoint': str(checkpoint_path),
        'num_samples': len(dataset) if not args.num_samples else args.num_samples,
        'tau': args.tau,
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
