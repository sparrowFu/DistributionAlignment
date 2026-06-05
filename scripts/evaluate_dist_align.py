"""
GaussianImageDistribution - Distribution Alignment Evaluation Script

This script evaluates the distribution alignment model using image-text retrieval.

Usage:
    python scripts/evaluate_dist_align.py
    python main.py --task eval_dist_align
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.caption_dataset import ImageCaptionDataset
from models.dist_align_model import DistributionAlignmentModel
from losses.dist_align_losses import DistributionAlignmentLoss, CombinedDistributionLoss
from utils.logger import get_logger, log_exception
from utils.seed import set_seed
from utils.metrics import compute_recall_at_k


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
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to evaluate (None for all)")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output JSON path (uses config default if None)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    return parser.parse_args()


def filter_none_collate(batch):
    """Collate function that filters out None values."""
    filtered = [item for item in batch if item is not None]
    if not filtered:
        return None

    return {
        "image": [item["image"] for item in filtered],
        "captions": [item["captions"] for item in filtered],
    }


@torch.no_grad()
def evaluate(
    model: DistributionAlignmentModel,
    dataloader: DataLoader,
    recall_at_k: list,
    device: torch.device,
    num_samples: int = None
):
    """Evaluate model and compute Recall@K metrics."""
    model.eval()

    all_img_features = []
    all_text_features = []

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

        all_img_features.append(outputs['img_mu'].cpu())  # Use distribution mean
        all_text_features.append(outputs['text_mu'].cpu())

        if num_samples and len(all_img_features) * dataloader.batch_size >= num_samples:
            break

    # Concatenate features
    img_features = torch.cat(all_img_features, dim=0)
    text_features = torch.cat(all_text_features, dim=0)

    if num_samples:
        img_features = img_features[:num_samples]
        text_features = text_features[:num_samples]

    logger.info(f"Features shape: Images {img_features.shape}, Texts {text_features.shape}")

    # Compute similarity matrix
    similarity_matrix = torch.matmul(img_features, text_features.T)

    # Compute Recall@K
    logger.info("Computing Recall@K...")
    recall_metrics = {}
    for k in recall_at_k:
        recall_at_k_value = compute_recall_at_k(similarity_matrix, k)
        recall_metrics[f'recall@{k}'] = recall_at_k_value
        logger.info(f"Recall@{k}: {recall_at_k_value:.4f}")

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
        distribution_merging=config.DIST_ALIGN_DISTRIBUTION_MERGING
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

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=filter_none_collate
    )

    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Evaluate
    metrics = evaluate(
        model, dataloader, args.recall_at_k,
        args.device, args.num_samples
    )

    # Save results
    output_path = args.output_path or str(config.DIST_ALIGN_EVAL_RESULTS_PATH)
    results = {
        'checkpoint': str(checkpoint_path),
        'num_samples': len(dataset) if not args.num_samples else args.num_samples,
        'metrics': metrics
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
