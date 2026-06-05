"""
GaussianImageDistribution - CLIP Baseline Evaluation Script

This script evaluates the trained CLIP baseline model on image-text retrieval
tasks, computing Recall@K metrics.

Usage:
    python scripts/evaluate_clip_baseline.py
    python main.py --task eval_clip_baseline
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.caption_dataset import ImageCaptionDataset
from models.clip_baseline import CLIPFineTuneBaseline
from utils.metrics import compute_bidirectional_recall, format_recall_results
from utils.logger import get_logger, log_exception
from utils.seed import set_seed
from utils.io_utils import save_json


# Setup logger
logger = get_logger("eval_clip_baseline", config.EVAL_CLIP_BASELINE_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate CLIP Baseline Model")

    # Data arguments
    parser.add_argument("--captions-path", type=str, default=None,
                        help="Path to captions parquet file (uses config default if None)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Path to images directory (uses config default if None)")

    # Model arguments
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (uses best checkpoint if None)")
    parser.add_argument("--freeze-image", action="store_true",
                        help="Freeze image encoder")
    parser.add_argument("--freeze-text", action="store_true",
                        help="Freeze text encoder")

    # Evaluation arguments
    parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE,
                        help="Evaluation batch size")
    parser.add_argument("--recall-at-k", type=int, nargs="+", default=config.RECALL_AT_K,
                        help="Recall@K values to compute")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to evaluate (None for all)")

    # System arguments
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    # Output arguments
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output JSON path (uses config default if None)")

    return parser.parse_args()


def extract_features(
    model: CLIPFineTuneBaseline,
    dataloader: DataLoader,
    device: str,
    max_samples: int = None
):
    """Extract image and text features from the dataset."""
    model.eval()

    all_image_features = []
    all_text_features = []
    all_image_names = []
    sample_count = 0

    pbar = tqdm(dataloader, desc="Extracting features")

    with torch.no_grad():
        for batch in pbar:
            if batch is None:
                continue

            if max_samples is not None and sample_count >= max_samples:
                break

            # Get data - PIL images and text lists
            pil_images = batch["image"]
            captions_list = batch["captions"]
            image_names = batch["image_name"]

            # Use first caption for evaluation consistency
            selected_captions = [captions[0] for captions in captions_list]

            # Process with CLIP processor
            pixel_values = model.process_images(pil_images).to(device)
            text_inputs = model.process_text(selected_captions)
            input_ids = text_inputs["input_ids"].to(device)
            attention_mask = text_inputs["attention_mask"].to(device)

            # Forward pass
            image_features, text_features = model(
                images=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
            all_image_names.extend(image_names)

            sample_count += len(pil_images)
            pbar.set_postfix({"samples": sample_count})

    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)

    logger.info(f"Extracted features for {len(all_image_names)} samples")
    logger.info(f"Image features shape: {all_image_features.shape}")
    logger.info(f"Text features shape: {all_text_features.shape}")

    return all_image_features, all_text_features, all_image_names


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Parse paths
    captions_path = args.captions_path or config.CAPTIONS_PATH
    images_dir = args.images_dir or config.IMAGES_DIR
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else config.CLIP_BASELINE_BEST_CKPT
    output_path = Path(args.output_path) if args.output_path else config.CLIP_BASELINE_EVAL_RESULTS_PATH

    # Log configuration
    logger.info("=" * 60)
    logger.info("CLIP Baseline Evaluation")
    logger.info("=" * 60)
    logger.info(f"Device: {args.device}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Captions: {captions_path}")
    logger.info(f"Images: {images_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Recall@K: {args.recall_at_k}")
    logger.info(f"Num samples: {args.num_samples or 'All'}")
    logger.info("=" * 60)

    # Check checkpoint exists
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Please train the model first: python main.py --task train_clip_baseline")
        return

    # Create model and load checkpoint
    logger.info("Creating model...")
    model = CLIPFineTuneBaseline(
        freeze_image=args.freeze_image,
        freeze_text=args.freeze_text
    )
    model = model.to(args.device)

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    try:
        model.load(str(checkpoint_path))
    except Exception as e:
        log_exception(logger, e, "Failed to load checkpoint")
        return

    # Load dataset
    logger.info("Loading dataset...")
    dataset = ImageCaptionDataset(
        captions_path=captions_path,
        images_dir=images_dir,
        num_captions=config.NUM_CAPTIONS
    )

    # Limit samples if specified
    if args.num_samples is not None and args.num_samples < len(dataset):
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(args.num_samples)))
        logger.info(f"Using {args.num_samples} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=filter_none_collate
    )

    # Extract features
    logger.info("Extracting features...")
    start_time = time.time()

    image_features, text_features, image_names = extract_features(
        model, dataloader, args.device, args.num_samples
    )

    extract_time = time.time() - start_time
    logger.info(f"Feature extraction completed in {extract_time:.2f}s")

    # Compute recall metrics
    logger.info("Computing recall metrics...")
    results = compute_bidirectional_recall(
        image_features=image_features,
        text_features=text_features,
        k_values=args.recall_at_k
    )

    logger.info("\n" + format_recall_results(results))

    # Save results
    json_results = {
        "image_to_text": {
            f"recall@{k}": results["image_to_text"][k]
            for k in sorted(results["image_to_text"].keys())
        },
        "text_to_image": {
            f"recall@{k}": results["text_to_image"][k]
            for k in sorted(results["text_to_image"].keys())
        }
    }

    save_json(json_results, output_path)
    logger.info(f"Results saved to: {output_path}")

    logger.info("=" * 60)
    logger.info("Evaluation completed!")
    logger.info(f"Total samples: {len(image_names)}")
    logger.info("=" * 60)


def filter_none_collate(batch):
    """Collate function that filters out None values."""
    filtered = [item for item in batch if item is not None]
    if not filtered:
        return None

    return {
        "image": [item["image"] for item in filtered],
        "captions": [item["captions"] for item in filtered],
        "image_path": [item["image_path"] for item in filtered],
        "image_name": [item["image_name"] for item in filtered],
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Evaluation failed")
        raise
