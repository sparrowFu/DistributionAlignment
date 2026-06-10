"""
GaussianImageDistribution - CLIP Baseline Evaluation Script

This script evaluates the CLIP baseline model using image-text retrieval.

Usage:
    python scripts/evaluate_clip_baseline.py
    python main.py --task eval_clip_baseline
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
from models.clip_baseline import CLIPFineTuneBaseline
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


logger = get_logger("eval_clip_baseline", config.EVAL_CLIP_BASELINE_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate CLIP Baseline Model")

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

    return parser.parse_args()


@torch.no_grad()
def extract_features(
    model: CLIPFineTuneBaseline,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = None
):
    """Extract image and text features."""
    model.eval()

    all_img_features = []
    all_text_features = []
    sample_count = 0

    logger.info("Extracting features...")
    for batch in tqdm(dataloader):
        if batch is None:
            continue

        # Get data - PIL images and text lists
        pil_images = batch["image"]
        caption_lists = batch["captions"]

        # Use first caption for evaluation consistency
        selected_captions = [captions[0] for captions in caption_lists]

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

        all_img_features.append(image_features.cpu())
        all_text_features.append(text_features.cpu())

        sample_count += len(pil_images)
        if num_samples and sample_count >= num_samples:
            break

    # Concatenate features
    img_features = torch.cat(all_img_features, dim=0)
    text_features = torch.cat(all_text_features, dim=0)

    if num_samples:
        img_features = img_features[:num_samples]
        text_features = text_features[:num_samples]

    logger.info(f"Features shape: Images {img_features.shape}, Texts {text_features.shape}")

    return img_features, text_features


def compute_recall_chunked(
    img_features: torch.Tensor,
    text_features: torch.Tensor,
    k_values: list,
    chunk_size: int = 1000
) -> dict:
    """Compute Recall@K in chunks to avoid OOM on large matrices."""
    n = img_features.shape[0]

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


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set random seed
    set_seed(config.SEED)

    # Load model
    checkpoint_path = args.checkpoint or str(config.CLIP_BASELINE_BEST_CKPT)
    logger.info(f"Loading model from {checkpoint_path}")

    model = CLIPFineTuneBaseline()

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

    # Extract features
    img_features, text_features = extract_features(
        model, dataloader, args.device, args.num_samples
    )

    # Compute Recall@K (chunked to avoid OOM)
    recall_metrics = compute_recall_chunked(
        img_features, text_features, args.recall_at_k, chunk_size=1000
    )

    # Save results
    output_path = args.output_path or str(config.CLIP_BASELINE_EVAL_RESULTS_PATH)
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
