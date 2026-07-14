"""
GaussianImageDistribution - ProLIP Zero-Shot Evaluation Script

Evaluates the ProLIP Zero-Shot baseline using image-text retrieval with
Recall@K. No training required -- uses frozen pretrained ProLIP directly.
Reports both directions (I2T and T2I) under cosine and ProLIP's uncertainty-aware
CSD similarity. Uses a random subset of samples for efficiency.

Usage:
    python scripts/evaluate_prolip_zero_shot.py
    python main.py --task eval_prolip_zero_shot
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
from models.prolip_model import ProLIPModel
from utils.logger import get_logger, log_exception
from utils.retrieval_metrics import compute_retrieval_metrics
from utils.seed import set_seed


logger = get_logger("eval_prolip_zero_shot", config.EVAL_PROLIP_ZERO_SHOT_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate ProLIP Zero-Shot Baseline")

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
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    return parser.parse_args()


@torch.no_grad()
def extract_features(model, dataloader, device, num_samples=None):
    """Extract ProLIP mean and log-variance features (no training)."""
    model.eval()

    all_img_mu, all_text_mu = [], []
    all_img_logvar, all_text_logvar = [], []
    sample_count = 0

    logger.info("Extracting features...")
    for batch in tqdm(dataloader, desc="Extracting features"):
        if batch is None:
            continue

        pil_images = batch["image"]
        caption_lists = batch["captions"]

        pixel_values = model.process_images(pil_images)

        # Use first caption per image (1:1 retrieval, diagonal ground truth)
        selected_captions = [captions[0] for captions in caption_lists]
        text_inputs = model.process_text(selected_captions)
        input_ids = text_inputs["input_ids"].to(device)

        outputs = model(pixel_values, input_ids)

        all_img_mu.append(outputs["img_mu"].cpu())
        all_text_mu.append(outputs["text_mu"].cpu())
        all_img_logvar.append(outputs["img_logvar"].cpu())
        all_text_logvar.append(outputs["text_logvar"].cpu())

        sample_count += len(pil_images)
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

    logger.info(f"Features shape: Images {img_mu.shape}, Texts {text_mu.shape}")
    return img_mu, img_logvar, text_mu, text_logvar


def main():
    """Main evaluation function."""
    args = parse_args()
    set_seed(config.SEED)

    # Frozen pretrained ProLIP (zero-shot, no checkpoint)
    logger.info("Loading frozen ProLIP model (zero-shot, no training)...")
    model = ProLIPModel(freeze=True)
    model = model.to(args.device)
    logger.info(f"Trainable parameters: {model.num_trainable_parameters():,} (expect 0)")

    # Load dataset
    captions_path = args.captions_path or config.CAPTIONS_PATH
    images_dir = args.images_dir or config.IMAGES_DIR

    dataset = ImageCaptionDataset(
        captions_path=captions_path,
        images_dir=images_dir,
        num_captions=config.NUM_CAPTIONS,
    )

    num_samples = args.num_samples
    if num_samples and num_samples < len(dataset):
        generator = torch.Generator().manual_seed(config.SEED)
        indices = torch.randperm(len(dataset), generator=generator)[:num_samples].tolist()
        dataset = Subset(dataset, indices)
        logger.info(f"Using {num_samples} samples (random subset)")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=filter_none_collate,
    )
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Extract features
    img_mu, img_logvar, text_mu, text_logvar = extract_features(
        model, dataloader, args.device, args.num_samples,
    )

    # Compute Recall@K (I2T + T2I, cosine + CSD)
    metrics = compute_retrieval_metrics(
        img_mu, img_logvar, text_mu, text_logvar,
        args.recall_at_k, chunk_size=1000,
    )

    for direction, by_metric in metrics.items():
        for metric_name, recalls in by_metric.items():
            for k, v in recalls.items():
                logger.info(f"{direction}/{metric_name}/{k}: {v:.4f}")

    # Save results
    output_path = args.output_path or str(config.PROLIP_ZERO_SHOT_EVAL_RESULTS_PATH)
    results = {
        "model": "ProLIP ViT-H/14 (zero-shot, frozen)",
        "num_samples": args.num_samples,
        "metrics": metrics,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Evaluation failed")
        raise
