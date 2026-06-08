"""
GaussianImageDistribution - Freeze-Align Evaluation Script

This script evaluates the Freeze-Align model using image-text retrieval
with Recall@K metrics. Uses a random subset of samples for efficiency.

Usage:
    python scripts/evaluate_freeze_align.py
    python main.py --task eval_freeze_align
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.caption_dataset import ImageCaptionDataset
from models.freeze_align_model import FreezeAlignModel
from utils.logger import get_logger, log_exception
from utils.seed import set_seed
from utils.metrics import compute_recall_at_k


logger = get_logger("eval_freeze_align", config.EVAL_FREEZE_ALIGN_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Freeze-Align Model")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (uses best checkpoint if None)")
    parser.add_argument("--captions-path", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    parser.add_argument("--recall-at-k", type=int, nargs="+", default=config.RECALL_AT_K)
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="Number of samples to evaluate (default: 5000)")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

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
def extract_features(
    model: FreezeAlignModel,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = None,
):
    """Extract projected image and text features."""
    model.eval()

    all_img_features = []
    all_text_features = []
    sample_count = 0

    logger.info("Extracting features...")
    for batch in tqdm(dataloader):
        if batch is None:
            continue

        pil_images = batch["image"]
        caption_lists = batch["captions"]

        pixel_values = model.process_images(pil_images).to(device)

        # Use first caption per image for retrieval evaluation
        selected_captions = [captions[0] for captions in caption_lists]
        text_inputs = model.process_text(selected_captions)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # Forward: get projected features
        outputs = model(pixel_values, input_ids, attention_mask)

        # Normalize projected features
        proj_img = F.normalize(outputs["proj_img_features"], dim=-1)
        proj_text = F.normalize(outputs["proj_text_features"], dim=-1)

        all_img_features.append(proj_img.cpu())
        all_text_features.append(proj_text.cpu())

        sample_count += len(pil_images)
        if num_samples and sample_count >= num_samples:
            break

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
    chunk_size: int = 1000,
) -> dict:
    """Compute Recall@K in chunks to avoid OOM."""
    n = img_features.shape[0]
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
        logger.info(f"Recall@{k}: {recall_metrics[f'recall@{k}']:.4f}")

    return recall_metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    set_seed(config.SEED)

    # Load model
    checkpoint_path = args.checkpoint or str(config.FREEZE_ALIGN_BEST_CKPT)
    logger.info(f"Loading model from {checkpoint_path}")

    model = FreezeAlignModel(
        proj_dim=config.FREEZE_ALIGN_PROJ_DIM,
        dropout_rate=config.FREEZE_ALIGN_DROPOUT_RATE,
    )
    model.load(checkpoint_path)
    model = model.to(args.device)

    # Load dataset
    captions_path = args.captions_path or config.CAPTIONS_PATH
    images_dir = args.images_dir or config.IMAGES_DIR

    dataset = ImageCaptionDataset(
        captions_path=captions_path,
        images_dir=images_dir,
        num_captions=config.NUM_CAPTIONS,
    )

    # Use subset for evaluation
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
    img_features, text_features = extract_features(
        model, dataloader, args.device, args.num_samples,
    )

    # Compute Recall@K
    recall_metrics = compute_recall_chunked(
        img_features, text_features, args.recall_at_k, chunk_size=1000,
    )

    # Save results
    output_path = args.output_path or str(config.FREEZE_ALIGN_EVAL_RESULTS_PATH)
    results = {
        "checkpoint": str(checkpoint_path),
        "num_samples": args.num_samples,
        "metrics": recall_metrics,
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
