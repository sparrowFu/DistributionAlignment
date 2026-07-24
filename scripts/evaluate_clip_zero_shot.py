"""
GaussianImageDistribution - CLIP Zero-Shot Evaluation Script

This script evaluates the CLIP Zero-Shot baseline using image-text retrieval
with Recall@K metrics. No training required - uses frozen CLIP directly.
Uses a random subset of samples for efficiency.

Usage:
    python scripts/evaluate_clip_zero_shot.py
    python main.py --task eval_clip_zero_shot
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from models.clip_baseline import CLIPFineTuneBaseline
from utils.eval_common import build_eval_dataloader, VALID_DATASETS
from utils.eval_results import append_eval_results, groups_to_flat, print_recall_groups
from utils.logger import get_logger, log_exception
from utils.retrieval import compute_recall_bidirectional
from utils.seed import set_seed


logger = get_logger("eval_clip_zero_shot", config.EVAL_CLIP_ZERO_SHOT_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate CLIP Zero-Shot Baseline")

    parser.add_argument("--dataset", type=str, default="coco",
                        choices=list(VALID_DATASETS),
                        help="Dataset to evaluate on (coco=MSCOCO, flickr=flickr30k). Zero-shot "
                             "uses no checkpoint, so --dataset only selects the eval data. "
                             "Default: coco")
    parser.add_argument("--captions-path", type=str, default=None,
                        help="Path to captions file (coco only; overrides config default if set)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Path to images directory (coco only; overrides config default if set)")
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
def extract_features(
    model: CLIPFineTuneBaseline,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = None,
):
    """Extract normalized CLIP image and text features (no training)."""
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

        selected_captions = [captions[0] for captions in caption_lists]
        text_inputs = model.process_text(selected_captions)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # Extract normalized features from frozen CLIP
        img_feat, text_feat = model(
            images=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            normalize=True,
        )

        all_img_features.append(img_feat.cpu())
        all_text_features.append(text_feat.cpu())

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


def main():
    """Main evaluation function."""
    args = parse_args()
    set_seed(config.SEED)

    # Use frozen CLIP baseline (no checkpoint needed)
    logger.info("Loading frozen CLIP model (zero-shot, no training)...")
    model = CLIPFineTuneBaseline(
        freeze_image=True,
        freeze_text=True,
    )
    model = model.to(args.device)

    # Load dataset (selected by --dataset: coco=MSCOCO, flickr=flickr30k test).
    # Zero-shot uses frozen CLIP, so --dataset only selects the eval data.
    dataloader, num_eval_samples = build_eval_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
        num_samples=args.num_samples,
        captions_path=args.captions_path,
        images_dir=args.images_dir,
    )
    logger.info(f"Dataset loaded ({args.dataset}): {num_eval_samples} samples")

    # Extract features
    img_features, text_features = extract_features(
        model, dataloader, args.device, args.num_samples,
    )

    # Compute bidirectional Recall@K (image->text and text->image, cosine)
    bidir = compute_recall_bidirectional(
        img_features, text_features, args.recall_at_k, chunk_size=1000, normalize=True)
    groups = [{
        "family": "recall",
        "label": "Recall@K",
        "per_k": {
            k: {
                "i2t": bidir[f"recall_i2t@{k}"],
                "t2i": bidir[f"recall_t2i@{k}"],
                "mean": bidir[f"recall@{k}"],
            }
            for k in args.recall_at_k
        },
    }]
    print_recall_groups(groups, logger)

    # Append results (never overwrite prior runs); time is stamped after dataset.
    output_path = args.output_path or str(config.CLIP_ZERO_SHOT_EVAL_RESULTS_PATH)
    append_eval_results(output_path, {
        "model": "CLIP ViT-L/14 (zero-shot, frozen)",
        "dataset": args.dataset,
        "num_samples": num_eval_samples,
        "metrics": groups_to_flat(groups),
    }, logger)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Evaluation failed")
        raise
