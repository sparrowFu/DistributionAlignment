"""
GaussianImageDistribution - CLIP Baseline Evaluation Script

This script evaluates the CLIP baseline model using image-text retrieval.

Usage:
    python scripts/evaluate_clip_baseline.py
    python main.py --task eval_clip_baseline
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
from utils.eval_common import build_eval_dataloader, resolve_checkpoint, VALID_DATASETS
from utils.eval_results import append_eval_results, groups_to_flat, print_recall_groups
from utils.logger import get_logger, log_exception
from utils.retrieval import compute_recall_bidirectional
from utils.seed import set_seed


logger = get_logger("eval_clip_baseline", config.EVAL_CLIP_BASELINE_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate CLIP Baseline Model")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. If None, auto-selects "
                             "clip_baseline_{dataset}_best.pt from --dataset "
                             "(pass a ..._last.pt path to evaluate the last checkpoint).")
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=list(VALID_DATASETS),
                        help="Dataset to evaluate on and to auto-select the checkpoint for "
                             "(coco=MSCOCO, flickr=flickr30k). Default: coco")
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


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set random seed
    set_seed(config.SEED)

    # Load model (auto-select checkpoint by dataset: clip_baseline_{dataset}_best.pt)
    checkpoint_path = args.checkpoint or str(resolve_checkpoint("clip_baseline", args.dataset))
    logger.info(f"Loading model from {checkpoint_path}")

    model = CLIPFineTuneBaseline()

    model.load(checkpoint_path)
    model = model.to(args.device)

    # Load dataset (auto-selected by --dataset: coco=MSCOCO, flickr=flickr30k test)
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
        model, dataloader, args.device, args.num_samples
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
    output_path = args.output_path or str(config.CLIP_BASELINE_EVAL_RESULTS_PATH)
    append_eval_results(output_path, {
        'checkpoint': str(checkpoint_path),
        'dataset': args.dataset,
        'num_samples': num_eval_samples,
        'metrics': groups_to_flat(groups),
    }, logger)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Evaluation failed")
        raise
