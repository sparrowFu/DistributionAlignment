"""
CLIP Zero-Shot Evaluation Script

This script evaluates the CLIP Zero-Shot baseline using image-text retrieval
with Recall@K metrics. No training required - uses frozen CLIP directly.
Uses a random subset of samples for efficiency.
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
from utils.retrieval import compute_recall_bidirectional, compute_multicaption_recall
from utils.seed import set_seed


logger = get_logger("eval_clip_zero_shot", config.EVAL_CLIP_ZERO_SHOT_LOG_PATH)


def parse_args():
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
    all_cap_features = []  # per-batch (B, K, D)
    sample_count = 0

    logger.info("Extracting features (all K captions per image)...")
    for batch in tqdm(dataloader):
        if batch is None:
            continue

        pil_images = batch["image"]
        caption_lists = batch["captions"]
        B = len(pil_images)
        K = len(caption_lists[0])

        all_captions = []
        for captions in caption_lists:
            all_captions.extend(captions)

        pixel_values = model.process_images(pil_images).to(device)

        text_inputs = model.process_text(all_captions)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        img_feat, text_feat = model(
            images=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            normalize=True,
        )

        all_img_features.append(img_feat.cpu())
        all_cap_features.append(text_feat.cpu().view(B, K, -1))

        sample_count += len(pil_images)
        if num_samples and sample_count >= num_samples:
            break

    img_features = torch.cat(all_img_features, dim=0)
    text_mus = torch.cat(all_cap_features, dim=0)          # (N, K, D)

    if num_samples:
        img_features = img_features[:num_samples]
        text_mus = text_mus[:num_samples]

    logger.info(f"Features shape: Images {img_features.shape}, Captions {text_mus.shape}")
    return img_features, text_mus


def main():
    args = parse_args()
    set_seed(config.SEED)

    # Use frozen CLIP baseline (no checkpoint needed)
    logger.info("Loading frozen CLIP model (zero-shot, no training)...")
    model = CLIPFineTuneBaseline(
        freeze_image=True,
        freeze_text=True,
    )
    model = model.to(args.device)

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

    img_features, text_mus = extract_features(
        model, dataloader, args.device, args.num_samples,
    )

    groups = []

    # --- Protocol A (legacy): 1:1 retrieval against the FIRST caption. ------
    bidir = compute_recall_bidirectional(
        img_features, text_mus[:, 0], args.recall_at_k,
        chunk_size=1000, normalize=True)
    groups.append({
        "family": "recall",
        "label": "Recall@K (legacy 1:1, first caption only)",
        "per_k": {
            k: {
                "i2t": bidir[f"recall_i2t@{k}"],
                "t2i": bidir[f"recall_t2i@{k}"],
                "mean": bidir[f"recall@{k}"],
            }
            for k in args.recall_at_k
        },
    })

    # --- Protocol B (unified multi-caption): same protocol as the
    # fine-tuned baselines' mc_cos_recall rows. No variance heads -> zero
    # logvars make the discounted score rank-identical to cosine. ---------
    dev = torch.device(args.device)
    mc = compute_multicaption_recall(
        img_features.to(dev), torch.zeros_like(img_features).to(dev),
        text_mus.to(dev), torch.zeros_like(text_mus).to(dev),
        args.recall_at_k,
    )
    groups.append({
        "family": "mc_cos_recall",
        "label": "Multi-caption Recall@K (unified protocol, cosine)",
        "per_k": {
            k: {
                "i2t": mc[f"mc_cos_recall_i2t@{k}"],
                "t2i": mc[f"mc_cos_recall_t2i@{k}"],
                "mean": mc[f"mc_cos_recall@{k}"],
            }
            for k in args.recall_at_k
        },
    })
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
