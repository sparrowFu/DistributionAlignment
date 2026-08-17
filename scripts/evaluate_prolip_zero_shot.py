"""
ProLIP Zero-Shot Evaluation Script

Evaluates the ProLIP Zero-Shot baseline using image-text retrieval with
Recall@K. No training required -- uses frozen pretrained ProLIP directly.
Reports both directions (I2T and T2I) under cosine and ProLIP's uncertainty-aware
CSD similarity. Uses a random subset of samples for efficiency.
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from models.prolip_model import ProLIPModel
from utils.eval_common import build_eval_dataloader, VALID_DATASETS
from utils.eval_results import append_eval_results, groups_to_flat, print_recall_groups
from utils.logger import get_logger, log_exception
from utils.retrieval_metrics import compute_retrieval_metrics
from utils.seed import set_seed


logger = get_logger("eval_prolip_zero_shot", config.EVAL_PROLIP_ZERO_SHOT_LOG_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ProLIP Zero-Shot Baseline")

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
    args = parse_args()
    set_seed(config.SEED)

    # Frozen pretrained ProLIP (zero-shot, no checkpoint)
    logger.info("Loading frozen ProLIP model (zero-shot, no training)...")
    model = ProLIPModel(freeze=True)
    model = model.to(args.device)
    logger.info(f"Trainable parameters: {model.num_trainable_parameters():,} (expect 0)")

    # Zero-shot uses frozen ProLIP, so --dataset only selects the eval data.
    dataloader, num_eval_samples = build_eval_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
        num_samples=args.num_samples,
        captions_path=args.captions_path,
        images_dir=args.images_dir,
    )
    logger.info(f"Dataset loaded ({args.dataset}): {num_eval_samples} samples")

    img_mu, img_logvar, text_mu, text_logvar = extract_features(
        model, dataloader, args.device, args.num_samples,
    )

    # Compute Recall@K (I2T + T2I, cosine + CSD)
    metrics = compute_retrieval_metrics(
        img_mu, img_logvar, text_mu, text_logvar,
        args.recall_at_k, chunk_size=1000,
    )

    # Group into cosine / csd families (each with i2t, t2i, mean) for unified
    # printing and a flat metrics dict.
    groups = []
    for metric_name, label in [("cosine", "Cosine Recall@K"), ("csd", "CSD Recall@K")]:
        per_k = {}
        for k in args.recall_at_k:
            i2t = metrics["i2t"][metric_name][f"recall@{k}"]
            t2i = metrics["t2i"][metric_name][f"recall@{k}"]
            per_k[k] = {"i2t": i2t, "t2i": t2i, "mean": (i2t + t2i) / 2}
        groups.append({"family": f"{metric_name}_recall", "label": label, "per_k": per_k})

    print_recall_groups(groups, logger)

    # Append results (never overwrite prior runs); time is stamped after dataset.
    output_path = args.output_path or str(config.PROLIP_ZERO_SHOT_EVAL_RESULTS_PATH)
    append_eval_results(output_path, {
        "model": "ProLIP ViT-H/14 (zero-shot, frozen)",
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
