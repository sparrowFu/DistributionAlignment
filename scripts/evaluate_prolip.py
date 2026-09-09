"""
ProLIP Fine-tuned Evaluation Script

Evaluates the fine-tuned ProLIP model using image-text retrieval with Recall@K.
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
from utils.eval_common import build_eval_dataloader, resolve_checkpoint, VALID_DATASETS
from utils.eval_results import append_eval_results, groups_to_flat, print_recall_groups
from utils.logger import get_logger, log_exception
from utils.retrieval_metrics import compute_retrieval_metrics
from utils.retrieval import compute_multicaption_recall
from utils.seed import set_seed


logger = get_logger("eval_prolip", config.EVAL_PROLIP_LOG_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ProLIP (Fine-tuned)")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. If None, auto-selects "
                             "prolip_{dataset}_best.pt from --dataset "
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
def extract_features(model, dataloader, device, num_samples=None):
    """Extract ProLIP mean and log-variance features."""
    model.eval()

    all_img_mu, all_text_mus = [], []
    all_img_logvar, all_text_logvars = [], []
    sample_count = 0

    logger.info("Extracting features...")
    for batch in tqdm(dataloader, desc="Extracting features"):
        if batch is None:
            continue

        pil_images = batch["image"]
        caption_lists = batch["captions"]
        B = len(pil_images)
        K = len(caption_lists[0])

        all_captions = []
        for captions in caption_lists:
            all_captions.extend(captions)

        pixel_values = model.process_images(pil_images)

        # Encode ALL K captions; per-caption means go into the (N, K, D)
        # tensor for the unified multi-caption protocol.
        text_inputs = model.process_text(all_captions)
        input_ids = text_inputs["input_ids"].to(device)

        outputs = model(pixel_values, input_ids)          # text_* are (B*K, D)

        all_img_mu.append(outputs["img_mu"].cpu())
        all_text_mus.append(outputs["text_mu"].cpu().view(B, K, -1))
        all_img_logvar.append(outputs["img_logvar"].cpu())
        all_text_logvars.append(outputs["text_logvar"].cpu().view(B, K, -1))

        sample_count += len(pil_images)
        if num_samples and sample_count >= num_samples:
            break

    img_mu = torch.cat(all_img_mu, dim=0)
    text_mus = torch.cat(all_text_mus, dim=0)              # (N, K, D)
    img_logvar = torch.cat(all_img_logvar, dim=0)
    text_logvars = torch.cat(all_text_logvars, dim=0)      # (N, K, D)

    if num_samples:
        img_mu = img_mu[:num_samples]
        text_mus = text_mus[:num_samples]
        img_logvar = img_logvar[:num_samples]
        text_logvars = text_logvars[:num_samples]

    logger.info(f"Features shape: Images {img_mu.shape}, Captions {text_mus.shape}")
    return img_mu, img_logvar, text_mus, text_logvars


def main():
    args = parse_args()
    set_seed(config.SEED)

    # auto-selects prolip_{dataset}_best.pt
    checkpoint_path = args.checkpoint or str(resolve_checkpoint("prolip", args.dataset))
    logger.info(f"Loading model from {checkpoint_path}")

    model = ProLIPModel()
    model.load(checkpoint_path)
    model = model.to(args.device)

    # coco=MSCOCO, flickr=flickr30k test split
    dataloader, num_eval_samples = build_eval_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
        num_samples=args.num_samples,
        captions_path=args.captions_path,
        images_dir=args.images_dir,
    )
    logger.info(f"Dataset loaded ({args.dataset}): {num_eval_samples} samples")

    img_mu, img_logvar, text_mus, text_logvars = extract_features(
        model, dataloader, args.device, args.num_samples,
    )

    groups = []

    # --- Protocol A (legacy, kept for comparability): 1:1 retrieval against
    # the FIRST caption of each image. Family keys unchanged. ---------------
    metrics = compute_retrieval_metrics(
        img_mu, img_logvar, text_mus[:, 0], text_logvars[:, 0],
        args.recall_at_k, chunk_size=1000,
    )
    for metric_name, label in [
        ("cosine", "Cosine Recall@K (legacy 1:1, first caption)"),
        ("csd", "CSD Recall@K (legacy 1:1, first caption)"),
    ]:
        per_k = {}
        for k in args.recall_at_k:
            i2t = metrics["i2t"][metric_name][f"recall@{k}"]
            t2i = metrics["t2i"][metric_name][f"recall@{k}"]
            per_k[k] = {"i2t": i2t, "t2i": t2i, "mean": (i2t + t2i) / 2}
        groups.append({"family": f"{metric_name}_recall", "label": label, "per_k": per_k})

    # --- Protocol B (unified multi-caption): N images vs N*K captions,
    # any-hit I2T + per-caption T2I -- the same protocol as MCDisp-Align's
    # mc_recall and eval_allhit. ProLIP's real logvars feed the
    # uncertainty-discounted score; the cosine family is scorer-identical
    # across all models. ------------------------------------------------------
    dev = torch.device(args.device)
    mc = compute_multicaption_recall(
        img_mu.to(dev), img_logvar.to(dev),
        text_mus.to(dev), text_logvars.to(dev),
        args.recall_at_k,
    )
    for fam, label in [
        ("mc_recall", "Multi-caption Recall@K (unified protocol, uncertainty-discounted)"),
        ("mc_cos_recall", "Multi-caption Recall@K (unified protocol, cosine)"),
    ]:
        groups.append({
            "family": fam, "label": label,
            "per_k": {
                k: {
                    "i2t": mc[f"{fam}_i2t@{k}"],
                    "t2i": mc[f"{fam}_t2i@{k}"],
                    "mean": mc[f"{fam}@{k}"],
                }
                for k in args.recall_at_k
            },
        })

    print_recall_groups(groups, logger)

    # Append results (never overwrite prior runs); time is stamped after dataset.
    output_path = args.output_path or str(config.PROLIP_EVAL_RESULTS_PATH)
    append_eval_results(output_path, {
        "model": "ProLIP ViT-H/14 (fine-tuned)",
        "checkpoint": str(checkpoint_path),
        "dataset": args.dataset,
        "num_samples": num_eval_samples,
        "metrics": groups_to_flat(groups),
    }, logger)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "ProLIP evaluation failed")
        raise
