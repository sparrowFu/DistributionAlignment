"""
PCME++ Evaluation Script (B5)

Primary protocol: the official PCME++ COCO test evaluation (vendored
pcmepp/evaluation.py) -- all 5 captions per test image, CSD similarity,
i2t (any-of-5 hit) and t2i (per-caption) Recall@1/5/10 + medR.

Also reports the shared project protocols for cross-baseline comparability:
Protocol A (legacy 1:1 vs the first caption, cosine + CSD) and Protocol B
(unified multi-caption: N images vs N*K captions, uncertainty-discounted +
cosine). Features come from the official eval pipeline (official image
transform, tokenize2 without perturbation); img mean is l2-normalized by the
model and 'std' is log sigma^2, feeding the shared scorers unchanged.
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from models.pcmepp_model import PCMEPPModel
from pcmepp.evaluation import compute_csd_sims, eval_i2t_val, eval_t2i_val
from utils.eval_common import build_eval_dataloader, resolve_checkpoint, VALID_DATASETS
from utils.eval_results import append_eval_results, groups_to_flat, print_recall_groups
from utils.logger import get_logger, log_exception
from utils.retrieval_metrics import compute_retrieval_metrics
from utils.retrieval import compute_multicaption_recall
from utils.seed import set_seed


logger = get_logger("eval_pcmepp", config.EVAL_PCMEPP_LOG_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PCME++ (B5)")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. If None, auto-selects "
                             "pcmepp_{dataset}_best.pt from --dataset "
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
    """Extract PCME++ mu / log-variance features for all K captions
    (official eval transform, no token perturbation)."""
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

        # official eval image transform (CPU) -> device
        pixel_values = model.process_images(pil_images).to(device)

        text_inputs = model.process_text(all_captions)   # (B*K, 77), (B*K,)
        input_ids = text_inputs["input_ids"].view(B, K, -1).to(device)
        lengths = text_inputs["lengths"].view(B, K).to(device)

        outputs = model(pixel_values, input_ids, lengths)

        all_img_mu.append(outputs["img_mu"].cpu())
        all_text_mus.append(outputs["text_mus"].cpu())
        all_img_logvar.append(outputs["img_logvar"].cpu())
        all_text_logvars.append(outputs["text_logvars"].cpu())

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

    # auto-selects pcmepp_{dataset}_best.pt
    checkpoint_path = args.checkpoint or str(resolve_checkpoint("pcmepp", args.dataset))
    logger.info(f"Loading model from {checkpoint_path}")

    model = PCMEPPModel()
    model.load(checkpoint_path)
    model = model.to(args.device)

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

    # --- Official PCME++ COCO test protocol (paper Table 1): N images vs
    # 5N captions (grouped 5-per-image), CSD similarity from the vendored
    # official evaluation.py; i2t = any-of-the-5 hit, t2i = per caption.
    n_imgs = img_mu.shape[0]
    k_caps = text_mus.shape[1]
    cap_mu_flat = text_mus.reshape(n_imgs * k_caps, -1)
    cap_logvar_flat = text_logvars.reshape(n_imgs * k_caps, -1)
    if k_caps == 5:
        sims = compute_csd_sims(img_mu, cap_mu_flat, img_logvar, cap_logvar_flat)
        i2t = eval_i2t_val(n_imgs, sims)
        t2i = eval_t2i_val(n_imgs, sims)
        ret_at = {1: 0, 5: 1, 10: 2}  # (r1, r5, r10, medr, meanr)
        per_k = {}
        for k in args.recall_at_k:
            if k not in ret_at:
                logger.warning(f"official protocol reports R@1/5/10 only; skip k={k}")
                continue
            pos = ret_at[k]
            per_k[k] = {"i2t": i2t[pos], "t2i": t2i[pos],
                        "mean": (i2t[pos] + t2i[pos]) / 2}
        groups.append({
            "family": "official_csd_recall",
            "label": ("Official CSD Recall@K (5-caption protocol, paper) "
                      f"medR i2t {i2t[3]:.0f} / t2i {t2i[3]:.0f}"),
            "per_k": per_k,
        })
    else:
        logger.warning(f"official 5-caption protocol needs K=5 captions, got {k_caps}; skipped")

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
    # any-hit I2T + per-caption T2I -- the same protocol as the other
    # baselines. PCME++'s log sigma^2 feeds the uncertainty-discounted
    # score; the cosine family is scorer-identical across all models. ------
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
    output_path = args.output_path or str(config.PCMEPP_EVAL_RESULTS_PATH)
    append_eval_results(output_path, {
        "model": "PCME++ (CLIP ViT-L/14, full fine-tune)",
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
        log_exception(logger, e, "PCME++ evaluation failed")
        raise
