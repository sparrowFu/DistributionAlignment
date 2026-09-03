"""
MCDisp_Align Evaluation Script

Evaluates the MCDisp_Align (Multi-Caption Semantic Dispersion Guided Distribution
Alignment) model with the standard multi-caption retrieval protocol
(N images vs N*K captions; I2T any-of-K-hit, T2I per-caption single-positive)
under the plain cosine of the means and the same protocol the trainer uses
for checkpoint selection. This is the canonical MS-COCO/Flickr30k protocol,
comparable to published baselines.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from models.mcdisp_align_model import MCDispAlignModel
from utils.eval_common import build_eval_dataloader, resolve_checkpoint, VALID_DATASETS
from utils.eval_results import append_eval_results, groups_to_flat, print_recall_groups
from utils.logger import get_logger, log_exception
from utils.retrieval import (
    compute_multicaption_recall,
    compute_multicaption_recall_dist,
)
from utils.seed import set_seed


logger = get_logger("eval_mcdisp_align", config.EVAL_MCDISP_ALIGN_LOG_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MCDisp_Align Model")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. If None, auto-selects "
                             "mcdisp_align_{dataset}_best.pt from --dataset "
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
    parser.add_argument("--tau", type=float, default=config.MCDISP_ALIGN_TAU,
                        help="Kept for record only; the retrieval score is the "
                             "plain cosine, on which tau has no effect")

    return parser.parse_args()


@torch.no_grad()
def extract_features(
    model: MCDispAlignModel,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = None
):
    """Extract image and text distribution features (mu and logvar)."""
    model.eval()

    all_img_mu = []
    all_text_mu = []
    all_img_logvar = []
    all_text_logvar = []
    all_text_mus = []        # per-caption means (B, K, D) -- I2T pair-count metric
    all_text_logvars = []    # per-caption log-variances (B, K, D)
    all_img_U = []           # image low-rank covariance factors (B, D, r), if any
    sample_count = 0

    logger.info("Extracting features...")
    for batch in tqdm(dataloader):
        if batch is None:
            continue

        pil_images = batch["image"]
        caption_lists = batch["captions"]

        pixel_values = model.process_images(pil_images).to(device)

        batch_size = len(pil_images)
        num_captions = len(caption_lists[0])

        all_captions = []
        for caption_list in caption_lists:
            all_captions.extend(caption_list)

        text_inputs = model.process_text(all_captions)

        # Reshape to [B, K, max_len]
        input_ids = text_inputs["input_ids"].view(batch_size, num_captions, -1).to(device)
        attention_mask = text_inputs["attention_mask"].view(batch_size, num_captions, -1).to(device)

        outputs = model(pixel_values, input_ids, attention_mask)

        all_img_mu.append(outputs['img_mu'].cpu())
        all_text_mu.append(outputs['text_mu'].cpu())
        all_img_logvar.append(outputs['img_logvar'].cpu())
        all_text_logvar.append(outputs['text_logvar'].cpu())
        all_text_mus.append(outputs['text_mus'].cpu())
        all_text_logvars.append(outputs['text_logvars'].cpu())
        if outputs['img_U'] is not None:
            all_img_U.append(outputs['img_U'].cpu())

        sample_count += batch_size
        if num_samples and sample_count >= num_samples:
            break

    img_mu = torch.cat(all_img_mu, dim=0)
    text_mu = torch.cat(all_text_mu, dim=0)
    img_logvar = torch.cat(all_img_logvar, dim=0)
    text_logvar = torch.cat(all_text_logvar, dim=0)
    text_mus = torch.cat(all_text_mus, dim=0)          # (N, K, D)
    text_logvars = torch.cat(all_text_logvars, dim=0)  # (N, K, D)
    img_U = torch.cat(all_img_U, dim=0) if all_img_U else None  # (N, D, r) or None

    if num_samples:
        img_mu = img_mu[:num_samples]
        text_mu = text_mu[:num_samples]
        img_logvar = img_logvar[:num_samples]
        text_logvar = text_logvar[:num_samples]
        text_mus = text_mus[:num_samples]
        text_logvars = text_logvars[:num_samples]
        img_U = img_U[:num_samples] if img_U is not None else None

    logger.info(f"Features shape: Images {img_mu.shape}, Texts {text_mu.shape}")

    return img_mu, text_mu, img_logvar, text_logvar, text_mus, text_logvars, img_U


def main():
    args = parse_args()

    set_seed(config.SEED)

    # auto-selects mcdisp_align_{dataset}_best.pt
    checkpoint_path = args.checkpoint or str(resolve_checkpoint("mcdisp_align", args.dataset))
    logger.info(f"Loading model from {checkpoint_path}")

    model = MCDispAlignModel(
        freeze_clip=config.MCDISP_ALIGN_FREEZE_CLIP,
        cov_rank=config.MCDISP_ALIGN_COV_RANK,
    )

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

    # text_mus/text_logvars are the per-caption outputs used by the
    # multi-caption retrieval protocol.
    img_mu, text_mu, img_logvar, text_logvar, text_mus, text_logvars, img_U = extract_features(
        model, dataloader, args.device, args.num_samples
    )

    img_mu_d = img_mu.to(args.device)
    img_lv_d = img_logvar.to(args.device)
    text_mus_d = text_mus.to(args.device)
    text_logvars_d = text_logvars.to(args.device)
    img_U_d = img_U.to(args.device) if img_U is not None else None

    # Primary metric: standard multi-caption bidirectional Recall (N images vs
    # N*K captions) under the plain-cosine MCDisp_Align score -- the canonical
    # MS-COCO/Flickr one-image-many-captions protocol, comparable to published
    # baselines. Train-time checkpoint selection uses the same protocol.
    mc = compute_multicaption_recall(
        img_mu_d, img_lv_d, text_mus_d, text_logvars_d,
        args.recall_at_k, tau=args.tau,
    )

    # Distribution-aware families on the SAME full pool: the learned
    # (co)variances enter the retrieval score directly. This is the
    # "does the distribution earn its keep at inference?" readout.
    dmc = compute_multicaption_recall_dist(
        img_mu_d, img_lv_d, img_U_d, text_mus_d, text_logvars_d,
        args.recall_at_k,
    )

    groups = [
        {
            "family": "mc_recall",
            "label": "Multi-caption Recall@K (plain-cosine MCDisp_Align score; N vs N*K)",
            "per_k": {
                k: {
                    "i2t": mc[f"mc_recall_i2t@{k}"],
                    "t2i": mc[f"mc_recall_t2i@{k}"],
                    "mean": mc[f"mc_recall@{k}"],
                }
                for k in args.recall_at_k
            },
        },
        {
            "family": "mc_overlap_recall",
            "label": "Multi-caption Recall@K (Gaussian overlap score psi; means + (co)variances score)",
            "per_k": {
                k: {
                    "i2t": dmc.get(f"mc_overlap_recall_i2t@{k}", 0.0),
                    "t2i": dmc.get(f"mc_overlap_recall_t2i@{k}", 0.0),
                    "mean": dmc.get(f"mc_overlap_recall@{k}", 0.0),
                }
                for k in args.recall_at_k
            },
        },
        {
            "family": "mc_ellip_recall",
            "label": "Multi-caption Recall@K (ellipsoid membership depth; -Mahalanobis of caption mean in image ellipsoid)",
            "per_k": {
                k: {
                    "i2t": dmc.get(f"mc_ellip_recall_i2t@{k}", 0.0),
                    "t2i": dmc.get(f"mc_ellip_recall_t2i@{k}", 0.0),
                    "mean": dmc.get(f"mc_ellip_recall@{k}", 0.0),
                }
                for k in args.recall_at_k
            },
        },
        {
            "family": "mc_csd_recall",
            "label": "Multi-caption Recall@K (ProLIP-style CSD; gallery uncertainty discount)",
            "per_k": {
                k: {
                    "i2t": mc.get(f"mc_csd_recall_i2t@{k}", 0.0),
                    "t2i": mc.get(f"mc_csd_recall_t2i@{k}", 0.0),
                    "mean": mc.get(f"mc_csd_recall@{k}", 0.0),
                }
                for k in args.recall_at_k
            },
        },
    ]

    print_recall_groups(groups, logger)
    recall_metrics = groups_to_flat(groups)

    # Append results (never overwrite prior runs); time is stamped after dataset.
    output_path = args.output_path or str(config.MCDISP_ALIGN_EVAL_RESULTS_PATH)
    append_eval_results(output_path, {
        'checkpoint': str(checkpoint_path),
        'dataset': args.dataset,
        'num_samples': num_eval_samples,
        'tau': args.tau,
        'metrics': recall_metrics,
    }, logger)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Evaluation failed")
        raise
