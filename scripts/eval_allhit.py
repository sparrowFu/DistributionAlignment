"""
All-hit@K & Cover-rank Multi-Caption Retrieval Evaluation (standandalone)

Metrics (image query -> ranked caption gallery of all N*K eval captions):

  all-hit@K     fraction of images whose top-K retrieved captions are EXACTLY
                their own K captions (every own caption outranks every foreign
                one). any-hit@K (standard I2T recall) and paircount@K (mean #
                of own captions in the top-K) are reported for context.
  cover-rank    for each image, the smallest retrieval depth R_i at which its
                ENTIRE own caption set has been returned (the rank of its
                last-ranked own caption); reported as the MEAN over images --
                "on average, how many captions must the model return to fully
                cover an image's caption set". Median, censored mean
                (uncovered-within-k_max counted as k_max) and covered@k_max
                are reported alongside. cover-rank generalizes all-hit:
                R_i == K  <=>  the all-hit@K event.

Supports 5 models x 2 datasets, each scored with its native families:

    clip_zero_shot   frozen CLIP ViT-L/14          -- cosine
    clip_baseline    fine-tuned CLIP (checkpoint)  -- cosine
    prolip_zero_shot frozen ProLIP ViT-H/14        -- cosine, csd
    prolip           fine-tuned ProLIP (ckpt)      -- cosine, csd
    mcdisp_align     MCDisp_Align (checkpoint)     -- cosine, csd, mcdisp

  cosine : plain cosine of the means (no variance enters the score)
  csd    : ProLIP-style contraction-subspace distance -- cos(mu_i, mu_j) -
           0.5 * sum_d sigma_j,d^2, the GALLERY side discounted by its
           uncertainty (utils/retrieval_metrics.py semantics)
  mcdisp : THIS repo's uncertainty-discounted cosine, the same score the
           L_set contrastive loss optimizes (mcdisp_align only):
           sim = (mu_i . mu_j) / (tau * sqrt(1+mean sigma_i^2)
                                  * sqrt(1+mean sigma_j^2))

Datasets: coco (ImageCaptionDataset subset per utils/eval_common.py) and
flickr (Flickr30k test split). K is read from the loaded data (5 for both).

NOTE (checkpoint compatibility): this repo's config PROJECT_ROOT points at
the shared DistributionAlignment directory, so auto-selected checkpoints may
be weights trained with the NEWER four-group objective (MLP covariance head),
which do NOT load into this repo's MCDispAlignModel (single-Linear cov head).
For mcdisp_align, pass --checkpoint explicitly with a checkpoint saved by
THIS repo's model (e.g. checkpoints/mcdisp_align_pre_methodology_fix_*/*).

Usage:
    python scripts/eval_allhit.py --model mcdisp_align --dataset coco
    python scripts/eval_allhit.py --model clip_zero_shot --dataset flickr
    python scripts/eval_allhit.py --model prolip --dataset coco --num-samples 1000
"""

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.eval_common import build_eval_dataloader, resolve_checkpoint, VALID_DATASETS
from utils.eval_results import append_eval_results
from utils.logger import get_logger, log_exception
from utils.retrieval import compute_multicaption_allhit, compute_multicaption_coverrank
from utils.seed import set_seed

MODELS = ("clip_zero_shot", "clip_baseline", "prolip_zero_shot", "prolip",
          "mcdisp_align")
# models whose checkpoints are auto-selected as {name}_{dataset}_best.pt
FINETUNED = {"clip_baseline": "clip_baseline", "prolip": "prolip",
             "mcdisp_align": "mcdisp_align"}

logger = get_logger("eval_allhit", config.EVAL_ALLHIT_LOG_PATH)


def parse_args():
    parser = argparse.ArgumentParser(
        description="All-hit@K & cover-rank multi-caption retrieval evaluation "
                    "(top-K captions are EXACTLY the image's own K captions; "
                    "mean retrieval depth to cover the whole caption set)")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS),
                        help="Model to evaluate")
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=list(VALID_DATASETS),
                        help="Dataset to evaluate on (coco=MSCOCO, flickr=flickr30k)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for fine-tuned models. If None, "
                             "auto-selects {model}_{dataset}_best.pt "
                             "(ignored for the zero-shot variants; see the "
                             "checkpoint-compatibility NOTE above for "
                             "mcdisp_align)")
    parser.add_argument("--captions-path", type=str, default=None,
                        help="Captions file override (coco only)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Images directory override (coco only)")
    parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE,
                        help="Evaluation batch size")
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="Number of images to evaluate (default: 5000; "
                             "0/null on flickr = full test split)")
    parser.add_argument("--k-hit", type=int, default=None,
                        help="Top-k cutoff for the all-hit test (default: K, "
                             "the whole caption set -- the all-own-set metric)")
    parser.add_argument("--k-max", type=int, default=100,
                        help="Retrieval-depth cap for the cover-rank metric "
                             "(default: 100; images not covered within this "
                             "depth are censored)")
    parser.add_argument("--chunk-rows", type=int, default=512,
                        help="Image-row chunk for the score matrices")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output JSON path (uses config default if None)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    return parser.parse_args()


# --------------------------------------------------------------------------
# Model loading (mirrors the per-model evaluate_* scripts)
# --------------------------------------------------------------------------
def build_model(model_name: str, checkpoint_path: Optional[str], device):
    if model_name == "clip_zero_shot":
        from models.clip_baseline import CLIPFineTuneBaseline
        model = CLIPFineTuneBaseline(freeze_image=True, freeze_text=True)
    elif model_name == "clip_baseline":
        from models.clip_baseline import CLIPFineTuneBaseline
        model = CLIPFineTuneBaseline()
        model.load(checkpoint_path)
    elif model_name == "prolip_zero_shot":
        from models.prolip_model import ProLIPModel
        model = ProLIPModel(freeze=True)
    elif model_name == "prolip":
        from models.prolip_model import ProLIPModel
        model = ProLIPModel()
        model.load(checkpoint_path)
    elif model_name == "mcdisp_align":
        from models.mcdisp_align_model import MCDispAlignModel
        model = MCDispAlignModel(
            freeze_clip=config.MCDISP_ALIGN_FREEZE_CLIP,
            distribution_merging=config.MCDISP_ALIGN_DISTRIBUTION_MERGING,
            cov_rank=config.MCDISP_ALIGN_COV_RANK,
        )
        model.load(checkpoint_path)
    else:
        raise ValueError(f"Unknown model {model_name!r}")
    return model.to(device).eval()


# --------------------------------------------------------------------------
# Feature extraction -> a uniform dict:
#   img_mu (N, D), text_mus (N, K, D)                 [always]
#   img_logvar (N, D), text_logvars (N, K, D)         [prolip, mcdisp_align]
# --------------------------------------------------------------------------
@torch.no_grad()
def extract_features(model, model_name: str, dataloader: DataLoader,
                     device, num_samples: Optional[int]) -> Dict[str, torch.Tensor]:
    model.eval()
    acc: Dict[str, List[torch.Tensor]] = {
        k: [] for k in ("img_mu", "text_mus", "img_logvar", "text_logvars")}
    sample_count = 0
    K = None

    logger.info("Extracting features (all K captions per image)...")
    for batch in tqdm(dataloader, desc="Extracting"):
        if batch is None:
            continue
        pil_images = batch["image"]
        caption_lists = batch["captions"]
        B = len(pil_images)
        K = len(caption_lists[0])
        all_captions: List[str] = []
        for captions in caption_lists:
            all_captions.extend(captions)

        if model_name in ("clip_zero_shot", "clip_baseline"):
            pixel_values = model.process_images(pil_images).to(device)
            text_inputs = model.process_text(all_captions)
            input_ids = text_inputs["input_ids"].to(device)
            attention_mask = text_inputs["attention_mask"].to(device)
            img_feat, text_feat = model(
                images=pixel_values, input_ids=input_ids,
                attention_mask=attention_mask)          # (B, D), (B*K, D)
            acc["img_mu"].append(img_feat.cpu())
            acc["text_mus"].append(text_feat.cpu())
        elif model_name in ("prolip_zero_shot", "prolip"):
            pixel_values = model.process_images(pil_images)
            text_inputs = model.process_text(all_captions)
            input_ids = text_inputs["input_ids"].to(device)
            out = model(pixel_values, input_ids)
            acc["img_mu"].append(out["img_mu"].cpu())
            acc["text_mus"].append(out["text_mu"].cpu())
            acc["img_logvar"].append(out["img_logvar"].cpu())
            acc["text_logvars"].append(out["text_logvar"].cpu())
        elif model_name == "mcdisp_align":
            pixel_values = model.process_images(pil_images).to(device)
            text_inputs = model.process_text(all_captions)
            input_ids = text_inputs["input_ids"].view(B, K, -1).to(device)
            attention_mask = text_inputs["attention_mask"].view(B, K, -1).to(device)
            out = model(pixel_values, input_ids, attention_mask)
            acc["img_mu"].append(out["img_mu"].cpu())
            acc["text_mus"].append(out["text_mus"].cpu())
            acc["img_logvar"].append(out["img_logvar"].cpu())
            acc["text_logvars"].append(out["text_logvars"].cpu())
        else:
            raise ValueError(f"Unknown model {model_name!r}")

        sample_count += B
        if num_samples and sample_count >= num_samples:
            break

    feats = {"img_mu": torch.cat(acc["img_mu"], dim=0),
             "text_mus": torch.cat(acc["text_mus"], dim=0)}
    N = feats["img_mu"].shape[0]
    if acc["text_mus"][0].dim() == 2:                    # (B*K, D) -> (N, K, D)
        feats["text_mus"] = feats["text_mus"].view(N, K, -1)
    if acc["img_logvar"]:
        feats["img_logvar"] = torch.cat(acc["img_logvar"], dim=0)
        feats["text_logvars"] = torch.cat(acc["text_logvars"], dim=0)
        feats["text_logvars"] = feats["text_logvars"].view(N, K, -1)

    logger.info(f"Features: images {feats['img_mu'].shape}, "
                f"captions {feats['text_mus'].shape}")
    return feats


# --------------------------------------------------------------------------
# Score families: {name: fn(row_slice) -> (n_rows, N*K) score matrix}
# (gallery side = captions, image rows come in via the slice)
# --------------------------------------------------------------------------
def build_scorers(model_name: str, feats: Dict[str, torch.Tensor],
                  device) -> Dict[str, Callable[[slice], torch.Tensor]]:
    img_mu = feats["img_mu"].to(device)                   # (N, D)
    text_mus = feats["text_mus"].to(device)               # (N, K, D)
    N, K, D = text_mus.shape
    cap_mu = text_mus.reshape(N * K, D)                   # (Q, D)
    img_n = F.normalize(img_mu, dim=-1)
    cap_n = F.normalize(cap_mu, dim=-1)

    scorers: Dict[str, Callable[[slice], torch.Tensor]] = {
        "cos": lambda sl: img_n[sl] @ cap_n.T,
    }

    if "text_logvars" in feats:                           # csd: gallery discount
        cap_unc = torch.exp(feats["text_logvars"].to(device)).sum(-1).reshape(N * K)
        scorers["csd"] = lambda sl: (img_n[sl] @ cap_n.T) - 0.5 * cap_unc.unsqueeze(0)

    if model_name == "mcdisp_align":
        # the L_set uncertainty-discounted cosine (train=eval agreement):
        # sim = (mu_i . mu_j) / (tau * sqrt(1+mean sigma_i^2) * sqrt(1+mean sigma_j^2))
        tau = config.MCDISP_ALIGN_TAU
        img_scale = torch.sqrt(
            1.0 + torch.exp(feats["img_logvar"].to(device)).mean(dim=-1))    # (N,)
        cap_scale = torch.sqrt(
            1.0 + torch.exp(feats["text_logvars"].to(device)).mean(dim=-1).reshape(N * K))

        def _mcdisp(sl: slice):
            sim = img_n[sl] @ cap_n.T
            return sim / (tau * img_scale[sl].unsqueeze(1) * cap_scale.unsqueeze(0))

        scorers["mcdisp"] = _mcdisp

    return scorers


def main():
    args = parse_args()
    set_seed(config.SEED)

    checkpoint_path = None
    if args.model in FINETUNED:
        checkpoint_path = args.checkpoint or str(
            resolve_checkpoint(FINETUNED[args.model], args.dataset))
        logger.info(f"Loading {args.model} from {checkpoint_path}")
    else:
        logger.info(f"Loading frozen {args.model} (zero-shot, no checkpoint)")
    model = build_model(args.model, checkpoint_path, args.device)

    dataloader, num_eval_samples = build_eval_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
        num_samples=args.num_samples,
        captions_path=args.captions_path,
        images_dir=args.images_dir,
    )
    logger.info(f"Dataset loaded ({args.dataset}): {num_eval_samples} samples")

    feats = extract_features(model, args.model, dataloader,
                             args.device, args.num_samples)
    N, K, _ = feats["text_mus"].shape
    k_hit = args.k_hit or K
    if k_hit > K:
        logger.warning(
            f"k_hit={k_hit} > K={K}: all-hit is 0 by construction "
            "(more slots than own captions); anyhit/paircount stay meaningful.")

    scorers = build_scorers(args.model, feats, args.device)
    metrics = compute_multicaption_allhit(
        scorers, n_images=N, k_per_image=K, k_hit=k_hit,
        chunk_rows=args.chunk_rows)
    if args.k_max >= K:
        cover = compute_multicaption_coverrank(
            scorers, n_images=N, k_per_image=K, k_max=args.k_max,
            chunk_rows=args.chunk_rows)
        metrics.update(cover)
    else:
        logger.warning(f"--k-max={args.k_max} < K={K}: cover-rank skipped")
        args.k_max = None

    logger.info("=" * 64)
    logger.info(f"All-hit@{k_hit} & cover-rank | model={args.model} "
                f"| dataset={args.dataset} | N={N} images, K={K} captions/image")
    for fam in scorers:
        logger.info(
            f"  [{fam:8s}] all-hit@{k_hit}={metrics[f'{fam}_allhit@{k_hit}']:.4f}  "
            f"any-hit@{k_hit}={metrics[f'{fam}_anyhit@{k_hit}']:.4f}  "
            f"paircount@{k_hit}={metrics[f'{fam}_paircount@{k_hit}']:.3f}/{K}")
        if args.k_max:
            km = args.k_max
            logger.info(
                f"  [{fam:8s}] cover-rank mean={metrics[f'{fam}_coverrank_mean@{km}']:.2f}  "
                f"median={metrics[f'{fam}_coverrank_median@{km}']:.2f}  "
                f"censored-mean={metrics[f'{fam}_coverrank_censored_mean@{km}']:.2f}  "
                f"covered@{km}={metrics[f'{fam}_covered@{km}']:.4f}")
    logger.info("=" * 64)

    output_path = args.output_path or str(config.ALLHIT_EVAL_RESULTS_PATH)
    append_eval_results(output_path, {
        "model": args.model,
        "checkpoint": checkpoint_path,
        "dataset": args.dataset,
        "num_samples": num_eval_samples,
        "metric": f"allhit@{k_hit}+coverrank@{args.k_max}",
        "k_per_image": K,
        "metrics": metrics,
    }, logger)
    logger.info(f"Results appended to {output_path}")
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "All-hit evaluation failed")
        raise
