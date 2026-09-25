"""Unified SET-LEVEL retrieval eval across all seed-42 checkpoints.

Protocol (the one MCDisp-Align is designed for): N images vs N caption SETS,
diagonal ground truth, Recall@{1,5,10} in both directions:

  I2T  (image -> text set):  query = image feature, gallery = text-set center
  T2I  (text set -> image):  query = text-set center, gallery = image features

Text-set representation and scorer are each model's own:

  clip    mean of the K caption embeddings, cosine
  prolip  moment-matched merge of the K per-caption Gaussians
          (models.prolip_model.merge_distributions_moment_matching),
          CSD (mu_i . mu_j - 0.5 * sum sigma_j^2, gallery-side discount) as the
          ProLIP-native scorer + cosine
  mcdisp  the model's own merged text distribution (moment matching inside
          forward), uncertainty-discounted score (= the L_set score) + cosine

The cosine family is scorer-identical across all models, so it is the
cross-model comparable column. Feature extraction is reused verbatim from the
per-model eval scripts, so numbers match those scripts' set-level groups.
"""

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from models.clip_baseline import CLIPFineTuneBaseline
from models.mcdisp_align_model import MCDispAlignModel
from models.prolip_model import ProLIPModel, merge_distributions_moment_matching
from scripts.evaluate_clip_baseline import extract_features as extract_clip
from scripts.evaluate_mcdisp_align import extract_features as extract_mcdisp
from scripts.evaluate_prolip import extract_features as extract_prolip
from utils.eval_common import build_eval_dataloader, VALID_DATASETS
from utils.eval_results import append_eval_results
from utils.logger import get_logger
from utils.retrieval import (
    compute_recall_bidirectional,
    compute_recall_mcdisp_align_chunked,
)
from utils.retrieval_metrics import compute_retrieval_metrics
from utils.seed import set_seed


logger = get_logger("eval_setlevel_all", config.LOG_DIR / "eval_setlevel_all.log")


# variant name -> (model family, checkpoint subdir under --ckpt-root,
#                  filename template; {ds} = dataset tag, {which} = best|last).
# Zero-shot variants carry no checkpoint (frozen pretrained weights).
VARIANTS = {
    "clip":                 ("clip",    "clip",                                    "clip_baseline_{ds}_{which}.pt"),
    "clip_zeroshot":        ("clip_zs", None,                                      None),
    "prolip":               ("prolip",  "prolip",                                  "prolip_{ds}_{which}.pt"),
    "prolip_zeroshot":      ("prolip_zs", None,                                    None),
    "mcdisp_sg":            ("mcdisp",  "mcdisp_align/sg",                         "mcdisp_align_kl_{ds}_{which}.pt"),
    "mcdisp_no_sg":         ("mcdisp",  "mcdisp_align/control_no_sg",              "mcdisp_align_kl_{ds}_{which}.pt"),
    "mcdisp_no_sg_flickrB": ("mcdisp",  "mcdisp_align/control_no_sg/flickr_runB",  "mcdisp_align_kl_{ds}_{which}.pt"),
    "mcdisp_residual_es5":  ("mcdisp",  "mcdisp_align/residual_es5",               "mcdisp_align_kl_{ds}_{which}.pt"),
}


def _per_k(flat, prefix, ks):
    """Flat {prefix}_i2t@{k} dict -> nested {k: {i2t, t2i}} for the table."""
    return {k: {"i2t": flat[f"{prefix}_i2t@{k}"],
                "t2i": flat[f"{prefix}_t2i@{k}"]}
            for k in ks}


def eval_clip(ckpt, dataloader, device, num_samples, ks):
    model = CLIPFineTuneBaseline()
    model.load(ckpt)
    return _run_clip(model, dataloader, device, num_samples, ks)


def eval_clip_zeroshot(_ckpt, dataloader, device, num_samples, ks):
    return _run_clip(CLIPFineTuneBaseline(freeze_image=True, freeze_text=True),
                     dataloader, device, num_samples, ks)


def _run_clip(model, dataloader, device, num_samples, ks):
    model = model.to(device)
    img_f, text_mus = extract_clip(model, dataloader, device, num_samples)
    center = text_mus.mean(dim=1)
    cos = compute_recall_bidirectional(
        img_f.to(device), center.to(device), ks, normalize=True)
    del model
    return {"cos": _per_k(cos, "recall", ks)}


def eval_mcdisp(ckpt, dataloader, device, num_samples, ks, tau):
    model = MCDispAlignModel(
        freeze_clip=config.MCDISP_ALIGN_FREEZE_CLIP,
        distribution_merging=config.MCDISP_ALIGN_DISTRIBUTION_MERGING,
        cov_rank=config.MCDISP_ALIGN_COV_RANK,
    )
    model.load(ckpt)
    model = model.to(device)
    img_mu, text_mu, img_lv, text_lv, _text_mus, _text_lvs = extract_mcdisp(
        model, dataloader, device, num_samples)
    disc = compute_recall_mcdisp_align_chunked(
        img_mu.to(device), img_lv.to(device),
        text_mu.to(device), text_lv.to(device), ks, tau=tau)
    cos = compute_recall_bidirectional(
        img_mu.to(device), text_mu.to(device), ks, normalize=True)
    del model
    return {"mcdisp": _per_k(disc, "mcdisp_align_recall", ks),
            "cos": _per_k(cos, "recall", ks)}


def eval_prolip(ckpt, dataloader, device, num_samples, ks):
    model = ProLIPModel()
    model.load(ckpt)
    return _run_prolip(model, dataloader, device, num_samples, ks)


def eval_prolip_zeroshot(_ckpt, dataloader, device, num_samples, ks):
    return _run_prolip(ProLIPModel(freeze=True),
                       dataloader, device, num_samples, ks)


def _run_prolip(model, dataloader, device, num_samples, ks):
    model = model.to(device)
    img_mu, img_lv, text_mus, text_lvs = extract_prolip(
        model, dataloader, device, num_samples)
    set_mu, set_lv = merge_distributions_moment_matching(text_mus, text_lvs)
    m = compute_retrieval_metrics(
        img_mu.to(device), img_lv.to(device),
        set_mu.to(device), set_lv.to(device), ks)
    del model
    return {"csd": {k: {"i2t": m["i2t"]["csd"][f"recall@{k}"],
                        "t2i": m["t2i"]["csd"][f"recall@{k}"]}
                    for k in ks},
            "cos": {k: {"i2t": m["i2t"]["cosine"][f"recall@{k}"],
                        "t2i": m["t2i"]["cosine"][f"recall@{k}"]}
                    for k in ks}}


def _flat(scorer, per_k):
    """per_k {k: {i2t, t2i}} -> flat keys {scorer}_i2t@{k} / {scorer}_t2i@{k}."""
    out = {}
    for k, comps in per_k.items():
        out[f"{scorer}_i2t@{k}"] = comps["i2t"]
        out[f"{scorer}_t2i@{k}"] = comps["t2i"]
    return out


def print_table(rows, ks):
    """rows: list of (variant, scorer, per_k dict). One aligned table."""
    header = f"{'variant':<24}{'scorer':<8}" + "".join(
        f"{'i2t@' + str(k):>9}{'t2i@' + str(k):>9}" for k in ks)
    lines = [header, "-" * len(header)]
    for variant, scorer, per_k in rows:
        cells = "".join(f"{per_k[k][d]:>9.4f}" for k in ks for d in ("i2t", "t2i"))
        lines.append(f"{variant:<24}{scorer:<8}{cells}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt-root", type=str,
                        default="/mnt/disk1/xpfu/checkpoints_s42_20260922")
    parser.add_argument("--datasets", type=str, nargs="+", default=["coco", "flickr"],
                        choices=list(VALID_DATASETS))
    parser.add_argument("--variants", type=str, nargs="+", default=list(VARIANTS),
                        choices=list(VARIANTS),
                        help="Which checkpoint variants to evaluate (default: all)")
    parser.add_argument("--which", type=str, default="best", choices=["best", "last"],
                        help="Checkpoint suffix per run (best = validation-selected)")
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="COCO eval subset size (Flickr always uses the full 1k test)")
    parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    parser.add_argument("--tau", type=float, default=config.MCDISP_ALIGN_TAU,
                        help="Temperature of the mcdisp uncertainty-discounted "
                             "score (ranking-neutral: constant global scale)")
    parser.add_argument("--output-path", type=str,
                        default=str(config.OUTPUT_DIR / "seed42_diag" / "setlevel_all.json"))
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(config.SEED)
    ks = list(config.RECALL_AT_K)

    for dataset in args.datasets:
        dataloader, n = build_eval_dataloader(
            dataset, batch_size=args.batch_size,
            num_workers=config.NUM_WORKERS, num_samples=args.num_samples)
        logger.info(f"=== dataset={dataset} (N={n}) ===")
        rows = []

        for variant in args.variants:
            family, subdir, tmpl = VARIANTS[variant]
            if tmpl is None:  # zero-shot: frozen pretrained weights, no file
                ckpt = "zero-shot (pretrained, no checkpoint)"
            else:
                ckpt = str(Path(args.ckpt_root) / subdir / tmpl.format(
                    ds=dataset, which=args.which))
                if not Path(ckpt).exists():
                    logger.info(f"skip {variant} ({dataset}): {ckpt} not found")
                    continue
            logger.info(f"[{dataset}] {variant}: {ckpt}")

            evaluators = {
                "clip": eval_clip,
                "clip_zs": eval_clip_zeroshot,
                "prolip": eval_prolip,
                "prolip_zs": eval_prolip_zeroshot,
            }
            if family in evaluators:
                scorers = evaluators[family](
                    ckpt, dataloader, args.device, args.num_samples, ks)
            else:
                scorers = eval_mcdisp(ckpt, dataloader, args.device,
                                      args.num_samples, ks, args.tau)

            metrics = {}
            for scorer, per_k in scorers.items():
                metrics.update(_flat(scorer, per_k))
                rows.append((variant, scorer, per_k))
            append_eval_results(args.output_path, {
                "variant": variant,
                "checkpoint": ckpt,
                "dataset": dataset,
                "num_samples": n,
                "protocol": "set-level N-vs-N (image vs caption-set center)",
                "metrics": metrics,
            }, logger)
            gc.collect()
            torch.cuda.empty_cache()

        table = print_table(rows, ks)
        logger.info("set-level Recall table (%s, N=%d):\n%s", dataset, n, table)
        print(f"\n=== set-level Recall ({dataset}, N={n}) ===\n{table}\n")

    logger.info("All done. Results appended to %s", args.output_path)


if __name__ == "__main__":
    main()
