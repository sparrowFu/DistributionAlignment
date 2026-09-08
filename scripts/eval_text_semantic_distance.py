#!/usr/bin/env python
"""Analytical experiment: within-image caption semantic distance per model.

Motivation (user 2026-09-08): verify that MCDisp-Align preserves the semantic
richness of a image's multiple captions, whereas 1-to-1 contrastive models
(CLIP baseline / ProLIP) pull every caption toward the paired image and
collapse the within-image semantic spread.

Protocol:
  * Take a small set of images (default 100 per dataset; coco = seeded random
    subset, flickr = first N of the test split), each with K captions.
  * For every model, extract ONE representation center per caption:
      - CLIP (zero-shot / fine-tuned): text-encoder embedding
      - ProLIP (zero-shot / fine-tuned): caption Gaussian mean mu_k
      - MCDisp-Align (standard / kl): per-caption distribution center text_mus
  * Distances (cosine and euclidean, on RAW un-normalized vectors):
      - within-image: all C(K,2) caption pairs of the same image
      - cross-image:  same pair layout against a shifted image (partner index
        (i + N/2) % N, never i itself) as the model-specific reference scale
  * Dispersion ratio = mean(within) / mean(cross). A ratio << 1 means the
    model's own captions have collapsed toward each other relative to random
    captions; a ratio closer to the zero-shot reference means the caption
    semantics are preserved.

All models see the SAME images/captions (one collected batch list), so the
comparison is paired. Output: JSON + markdown table + box plot per dataset.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.logger import get_logger, log_exception
from utils.seed import set_seed
from utils.eval_common import build_eval_dataloader

# Exclude faulty CPU cores before DataLoader workers and torch threads.
from utils.cpu_affinity import apply_cpu_affinity
apply_cpu_affinity()

logger = get_logger("text_sem_dist", config.LOG_DIR / "text_semantic_distance.log")

# (key, display name) — evaluation order = table order
MODEL_SPECS = [
    ("clip_zero_shot", "CLIP zero-shot (frozen)"),
    ("clip_baseline", "CLIP fine-tuned (1-to-1)"),
    ("prolip_zero_shot", "ProLIP zero-shot (frozen)"),
    ("prolip", "ProLIP fine-tuned (1-to-1)"),
    ("mcdisp_align", "MCDisp-Align (standard)"),
    ("mcdisp_align_kl", "MCDisp-Align (KL)"),
]


def build_model(key: str, ckpt_root: Path, dataset: str):
    """Instantiate a model; trained variants load their seed checkpoint."""
    if key == "clip_zero_shot":
        from models.clip_baseline import CLIPFineTuneBaseline
        return CLIPFineTuneBaseline(freeze_image=True, freeze_text=True)
    if key == "clip_baseline":
        from models.clip_baseline import CLIPFineTuneBaseline
        m = CLIPFineTuneBaseline()
        m.load(str(ckpt_root / f"clip_baseline_{dataset}_best.pt"))
        return m
    if key == "prolip_zero_shot":
        from models.prolip_model import ProLIPModel
        return ProLIPModel(freeze=True)
    if key == "prolip":
        from models.prolip_model import ProLIPModel
        m = ProLIPModel()
        m.load(str(ckpt_root / f"prolip_{dataset}_best.pt"))
        return m
    if key in ("mcdisp_align", "mcdisp_align_kl"):
        from models.mcdisp_align_model import MCDispAlignModel
        m = MCDispAlignModel()
        m.load(str(ckpt_root / f"{key}_{dataset}_best.pt"))
        return m
    raise KeyError(key)


@torch.no_grad()
def caption_centers(key: str, model, batches, device) -> torch.Tensor:
    """Per-caption representation centers for the collected batches -> (N, K, D)."""
    outs, K = [], None
    for pil_images, caption_lists in tqdm(batches, desc=f"encode[{key}]", leave=False):
        B = len(caption_lists)
        K = len(caption_lists[0])
        flat = [c for cl in caption_lists for c in cl]
        ti = model.process_text(flat)
        input_ids = ti["input_ids"].to(device)
        attention_mask = ti["attention_mask"].to(device)

        if key.startswith("clip"):
            centers = model.encode_text(input_ids, attention_mask, normalize=False)
        elif key.startswith("prolip"):
            # ProLIP handles padding internally; attention_mask unused.
            centers = model.encode_texts(input_ids, normalize=False)["mean"]
        else:  # mcdisp: forward needs the images too; use per-caption centers
            pixel_values = model.process_images(pil_images).to(device)
            out = model(pixel_values,
                        input_ids.view(B, K, -1),
                        attention_mask.view(B, K, -1))
            centers = out["text_mus"].reshape(B * K, -1)

        outs.append(centers.float().cpu())
    return torch.cat(outs, dim=0).view(-1, K, centers.shape[-1])


def pairwise(centers: torch.Tensor):
    """(A, B) tensors of shape (N, P, D) over the C(K,2) within-image layout."""
    N, K, _ = centers.shape
    idx = torch.triu_indices(K, K, offset=1)
    A = centers[:, idx[0], :]
    # cross-image partner: shifted image, never the image itself
    shift = (torch.arange(N) + N // 2) % N
    B_within = centers[:, idx[1], :]
    B_cross = centers[shift][:, idx[1], :]
    return A, B_within, B_cross


def distances(A: torch.Tensor, B: torch.Tensor):
    """Cosine + euclidean distances for two aligned (N, P, D) tensors."""
    cos = (1.0 - F.cosine_similarity(A, B, dim=-1)).flatten()
    euc = (A - B).norm(dim=-1).flatten()
    return {"cosine": cos, "euclidean": euc}


def stats(t: torch.Tensor):
    t = t.double()
    return {"mean": t.mean().item(), "std": t.std(unbiased=True).item(),
            "n_pairs": int(t.numel())}


def run_dataset(dataset: str, args, out_dir: Path) -> dict:
    set_seed(config.SEED)  # fixes the coco subset selection

    logger.info(f"[{dataset}] building eval loader (num_images={args.num_images})")
    loader, _ = build_eval_dataloader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        num_samples=args.num_images if dataset == "coco" else None,
    )

    # Collect the shared sample set ONCE; every model sees identical data.
    batches, n_img = [], 0
    for batch in loader:
        if batch is None:
            continue
        batches.append((batch["image"], batch["captions"]))
        n_img += len(batch["image"])
        if n_img >= args.num_images:
            break
    logger.info(f"[{dataset}] collected {n_img} images x K captions")

    results = {}
    for key, disp in MODEL_SPECS:
        logger.info(f"[{dataset}] model: {disp}")
        model = build_model(key, Path(args.ckpt_root), dataset).to(args.device).eval()
        centers = caption_centers(key, model, batches, args.device)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

        A, Bw, Bc = pairwise(centers)
        wit = distances(A, Bw)
        crs = distances(A, Bc)
        results[key] = {
            "display": disp,
            "within": {m: stats(v) for m, v in wit.items()},
            "cross": {m: stats(v) for m, v in crs.items()},
            "ratio": {m: wit[m].mean().item() / crs[m].mean().item()
                      for m in wit},
            "raw": {"within_cosine": wit["cosine"].tolist(),
                    "cross_cosine": crs["cosine"].tolist()},
        }
        logger.info(f"[{dataset}] {disp}: within-cos={results[key]['within']['cosine']['mean']:.4f} "
                    f"cross-cos={results[key]['cross']['cosine']['mean']:.4f} "
                    f"ratio={results[key]['ratio']['cosine']:.3f}")

    # persist
    ds_dir = out_dir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    with open(ds_dir / "results.json", "w") as f:
        json.dump({"dataset": dataset, "num_images": n_img, "models": results}, f, indent=2)

    write_table(dataset, results, ds_dir / "summary.md")
    try:
        plot(dataset, results, ds_dir / "within_vs_cross_cosine.png")
    except Exception as e:  # matplotlib optional
        logger.warning(f"plot skipped: {e}")
    return results


def write_table(dataset: str, results: dict, path: Path):
    lines = [
        f"# Within-image caption semantic distance ({dataset})",
        "",
        "Distances between the K captions of the SAME image (within) vs captions of",
        "DIFFERENT images (cross); ratio = within/cross. Lower ratio = stronger",
        "within-image semantic collapse under 1-to-1 contrastive training.",
        "",
        "| Model | cos-within | cos-cross | cos-ratio | L2-within | L2-cross | L2-ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, r in results.items():
        w, c = r["within"], r["cross"]
        lines.append(
            f"| {r['display']} "
            f"| {w['cosine']['mean']:.4f}±{w['cosine']['std']:.4f} "
            f"| {c['cosine']['mean']:.4f}±{c['cosine']['std']:.4f} "
            f"| **{r['ratio']['cosine']:.3f}** "
            f"| {w['euclidean']['mean']:.3f}±{w['euclidean']['std']:.3f} "
            f"| {c['euclidean']['mean']:.3f}±{c['euclidean']['std']:.3f} "
            f"| {r['ratio']['euclidean']:.3f} |")
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"table written: {path}")


def plot(dataset: str, results: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(results)
    within = [results[k]["raw"]["within_cosine"] for k in keys]
    cross = [results[k]["raw"]["cross_cosine"] for k in keys]
    labels = [results[k]["display"] for k in keys]

    fig, ax = plt.subplots(figsize=(11, 5))
    pos = torch.arange(len(keys))
    bp1 = ax.boxplot(within, positions=pos - 0.18, widths=0.32,
                     patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(cross, positions=pos + 0.18, widths=0.32,
                     patch_artist=True, showfliers=False)
    for b in bp1["boxes"]:
        b.set_facecolor("#4C72B0")
    for b in bp2["boxes"]:
        b.set_facecolor("#DD8452")
    for i, k in enumerate(keys):
        ax.text(i, -0.06, f"ratio={results[k]['ratio']['cosine']:.3f}",
                ha="center", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("cosine distance")
    ax.set_title(f"Within-image vs cross-image caption distance ({dataset})")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["within-image", "cross-image"],
              loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"plot written: {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", nargs="+", default=["coco", "flickr"],
                   choices=["coco", "flickr"])
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--ckpt-root", type=str, default="checkpoints/seed42",
                   help="Directory holding the trained per-seed checkpoints")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "cuda":
            free, _total = torch.cuda.mem_get_info()
            if free < 8 << 30:  # protect the concurrent training run
                logger.info(f"GPU free {free >> 30}G < 8G -> falling back to cpu")
                args.device = "cpu"
    out_dir = Path(args.output_dir) if args.output_dir else (
        config.OUTPUT_DIR / "text_semantic_distance")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"device={args.device} ckpt_root={args.ckpt_root} out={out_dir}")

    for ds in args.datasets:
        run_dataset(ds, args, out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "text semantic distance analysis failed")
        raise
