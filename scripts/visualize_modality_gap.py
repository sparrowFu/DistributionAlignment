"""
GaussianImageDistribution - Modality Gap Visualization

Generates publication-quality figures comparing modality gap reduction
across all methods:
  Figure A: t-SNE 2D scatter plots
  Figure B: Modality gap distance bar chart
  Figure C: Cosine similarity distribution histograms
  Figure D: Variance magnitude distribution histograms (distribution models only)
  Figure E: Per-sample variance vs cosine similarity scatter (distribution models only)
  Figure F: Per-dimension variance profile bar chart (distribution models only)

Usage:
    python scripts/visualize_modality_gap.py
    python scripts/visualize_modality_gap.py --num-samples 2000 --device cuda
    python scripts/visualize_modality_gap.py --model-type dist_align
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.caption_dataset import ImageCaptionDataset, filter_none_collate
from models.clip_baseline import CLIPFineTuneBaseline
from models.dist_align_model import DistributionAlignmentModel
from models.prolip_model import ProLIPModel
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


logger = get_logger("visualize_gap", config.LOG_DIR / "visualize_modality_gap.log")


# =============================================================================
# Data Loading
# =============================================================================

def prepare_dataloader(num_samples, batch_size, captions_path, images_dir):
    """Prepare a shared dataloader with a fixed random subset."""
    dataset = ImageCaptionDataset(
        captions_path=captions_path,
        images_dir=images_dir,
        num_captions=config.NUM_CAPTIONS,
    )
    if num_samples and num_samples < len(dataset):
        generator = torch.Generator().manual_seed(config.SEED)
        indices = torch.randperm(len(dataset), generator=generator)[:num_samples].tolist()
        dataset = Subset(dataset, indices)
        logger.info(f"Using {num_samples} samples (random subset)")
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=filter_none_collate,
    )
    return dataloader


# =============================================================================
# Feature Extraction Helpers
# =============================================================================

def _iterate_batches(dataloader, model, device, num_samples):
    """Yield (pixel_values, input_ids, attention_mask) from dataloader."""
    sample_count = 0
    for batch in dataloader:
        if batch is None:
            continue
        pil_images = batch["image"]
        caption_lists = batch["captions"]
        selected_captions = [caps[0] for caps in caption_lists]

        pixel_values = model.process_images(pil_images).to(device)
        text_inputs = model.process_text(selected_captions)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        yield pixel_values, input_ids, attention_mask

        sample_count += len(pil_images)
        if num_samples and sample_count >= num_samples:
            break


def _finalize(all_img, all_text, num_samples):
    """Concatenate, truncate, convert to numpy."""
    img = torch.cat(all_img, dim=0)[:num_samples].numpy()
    text = torch.cat(all_text, dim=0)[:num_samples].numpy()
    return img, text


# =============================================================================
# Per-Method Feature Extraction
# =============================================================================

@torch.no_grad()
def extract_clip_zero_shot(model, dataloader, device, num_samples):
    """CLIP Zero-Shot: frozen CLIP, normalized features."""
    model.eval()
    all_img, all_text = [], []
    for pv, ids, mask in _iterate_batches(dataloader, model, device, num_samples):
        img_feat, text_feat = model(images=pv, input_ids=ids, attention_mask=mask, normalize=True)
        all_img.append(img_feat.cpu())
        all_text.append(text_feat.cpu())
    img_np, text_np = _finalize(all_img, all_text, num_samples)
    return {"img": img_np, "text": text_np, "img_logvar": None, "text_logvar": None}


@torch.no_grad()
def extract_clip_finetune(model, dataloader, device, num_samples):
    """CLIP Fine-tune: fine-tuned CLIP, normalized features."""
    model.eval()
    all_img, all_text = [], []
    for pv, ids, mask in _iterate_batches(dataloader, model, device, num_samples):
        img_feat, text_feat = model(images=pv, input_ids=ids, attention_mask=mask, normalize=True)
        all_img.append(img_feat.cpu())
        all_text.append(text_feat.cpu())
    img_np, text_np = _finalize(all_img, all_text, num_samples)
    return {"img": img_np, "text": text_np, "img_logvar": None, "text_logvar": None}


@torch.no_grad()
def extract_prolip(model, dataloader, device, num_samples):
    """ProLIP: real ViT-H/14 mu / log-variance heads."""
    model.eval()
    all_img, all_text, all_img_lv, all_text_lv = [], [], [], []
    for pv, ids, mask in _iterate_batches(dataloader, model, device, num_samples):
        ids_3d = ids.unsqueeze(1)
        mask_3d = mask.unsqueeze(1)
        outputs = model(pv, ids_3d, mask_3d)
        img_mu = F.normalize(outputs["img_mu"], dim=-1)
        text_mu = F.normalize(outputs["text_mu"], dim=-1)
        all_img.append(img_mu.cpu())
        all_text.append(text_mu.cpu())
        all_img_lv.append(outputs["img_logvar"].cpu())
        all_text_lv.append(outputs["text_logvar"].cpu())
    img_np, text_np = _finalize(all_img, all_text, num_samples)
    img_lv = torch.cat(all_img_lv, dim=0)[:num_samples].numpy()
    text_lv = torch.cat(all_text_lv, dim=0)[:num_samples].numpy()
    return {"img": img_np, "text": text_np, "img_logvar": img_lv, "text_logvar": text_lv}


@torch.no_grad()
def extract_dist_align(model, dataloader, device, num_samples):
    """Distribution Alignment: img_mu and text_mu, then normalize."""
    model.eval()
    all_img, all_text, all_img_lv, all_text_lv = [], [], [], []
    for pv, ids, mask in _iterate_batches(dataloader, model, device, num_samples):
        # Dist-Align expects (B, K, max_len), reshape to K=1
        ids_3d = ids.unsqueeze(1)
        mask_3d = mask.unsqueeze(1)
        outputs = model(pv, ids_3d, mask_3d)
        img_mu = F.normalize(outputs["img_mu"], dim=-1)
        text_mu = F.normalize(outputs["text_mu"], dim=-1)
        all_img.append(img_mu.cpu())
        all_text.append(text_mu.cpu())
        all_img_lv.append(outputs["img_logvar"].cpu())
        all_text_lv.append(outputs["text_logvar"].cpu())
    img_np, text_np = _finalize(all_img, all_text, num_samples)
    img_lv = torch.cat(all_img_lv, dim=0)[:num_samples].numpy()
    text_lv = torch.cat(all_text_lv, dim=0)[:num_samples].numpy()
    return {"img": img_np, "text": text_np, "img_logvar": img_lv, "text_logvar": text_lv}


# =============================================================================
# Method Configuration
# =============================================================================

METHOD_CONFIGS = {
    "CLIP Zero-Shot": {
        "model_fn": lambda: CLIPFineTuneBaseline(freeze_image=True, freeze_text=True),
        "checkpoint": None,
        "extract_fn": extract_clip_zero_shot,
    },
    "CLIP Fine-tune": {
        "model_fn": lambda: CLIPFineTuneBaseline(),
        "checkpoint": str(config.CLIP_BASELINE_BEST_CKPT),
        "extract_fn": extract_clip_finetune,
    },
    "ProLIP": {
        "model_fn": lambda: ProLIPModel(freeze=True),
        "checkpoint": str(config.PROLIP_BEST_CKPT),
        "extract_fn": extract_prolip,
    },
    "Ours": {
        "model_fn": lambda: DistributionAlignmentModel(
            freeze_clip=config.DIST_ALIGN_FREEZE_CLIP,
            distribution_merging=config.DIST_ALIGN_DISTRIBUTION_MERGING,
        ),
        "checkpoint": str(config.DIST_ALIGN_BEST_CKPT),
        "extract_fn": extract_dist_align,
    },
}


# =============================================================================
# Visualization Functions
# =============================================================================

def compute_metrics(img: np.ndarray, text: np.ndarray) -> Dict[str, float]:
    """Compute cosine distance metrics between matched image-text pairs."""
    cos_sims = np.sum(img * text, axis=1)  # (N,)
    cos_distances = 1.0 - cos_sims
    return {
        "mean_cosine_similarity": float(cos_sims.mean()),
        "mean_cosine_distance": float(cos_distances.mean()),
        "std_cosine_distance": float(cos_distances.std()),
    }


def _compute_grid(n_methods):
    """Compute (n_rows, n_cols) for a grid with at most 3 columns."""
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    return n_rows, n_cols


def plot_tsne_grid(features: Dict, output_dir: Path, perplexity: int = 30):
    """Figure A: grid of t-SNE scatter plots."""
    logger.info("Generating Figure A: t-SNE grid...")
    method_order = list(features.keys())
    n_methods = len(method_order)
    n_rows, n_cols = _compute_grid(n_methods)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    if n_methods == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, method_name in enumerate(method_order):
        ax = axes[idx]
        img = features[method_name]["img"]
        text = features[method_name]["text"]
        n = img.shape[0]

        # Run t-SNE on concatenated features
        all_feats = np.vstack([img, text])  # (2N, 768)
        tsne = TSNE(n_components=2, perplexity=min(perplexity, n - 1),
                    random_state=42, max_iter=1000)
        embedded = tsne.fit_transform(all_feats)

        img_2d = embedded[:n]
        text_2d = embedded[n:]

        # Plot
        ax.scatter(img_2d[:, 0], img_2d[:, 1], c="#4A90D9", alpha=0.4, s=8, label="Image")
        ax.scatter(text_2d[:, 0], text_2d[:, 1], c="#E74C3C", alpha=0.4, s=8, label="Text")

        # Centroids and gap line
        img_center = img_2d.mean(axis=0)
        text_center = text_2d.mean(axis=0)
        ax.plot([img_center[0], text_center[0]], [img_center[1], text_center[1]],
                "k--", alpha=0.7, linewidth=1.5)

        # Gap value (in original space)
        gap = 1.0 - np.mean(np.sum(img * text, axis=1))
        ax.set_title(f"{method_name}\nGap = {gap:.4f}", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.legend(fontsize=9, loc="upper right")

    # Hide unused subplot (if any)
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("t-SNE Visualization of Modality Gap", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(output_dir / "fig_a_tsne_grid.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_a_tsne_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Figure A saved.")


def plot_gap_bar_chart(features: Dict, output_dir: Path):
    """Figure B: Horizontal bar chart of average cosine distance."""
    logger.info("Generating Figure B: Gap bar chart...")

    method_order = list(features.keys())
    names, means, stds = [], [], []
    for name in method_order:
        m = compute_metrics(features[name]["img"], features[name]["text"])
        names.append(name)
        means.append(m["mean_cosine_distance"])
        stds.append(m["std_cosine_distance"])

    # Sort by gap (ascending: best at top)
    sorted_idx = np.argsort(means)
    names = [names[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]

    colors = ["#7FB3D8"] * len(names)
    for i, n in enumerate(names):
        if n == "Ours":
            colors[i] = "#E67E22"

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, means, xerr=stds, color=colors,
                   edgecolor="black", linewidth=0.5, capsize=4)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{mean:.4f}", va="center", fontsize=10)

    ax.set_xlabel("Average Cosine Distance (lower = smaller gap)", fontsize=12)
    ax.set_title("Modality Gap Distance Across Methods", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_dir / "fig_b_gap_bar_chart.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_b_gap_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Figure B saved.")


def plot_similarity_histograms(features: Dict, output_dir: Path):
    """Figure C: grid of cosine similarity distribution histograms."""
    logger.info("Generating Figure C: Similarity histograms...")

    method_order = list(features.keys())
    n_methods = len(method_order)
    n_rows, n_cols = _compute_grid(n_methods)

    # Compute all similarities and global range
    all_sims = {}
    for name in method_order:
        img = features[name]["img"]
        text = features[name]["text"]
        all_sims[name] = np.sum(img * text, axis=1)

    global_min = min(s.min() for s in all_sims.values())
    global_max = max(s.max() for s in all_sims.values())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_methods == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, method_name in enumerate(method_order):
        ax = axes[idx]
        sims = all_sims[method_name]

        ax.hist(sims, bins=50, color="#4A90D9", alpha=0.7,
                edgecolor="white", density=True)
        ax.axvline(sims.mean(), color="#E74C3C", linestyle="--", linewidth=2,
                   label=f"Mean = {sims.mean():.4f}")
        ax.set_xlim(global_min - 0.02, global_max + 0.02)
        ax.set_xlabel("Cosine Similarity", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(method_name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Distribution of Cosine Similarities (Image-Text Pairs)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(output_dir / "fig_c_similarity_histograms.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_c_similarity_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Figure C saved.")


def _get_dist_methods(features):
    """Return list of method names that have logvar (distribution models)."""
    return [
        name for name, feat in features.items()
        if feat.get("img_logvar") is not None or feat.get("text_logvar") is not None
    ]


def plot_variance_histograms(features: Dict, output_dir: Path):
    """Figure D: Variance magnitude distribution histograms.

    For each distribution model, plot histograms of per-sample average variance
    (exp(logvar)) for both image and text modalities.
    """
    logger.info("Generating Figure D: Variance histograms...")
    dist_methods = _get_dist_methods(features)

    if not dist_methods:
        logger.info("  No distribution models found. Skipping Figure D.")
        return

    n_methods = len(dist_methods)
    n_rows, n_cols = _compute_grid(n_methods)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_methods == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, method_name in enumerate(dist_methods):
        ax = axes[idx]
        feat = features[method_name]

        if feat["img_logvar"] is not None:
            img_var = np.exp(feat["img_logvar"])
            img_mean_var = img_var.mean(axis=1)  # per-sample average
            ax.hist(img_mean_var, bins=50, alpha=0.6, color="#4A90D9",
                    edgecolor="white", density=True, label="Image")

        if feat["text_logvar"] is not None:
            text_var = np.exp(feat["text_logvar"])
            text_mean_var = text_var.mean(axis=1)  # per-sample average
            ax.hist(text_mean_var, bins=50, alpha=0.6, color="#E74C3C",
                    edgecolor="white", density=True, label="Text")

        ax.set_xlabel("Mean Variance (exp(logvar))", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(method_name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Variance Magnitude Distribution (Distribution Models)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(output_dir / "fig_d_variance_histograms.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_d_variance_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Figure D saved.")


def plot_variance_vs_similarity(features: Dict, output_dir: Path):
    """Figure E: Per-sample variance vs cosine similarity scatter plot.

    x-axis: per-sample average variance, y-axis: cosine similarity.
    Annotate with Pearson correlation coefficient r.
    """
    logger.info("Generating Figure E: Variance vs similarity scatter...")
    dist_methods = _get_dist_methods(features)

    if not dist_methods:
        logger.info("  No distribution models found. Skipping Figure E.")
        return

    n_methods = len(dist_methods)
    n_rows, n_cols = _compute_grid(n_methods)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_methods == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, method_name in enumerate(dist_methods):
        ax = axes[idx]
        feat = features[method_name]
        img = feat["img"]
        text = feat["text"]

        # Cosine similarity per sample
        cos_sims = np.sum(img * text, axis=1)

        # Average variance per sample across both modalities
        vars_list = []
        if feat["img_logvar"] is not None:
            vars_list.append(np.exp(feat["img_logvar"]).mean(axis=1))
        if feat["text_logvar"] is not None:
            vars_list.append(np.exp(feat["text_logvar"]).mean(axis=1))

        avg_var = np.mean(np.stack(vars_list, axis=0), axis=0)

        ax.scatter(avg_var, cos_sims, alpha=0.3, s=6, c="#4A90D9")

        # Pearson correlation
        r = np.corrcoef(avg_var, cos_sims)[0, 1]
        ax.set_xlabel("Per-sample Mean Variance", fontsize=10)
        ax.set_ylabel("Cosine Similarity", fontsize=10)
        ax.set_title(f"{method_name}\nr = {r:.4f}", fontsize=12, fontweight="bold")

    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Variance vs Cosine Similarity (Distribution Models)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(output_dir / "fig_e_variance_vs_similarity.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_e_variance_vs_similarity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Figure E saved.")


def plot_variance_profile(features: Dict, output_dir: Path):
    """Figure F: Per-dimension variance profile bar chart.

    Show top-50 dimensions with highest variance, comparing image vs text.
    """
    logger.info("Generating Figure F: Variance profile...")
    dist_methods = _get_dist_methods(features)

    if not dist_methods:
        logger.info("  No distribution models found. Skipping Figure F.")
        return

    n_methods = len(dist_methods)
    n_rows, n_cols = _compute_grid(n_methods)
    top_k = 50

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_methods == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, method_name in enumerate(dist_methods):
        ax = axes[idx]
        feat = features[method_name]

        # Compute per-dimension average variance
        dims_list = []
        if feat["img_logvar"] is not None:
            dims_list.append(np.exp(feat["img_logvar"]).mean(axis=0))
        if feat["text_logvar"] is not None:
            dims_list.append(np.exp(feat["text_logvar"]).mean(axis=0))

        # Use the maximum across modalities to pick top dimensions
        combined = np.max(np.stack(dims_list), axis=0)
        top_dims = np.argsort(combined)[-top_k:][::-1]

        dim_indices = np.arange(top_k)
        width = 0.35

        if feat["img_logvar"] is not None:
            img_dim_var = np.exp(feat["img_logvar"]).mean(axis=0)[top_dims]
            ax.bar(dim_indices - width / 2, img_dim_var, width,
                   color="#4A90D9", alpha=0.7, label="Image")

        if feat["text_logvar"] is not None:
            text_dim_var = np.exp(feat["text_logvar"]).mean(axis=0)[top_dims]
            ax.bar(dim_indices + width / 2, text_dim_var, width,
                   color="#E74C3C", alpha=0.7, label="Text")

        ax.set_xlabel(f"Top-{top_k} Dimensions", fontsize=10)
        ax.set_ylabel("Mean Variance", fontsize=10)
        ax.set_title(method_name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Per-Dimension Variance Profile (Top {top_k})",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(output_dir / "fig_f_variance_profile.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_f_variance_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Figure F saved.")


# =============================================================================
# Main
# =============================================================================

MODEL_TYPE_ALIASES = {
    "dist_align": "Ours",
    "clip_zero": "CLIP Zero-Shot",
    "clip_baseline": "CLIP Fine-tune",
    "prolip": "ProLIP",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Modality Gap Across Methods")
    parser.add_argument("--num-samples", type=int, default=3000,
                        help="Number of samples to use (default: 3000)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tsne-perplexity", type=int, default=30)
    parser.add_argument("--captions-path", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        choices=list(METHOD_CONFIGS.keys()),
                        help="Subset of methods to visualize (default: all)")
    parser.add_argument("--model-type", type=str, default=None,
                        choices=list(MODEL_TYPE_ALIASES.keys()),
                        help="Shortcut to select a single model by alias "
                             "(e.g. dist_align, clip_zero). Overrides --methods.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)

    output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR / "modality_gap_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --model-type takes priority over --methods
    if args.model_type is not None:
        methods = [MODEL_TYPE_ALIASES[args.model_type]]
    else:
        methods = args.methods or list(METHOD_CONFIGS.keys())

    logger.info("=" * 60)
    logger.info("Modality Gap Visualization")
    logger.info("=" * 60)
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Methods: {methods}")
    logger.info("=" * 60)

    # Prepare shared dataset
    captions_path = args.captions_path or config.CAPTIONS_PATH
    images_dir = args.images_dir or config.IMAGES_DIR
    dataloader = prepare_dataloader(args.num_samples, args.batch_size,
                                    captions_path, images_dir)

    # Extract features for all methods
    features = {}
    metrics = {}

    for method_name in methods:
        cfg = METHOD_CONFIGS[method_name]
        logger.info(f"Extracting features for: {method_name}")

        # Check if checkpoint exists before loading
        checkpoint_path = cfg.get("checkpoint")
        if checkpoint_path is not None and not Path(checkpoint_path).exists():
            logger.warning(f"Skipping '{method_name}': checkpoint not found at {checkpoint_path}")
            continue

        model = cfg["model_fn"]()
        if cfg["checkpoint"] is not None:
            model.load(cfg["checkpoint"])
        model = model.to(args.device)

        result = cfg["extract_fn"](model, dataloader, args.device, args.num_samples)
        features[method_name] = result
        metrics[method_name] = compute_metrics(result["img"], result["text"])
        logger.info(f"  Cosine distance: {metrics[method_name]['mean_cosine_distance']:.6f}")

        del model
        torch.cuda.empty_cache()

    if not features:
        logger.warning("No features extracted. Exiting.")
        return

    # Save metrics JSON
    metrics_path = output_dir / "modality_gap_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"num_samples": args.num_samples, "methods": metrics}, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Generate figures
    plot_tsne_grid(features, output_dir, perplexity=args.tsne_perplexity)
    plot_gap_bar_chart(features, output_dir)
    plot_similarity_histograms(features, output_dir)

    # Distribution-specific figures (D/E/F)
    plot_variance_histograms(features, output_dir)
    plot_variance_vs_similarity(features, output_dir)
    plot_variance_profile(features, output_dir)

    logger.info(f"All figures saved to: {output_dir}")
    logger.info("Visualization complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Visualization failed")
        raise
