"""
GaussianImageDistribution - Modality Gap Visualization

Generates three publication-quality figures comparing modality gap reduction
across all methods:
  Figure A: t-SNE 2D scatter plots (2x3 grid)
  Figure B: Modality gap distance bar chart
  Figure C: Cosine similarity distribution histograms (2x3 grid)

Usage:
    python scripts/visualize_modality_gap.py
    python scripts/visualize_modality_gap.py --num-samples 2000 --device cuda
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
from models.grove_model import GroVEModel
from models.d2p_model import D2PModel
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
    return _finalize(all_img, all_text, num_samples)


@torch.no_grad()
def extract_clip_finetune(model, dataloader, device, num_samples):
    """CLIP Fine-tune: fine-tuned CLIP, normalized features."""
    model.eval()
    all_img, all_text = [], []
    for pv, ids, mask in _iterate_batches(dataloader, model, device, num_samples):
        img_feat, text_feat = model(images=pv, input_ids=ids, attention_mask=mask, normalize=True)
        all_img.append(img_feat.cpu())
        all_text.append(text_feat.cpu())
    return _finalize(all_img, all_text, num_samples)


@torch.no_grad()
def extract_prolip(model, dataloader, device, num_samples):
    """ProLIP: CLIP + MLP mu heads."""
    model.eval()
    all_img, all_text = [], []
    for pv, ids, mask in _iterate_batches(dataloader, model, device, num_samples):
        ids_3d = ids.unsqueeze(1)
        mask_3d = mask.unsqueeze(1)
        outputs = model(pv, ids_3d, mask_3d)
        img_mu = F.normalize(outputs["img_mu"], dim=-1)
        text_mu = F.normalize(outputs["text_mu"], dim=-1)
        all_img.append(img_mu.cpu())
        all_text.append(text_mu.cpu())
    return _finalize(all_img, all_text, num_samples)


@torch.no_grad()
def extract_grove(model, dataloader, device, num_samples):
    """GroVE: GP posterior mu."""
    model.eval()
    all_img, all_text = [], []
    for pv, ids, mask in _iterate_batches(dataloader, model, device, num_samples):
        ids_3d = ids.unsqueeze(1)
        mask_3d = mask.unsqueeze(1)
        outputs = model(pv, ids_3d, mask_3d)
        img_mu = F.normalize(outputs["img_mu"], dim=-1)
        text_mu = F.normalize(outputs["text_mu"], dim=-1)
        all_img.append(img_mu.cpu())
        all_text.append(text_mu.cpu())
    return _finalize(all_img, all_text, num_samples)


@torch.no_grad()
def extract_d2p(model, dataloader, device, num_samples):
    """D2P: image point + text distribution mu."""
    model.eval()
    all_img, all_text = [], []
    for pv, ids, mask in _iterate_batches(dataloader, model, device, num_samples):
        ids_3d = ids.unsqueeze(1)
        mask_3d = mask.unsqueeze(1)
        outputs = model(pv, ids_3d, mask_3d)
        img_mu = F.normalize(outputs["img_mu"], dim=-1)
        text_mu = F.normalize(outputs["text_mu"], dim=-1)
        all_img.append(img_mu.cpu())
        all_text.append(text_mu.cpu())
    return _finalize(all_img, all_text, num_samples)


@torch.no_grad()
def extract_dist_align(model, dataloader, device, num_samples):
    """Distribution Alignment: img_mu and text_mu, then normalize."""
    model.eval()
    all_img, all_text = [], []
    for pv, ids, mask in _iterate_batches(dataloader, model, device, num_samples):
        # Dist-Align expects (B, K, max_len), reshape to K=1
        ids_3d = ids.unsqueeze(1)
        mask_3d = mask.unsqueeze(1)
        outputs = model(pv, ids_3d, mask_3d)
        img_mu = F.normalize(outputs["img_mu"], dim=-1)
        text_mu = F.normalize(outputs["text_mu"], dim=-1)
        all_img.append(img_mu.cpu())
        all_text.append(text_mu.cpu())
    return _finalize(all_img, all_text, num_samples)


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
        "model_fn": lambda: ProLIPModel(
            freeze_clip=True,
            dropout_rate=config.DIST_ALIGN_DROPOUT_RATE,
        ),
        "checkpoint": str(config.PROLIP_BEST_CKPT),
        "extract_fn": extract_prolip,
    },
    "GroVE": {
        "model_fn": lambda: GroVEModel(
            num_inducing=config.GROVE_NUM_INDUCING,
            freeze_clip=True,
        ),
        "checkpoint": str(config.GROVE_BEST_CKPT),
        "extract_fn": extract_grove,
    },
    "D2P": {
        "model_fn": lambda: D2PModel(
            freeze_clip=True,
            dropout_rate=config.D2P_DROPOUT_RATE,
        ),
        "checkpoint": str(config.D2P_BEST_CKPT),
        "extract_fn": extract_d2p,
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


def plot_tsne_grid(features: Dict, output_dir: Path, perplexity: int = 30):
    """Figure A: 2x3 grid of t-SNE scatter plots."""
    logger.info("Generating Figure A: t-SNE grid...")
    method_order = list(METHOD_CONFIGS.keys())
    n_methods = len(method_order)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, method_name in enumerate(method_order):
        ax = axes[idx]
        img = features[method_name]["img"]
        text = features[method_name]["text"]
        n = img.shape[0]

        # Run t-SNE on concatenated features
        all_feats = np.vstack([img, text])  # (2N, 768)
        tsne = TSNE(n_components=2, perplexity=min(perplexity, n - 1),
                    random_state=42, n_iter=1000)
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

    method_order = list(METHOD_CONFIGS.keys())
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
    """Figure C: 2x3 grid of cosine similarity distribution histograms."""
    logger.info("Generating Figure C: Similarity histograms...")

    method_order = list(METHOD_CONFIGS.keys())
    n_methods = len(method_order)

    # Compute all similarities and global range
    all_sims = {}
    for name in method_order:
        img = features[name]["img"]
        text = features[name]["text"]
        all_sims[name] = np.sum(img * text, axis=1)

    global_min = min(s.min() for s in all_sims.values())
    global_max = max(s.max() for s in all_sims.values())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
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


# =============================================================================
# Main
# =============================================================================

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
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)

    output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR / "modality_gap_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Modality Gap Visualization")
    logger.info("=" * 60)
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("=" * 60)

    # Prepare shared dataset
    captions_path = args.captions_path or config.CAPTIONS_PATH
    images_dir = args.images_dir or config.IMAGES_DIR
    dataloader = prepare_dataloader(args.num_samples, args.batch_size,
                                    captions_path, images_dir)

    # Extract features for all methods
    features = {}
    metrics = {}
    methods = args.methods or list(METHOD_CONFIGS.keys())

    for method_name in methods:
        cfg = METHOD_CONFIGS[method_name]
        logger.info(f"Extracting features for: {method_name}")

        model = cfg["model_fn"]()
        if cfg["checkpoint"] is not None:
            model.load(cfg["checkpoint"])
        model = model.to(args.device)

        img_feat, text_feat = cfg["extract_fn"](model, dataloader, args.device, args.num_samples)
        features[method_name] = {"img": img_feat, "text": text_feat}
        metrics[method_name] = compute_metrics(img_feat, text_feat)
        logger.info(f"  Cosine distance: {metrics[method_name]['mean_cosine_distance']:.6f}")

        del model
        torch.cuda.empty_cache()

    # Save metrics JSON
    metrics_path = output_dir / "modality_gap_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"num_samples": args.num_samples, "methods": metrics}, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Generate figures
    plot_tsne_grid(features, output_dir, perplexity=args.tsne_perplexity)
    plot_gap_bar_chart(features, output_dir)
    plot_similarity_histograms(features, output_dir)

    logger.info(f"All figures saved to: {output_dir}")
    logger.info("Visualization complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Visualization failed")
        raise
