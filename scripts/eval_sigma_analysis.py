"""
GaussianImageDistribution - Exp7: σ Semantic Analysis

Verifies the core hypothesis: σ²_img ≈ Var(μ_captions)

Three sub-experiments:
    A: σ² vs caption diversity correlation (Pearson/Spearman)
    B: σ² variation with caption diversity
    C: σ² t-SNE visualization

Usage:
    python scripts/eval_sigma_analysis.py
    python main.py --task eval_sigma_analysis
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.caption_dataset import ImageCaptionDataset, filter_none_collate
from models.mcdisp_align_model import MCDispAlignModel
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


logger = get_logger("sigma_analysis", config.SIGMA_ANALYSIS_LOG_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Exp7: σ Semantic Analysis")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def extract_all_features(
    model: MCDispAlignModel,
    dataloader: DataLoader,
    device: str,
    num_samples: int,
):
    """Extract all features needed for σ analysis."""
    model.eval()

    all_img_logvar = []
    all_text_mus = []  # Per-caption means
    all_img_mu = []
    all_text_mu = []
    sample_count = 0

    for batch in tqdm(dataloader, desc="Extracting features"):
        if batch is None:
            continue

        pil_images = batch["image"]
        caption_lists = batch["captions"]
        B = len(pil_images)
        K = len(caption_lists[0])

        pixel_values = model.process_images(pil_images).to(device)
        all_captions = [c for cl in caption_lists for c in cl]
        text_inputs = model.process_text(all_captions)
        input_ids = text_inputs["input_ids"].view(B, K, -1).to(device)
        attn_mask = text_inputs["attention_mask"].view(B, K, -1).to(device)

        outputs = model(pixel_values, input_ids, attn_mask)

        all_img_logvar.append(outputs["img_logvar"].cpu())
        all_text_mus.append(outputs["text_mus"].cpu())
        all_img_mu.append(outputs["img_mu"].cpu())
        all_text_mu.append(outputs["text_mu"].cpu())

        sample_count += B
        if sample_count >= num_samples:
            break

    img_logvar = torch.cat(all_img_logvar, dim=0)[:num_samples]
    text_mus = torch.cat(all_text_mus, dim=0)[:num_samples]
    img_mu = torch.cat(all_img_mu, dim=0)[:num_samples]
    text_mu = torch.cat(all_text_mu, dim=0)[:num_samples]

    return img_logvar, text_mus, img_mu, text_mu


def experiment_a_correlation(
    img_logvar: torch.Tensor,
    text_mus: torch.Tensor,
) -> dict:
    """
    Experiment A: σ²_img vs Var(μ_captions) correlation.

    For each image, compute:
    - σ²_img = exp(img_logvar), averaged across dimensions
    - caption_var = Var(μ_caption_1, ..., μ_caption_K), averaged across dimensions

    Then compute Pearson and Spearman correlation.
    """
    # σ²_img: average across dimensions → (N,)
    img_var = torch.exp(img_logvar).mean(dim=-1).numpy()

    # Var(μ_captions) across K captions: (N,)
    caption_var = text_mus.var(dim=1).mean(dim=-1).numpy()

    # Pearson correlation
    pearson_r = np.corrcoef(img_var, caption_var)[0, 1]

    # Spearman rank correlation
    from scipy.stats import spearmanr
    spearman_r, spearman_p = spearmanr(img_var, caption_var)

    # Also compute per-dimension correlation
    img_var_perdim = torch.exp(img_logvar).numpy()  # (N, D)
    caption_var_perdim = text_mus.var(dim=1).numpy()  # (N, D)
    dim_correlations = []
    for d in range(img_var_perdim.shape[1]):
        r = np.corrcoef(img_var_perdim[:, d], caption_var_perdim[:, d])[0, 1]
        dim_correlations.append(r)
    mean_dim_corr = np.mean(dim_correlations)

    results = {
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "mean_per_dim_correlation": float(mean_dim_corr),
        "img_var_mean": float(img_var.mean()),
        "img_var_std": float(img_var.std()),
        "caption_var_mean": float(caption_var.mean()),
        "caption_var_std": float(caption_var.std()),
    }

    logger.info(f"Experiment A: Pearson={pearson_r:.4f}, Spearman={spearman_r:.4f}")
    return results


def experiment_b_diversity(
    img_logvar: torch.Tensor,
    text_mus: torch.Tensor,
) -> dict:
    """
    Experiment B: σ² variation with caption diversity.

    Group samples by caption diversity (measured by pairwise cosine distance
    of caption means) and analyze how σ² changes.
    """
    N, K, D = text_mus.shape
    img_var = torch.exp(img_logvar).mean(dim=-1).numpy()  # (N,)

    # Compute caption diversity: average pairwise cosine distance
    caption_diversity = np.zeros(N)
    for i in range(N):
        mus_i = text_mus[i]  # (K, D)
        mus_i_norm = F.normalize(mus_i, dim=-1)
        sim_matrix = torch.matmul(mus_i_norm, mus_i_norm.T)
        # Average pairwise distance = 1 - average pairwise similarity
        mask = ~torch.eye(K, dtype=torch.bool)
        avg_sim = sim_matrix[mask].mean().item()
        caption_diversity[i] = 1.0 - avg_sim

    # Divide into quantile groups
    num_groups = 5
    quantiles = np.quantile(caption_diversity, np.linspace(0, 1, num_groups + 1))
    groups = {}
    for g in range(num_groups):
        mask = (caption_diversity >= quantiles[g]) & (caption_diversity < quantiles[g + 1])
        if mask.sum() > 0:
            groups[g] = {
                "diversity_range": f"[{quantiles[g]:.4f}, {quantiles[g+1]:.4f})",
                "count": int(mask.sum()),
                "mean_sigma2": float(img_var[mask].mean()),
                "std_sigma2": float(img_var[mask].std()),
            }

    # Correlation between diversity and σ²
    diversity_sigma_corr = np.corrcoef(caption_diversity, img_var)[0, 1]

    results = {
        "diversity_sigma_correlation": float(diversity_sigma_corr),
        "groups": groups,
    }

    logger.info(f"Experiment B: Diversity-σ² correlation={diversity_sigma_corr:.4f}")
    return results


def experiment_c_visualization(
    img_mu: torch.Tensor,
    img_logvar: torch.Tensor,
    output_dir: Path,
    num_vis: int = 1000,
) -> dict:
    """
    Experiment C: Prepare t-SNE data colored by σ².

    Saves image μ embeddings and σ² values for visualization.
    """
    # Subsample for visualization
    n = min(num_vis, img_mu.shape[0])
    indices = np.random.choice(img_mu.shape[0], n, replace=False)

    vis_mu = img_mu[indices].numpy()
    vis_sigma2 = torch.exp(img_logvar[indices]).mean(dim=-1).numpy()

    # Save for external visualization
    vis_path = output_dir / "tsne_data.npz"
    np.savez(vis_path, mu=vis_mu, sigma2=vis_sigma2)

    # Statistics
    results = {
        "num_samples": n,
        "sigma2_min": float(vis_sigma2.min()),
        "sigma2_max": float(vis_sigma2.max()),
        "sigma2_mean": float(vis_sigma2.mean()),
        "sigma2_median": float(np.median(vis_sigma2)),
        "visualization_saved": str(vis_path),
    }

    logger.info(f"Experiment C: Saved t-SNE data to {vis_path}")
    return results


def main():
    args = parse_args()
    set_seed(config.SEED)

    output_dir = Path(args.output_dir) if args.output_dir else config.SIGMA_ANALYSIS_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    checkpoint_path = args.checkpoint or str(config.MCDISP_ALIGN_BEST_CKPT)
    logger.info(f"Loading model from {checkpoint_path}")

    model = MCDispAlignModel(
        freeze_clip=config.MCDISP_ALIGN_FREEZE_CLIP,
        distribution_merging=config.MCDISP_ALIGN_DISTRIBUTION_MERGING,
    )
    if Path(checkpoint_path).exists():
        model.load(checkpoint_path)
    model = model.to(args.device)

    # Load dataset
    dataset = ImageCaptionDataset(
        captions_path=config.CAPTIONS_PATH,
        images_dir=config.IMAGES_DIR,
        num_captions=5,
    )

    # Subsample
    if args.num_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:args.num_samples].tolist()
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
        collate_fn=filter_none_collate,
    )

    # Extract features
    logger.info("Extracting features...")
    img_logvar, text_mus, img_mu, text_mu = extract_all_features(
        model, dataloader, args.device, args.num_samples,
    )

    logger.info(f"Features extracted: {img_logvar.shape[0]} samples")

    # Run experiments
    all_results = {}

    logger.info("\n=== Experiment A: σ² vs Caption Diversity Correlation ===")
    all_results["experiment_a"] = experiment_a_correlation(img_logvar, text_mus)

    logger.info("\n=== Experiment B: σ² Variation with Diversity ===")
    all_results["experiment_b"] = experiment_b_diversity(img_logvar, text_mus)

    logger.info("\n=== Experiment C: t-SNE Visualization Data ===")
    all_results["experiment_c"] = experiment_c_visualization(img_mu, img_logvar, output_dir)

    # Save results
    output_path = output_dir / "sigma_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Exp7: σ Semantic Analysis Results")
    print("=" * 60)
    r_a = all_results["experiment_a"]
    print(f"\nExperiment A: σ² vs Caption Diversity")
    print(f"  Pearson r:  {r_a['pearson_r']:.4f}")
    print(f"  Spearman r: {r_a['spearman_r']:.4f} (p={r_a['spearman_p']:.2e})")
    print(f"  Per-dim r:  {r_a['mean_per_dim_correlation']:.4f}")

    r_b = all_results["experiment_b"]
    print(f"\nExperiment B: Diversity Groups")
    print(f"  Diversity-σ² correlation: {r_b['diversity_sigma_correlation']:.4f}")
    for g, info in r_b["groups"].items():
        print(f"  Group {g} ({info['diversity_range']}): "
              f"σ²={info['mean_sigma2']:.4f} ± {info['std_sigma2']:.4f}")

    r_c = all_results["experiment_c"]
    print(f"\nExperiment C: Visualization data saved to {r_c['visualization_saved']}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "σ analysis failed")
        raise
