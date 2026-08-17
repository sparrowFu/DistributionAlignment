"""
Exp4: OOD Detection Experiment

Evaluates whether learned σ can distinguish in-domain vs out-of-domain samples.

In-domain: MSCOCO
OOD: SVHN / CIFAR-10 / TinyImageNet

Uses σ_norm or 1-confidence as OOD score.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.caption_dataset import ImageCaptionDataset, filter_none_collate
from models.mcdisp_align_model import MCDispAlignModel
from utils.calibration import compute_auroc, compute_fpr_at_tpr
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


logger = get_logger("eval_ood", config.OOD_LOG_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Exp4: OOD Detection")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ood-datasets", type=str, nargs="+",
                        default=["svhn", "cifar10"],
                        choices=["svhn", "cifar10", "tiny_imagenet"])
    parser.add_argument("--num-in-samples", type=int, default=5000)
    parser.add_argument("--num-ood-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def get_ood_dataset(name: str, num_samples: int, data_dir: Path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145457, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    if name == "svhn":
        dataset = datasets.SVHN(
            root=str(data_dir), split="test",
            download=True, transform=transform,
        )
    elif name == "cifar10":
        dataset = datasets.CIFAR10(
            root=str(data_dir), train=False,
            download=True, transform=transform,
        )
    elif name == "tiny_imagenet":
        # TinyImageNet requires manual download
        dataset = datasets.ImageFolder(
            root=str(data_dir / "tiny-imagenet-200" / "test"),
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown OOD dataset: {name}")

    # Subsample
    if num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = Subset(dataset, indices)

    return dataset


@torch.no_grad()
def extract_sigma_scores(
    model: MCDispAlignModel,
    dataloader: DataLoader,
    device: str,
):
    """Extract σ norm scores for each sample."""
    model.eval()
    scores = []

    for batch in tqdm(dataloader, desc="Extracting σ scores"):
        if batch is None:
            continue

        if isinstance(batch, dict):
            pil_images = batch["image"]
        else:
            # OOD path uses tensors directly (the other overload); skip dict batches here.
            continue

        pixel_values = model.process_images(pil_images).to(device)
        # Use dummy text to get full forward pass
        dummy_text = ["a photo"] * len(pil_images)
        text_inputs = model.process_text(dummy_text)
        input_ids = text_inputs["input_ids"].unsqueeze(1).to(device)  # (B, 1, L)
        attention_mask = text_inputs["attention_mask"].unsqueeze(1).to(device)

        outputs = model(pixel_values, input_ids, attention_mask)

        # σ norm = mean of σ² across dimensions
        sigma_norm = torch.exp(outputs["img_logvar"]).mean(dim=-1)  # (B,)
        scores.append(sigma_norm.cpu())

    if scores:
        return torch.cat(scores, dim=0).numpy()
    return np.array([])


@torch.no_grad()
def extract_sigma_from_tensors(
    model: MCDispAlignModel,
    dataloader: DataLoader,
    device: str,
):
    """Extract σ scores from tensor-based dataset (OOD datasets)."""
    model.eval()
    scores = []

    for batch in tqdm(dataloader, desc="Extracting OOD σ scores"):
        if isinstance(batch, (list, tuple)):
            images = batch[0]  # (B, C, H, W) already transformed
        else:
            continue

        # images are already tensors, pass directly
        images = images.to(device)

        # Get CLIP features
        img_features = model.clip_model.get_image_features(images)
        img_features = img_features.pooler_output

        # Get σ² from the logvar head. Apply the same _floor_logvar mapping the
        # forward pass uses (softplus + VAR_FLOOR) so OOD σ² is scale-consistent
        # with the trained variance — exp() of the raw head output would be wrong.
        img_logvar = model._floor_logvar(model.img_logvar_head(img_features))
        sigma_norm = torch.exp(img_logvar).mean(dim=-1)  # (B,)
        scores.append(sigma_norm.cpu())

    if scores:
        return torch.cat(scores, dim=0).numpy()
    return np.array([])


def main():
    args = parse_args()
    set_seed(config.SEED)

    output_dir = Path(args.output_dir) if args.output_dir else config.OOD_RESULTS_DIR
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
    model.eval()

    # Extract in-distribution σ scores (MSCOCO)
    logger.info("Extracting in-distribution scores from MSCOCO...")
    in_dataset = ImageCaptionDataset(
        captions_path=config.CAPTIONS_PATH,
        images_dir=config.IMAGES_DIR,
        num_captions=1,  # Only need 1 caption for OOD scoring
    )

    # Subsample
    if args.num_in_samples < len(in_dataset):
        indices = np.random.choice(len(in_dataset), args.num_in_samples, replace=False)
        in_dataset = Subset(in_dataset, indices)

    in_dataloader = DataLoader(
        in_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
        collate_fn=filter_none_collate,
    )

    in_scores = extract_sigma_scores(model, in_dataloader, args.device)

    if len(in_scores) == 0:
        logger.error("Failed to extract in-distribution scores")
        return

    logger.info(f"In-distribution scores: mean={in_scores.mean():.4f}, "
                f"std={in_scores.std():.4f}")

    # Evaluate each OOD dataset
    all_results = {}
    ood_data_dir = config.OOD_DATA_DIR
    ood_data_dir.mkdir(parents=True, exist_ok=True)

    for ood_name in args.ood_datasets:
        logger.info(f"\nEvaluating OOD detection on {ood_name}...")

        try:
            ood_dataset = get_ood_dataset(ood_name, args.num_ood_samples, ood_data_dir)
        except Exception as e:
            logger.warning(f"Failed to load {ood_name}: {e}. Skipping.")
            continue

        ood_dataloader = DataLoader(
            ood_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=0,
        )

        # Extract OOD scores
        ood_scores = extract_sigma_from_tensors(model, ood_dataloader, args.device)

        if len(ood_scores) == 0:
            logger.warning(f"No OOD scores for {ood_name}. Skipping.")
            continue

        logger.info(f"OOD scores ({ood_name}): mean={ood_scores.mean():.4f}, "
                    f"std={ood_scores.std():.4f}")

        # Compute AUROC and FPR@95TPR.
        # σ_norm is an OOD score (higher = more likely OOD), but
        # compute_auroc / compute_fpr_at_tpr expect a confidence score
        # (higher = more likely in-distribution; anomaly = 1 - score
        # internally). Negate σ to make it confidence-like.
        min_len = min(len(in_scores), len(ood_scores))
        in_s = in_scores[:min_len]
        out_s = ood_scores[:min_len]

        auroc = compute_auroc(-in_s, -out_s)
        fpr95 = compute_fpr_at_tpr(-in_s, -out_s, target_tpr=0.95)

        # Also compute with 1-confidence interpretation
        # Higher σ → lower confidence → more likely OOD
        in_conf = 1.0 / (1.0 + in_s)
        out_conf = 1.0 / (1.0 + out_s)
        auroc_conf = compute_auroc(in_conf, out_conf)

        results = {
            "ood_dataset": ood_name,
            "num_in": min_len,
            "num_ood": min_len,
            "in_mean": float(in_s.mean()),
            "in_std": float(in_s.std()),
            "ood_mean": float(out_s.mean()),
            "ood_std": float(out_s.std()),
            "auroc_sigma_norm": float(auroc),
            "fpr95_sigma_norm": float(fpr95),
            "auroc_inv_confidence": float(auroc_conf),
        }
        all_results[ood_name] = results

        logger.info(f"[{ood_name}] AUROC(σ)={auroc:.4f}, FPR95={fpr95:.4f}")

    # Save results
    output_path = output_dir / "ood_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Exp4: OOD Detection Results")
    print("=" * 60)
    print(f"{'Dataset':<15} {'AUROC(σ)↑':>12} {'FPR95↓':>10}")
    print("-" * 60)
    for name, r in all_results.items():
        print(f"{name:<15} {r['auroc_sigma_norm']:>12.4f} {r['fpr95_sigma_norm']:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "OOD detection failed")
        raise
