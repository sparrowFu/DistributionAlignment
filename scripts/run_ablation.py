"""
GaussianImageDistribution - Exp5: Ablation Study Script

Quantifies the contribution of each MSDA loss component by training with
different configurations and evaluating on retrieval.

Ablation configurations (see config.ABLATION_CONFIGS):
    - Full MSDA (set-NCE + mu + var + cover + cov + reg)
    - w/o L_var (variance semantic consistency)
    - w/o L_cover (multi-caption coverage)
    - w/o L_cov (covariance direction)
    - w/o L_mu (mean-center alignment)
    - diagonal only (cov_rank=0)
    - w/o uncertainty-discounted similarity (standard cosine)
    - K = 1 / 3 / 5 captions
    - lambda_var sensitivity: 0.1 / 0.5 / 1.0 / 2.0 / 5.0
    - lambda_cover sensitivity: 0.1 / 0.5 / 1.0 / 2.0
    - tau sensitivity: 0.05 / 0.07 / 0.1 / 0.2

Usage:
    python scripts/run_ablation.py --config all
    python scripts/run_ablation.py --config no_var
    python main.py --task run_ablation --config all
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.caption_dataset import ImageCaptionDataset, filter_none_collate
from data.vqa_dataset import VQADataset, vqa_collate_fn
from models.dist_align_model import DistributionAlignmentModel
from models.vqa_model import VQAModel
from losses.dist_align_losses import MSDALoss
from utils.logger import get_logger, log_exception
from utils.seed import set_seed
from utils.metrics import compute_recall_at_k

import torch.optim as optim
from tqdm import tqdm


logger = get_logger("ablation", config.ABLATION_LOG_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Exp5: Ablation Study")
    parser.add_argument("--config", type=str, default="all",
                        choices=list(config.ABLATION_CONFIGS.keys()) + ["all", "sensitivity"],
                        help="Ablation configuration to run")
    parser.add_argument("--epochs", type=int, default=config.DIST_ALIGN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DIST_ALIGN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.DIST_ALIGN_MLP_LR)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only evaluate existing checkpoints")
    return parser.parse_args()


def train_ablation(
    config_name: str,
    ablation_config: dict,
    args,
    output_dir: Path,
):
    """Train and evaluate a single ablation configuration."""
    set_seed(config.SEED)
    logger.info(f"\n{'='*60}")
    logger.info(f"Ablation: {config_name}")
    logger.info(f"Config: {ablation_config}")
    logger.info(f"{'='*60}")

    ckpt_dir = output_dir / "checkpoints" / config_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best.pt"

    # Skip training if requested
    if not args.skip_training:
        # Determine MSDA configuration from the ablation overrides
        cov_rank = ablation_config.get("cov_rank", config.MSDA_COV_RANK)
        num_captions = ablation_config.get("num_captions", 5)
        use_uncertainty_sim = ablation_config.get("use_uncertainty_sim", True)

        # Create model
        model = DistributionAlignmentModel(
            freeze_clip=True,
            distribution_merging=config.DIST_ALIGN_DISTRIBUTION_MERGING,
            dropout_rate=config.DIST_ALIGN_DROPOUT_RATE,
            cov_rank=cov_rank,
        )
        model = model.to(args.device)
        logger.info(f"Trainable params: {model.num_trainable_parameters():,}")

        # MSDA loss
        criterion = MSDALoss(
            lambda_ctr=ablation_config.get("lambda_ctr", config.MSDA_LAMBDA_CTR),
            lambda_mu=ablation_config.get("lambda_mu", config.MSDA_LAMBDA_MU),
            lambda_var=ablation_config.get("lambda_var", config.MSDA_LAMBDA_VAR),
            lambda_cover=ablation_config.get("lambda_cover", config.MSDA_LAMBDA_COVER),
            lambda_cov=ablation_config.get("lambda_cov", config.MSDA_LAMBDA_COV),
            lambda_reg=ablation_config.get("lambda_reg", config.MSDA_LAMBDA_REG),
            tau=ablation_config.get("temperature", config.MSDA_TAU),
            m_pos=config.MSDA_M_POS,
            target_var=config.MSDA_TARGET_VAR,
            m_neg=config.MSDA_M_NEG,
            use_uncertainty_sim=use_uncertainty_sim,
        )

        optimizer = optim.Adam(
            model.trainable_parameters(),
            lr=args.lr,
            weight_decay=config.DIST_ALIGN_WEIGHT_DECAY,
        )

        # Load dataset
        dataset = ImageCaptionDataset(
            captions_path=config.CAPTIONS_PATH,
            images_dir=config.IMAGES_DIR,
            num_captions=num_captions,
        )

        val_size = int(len(dataset) * 0.1)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(config.SEED),
        )

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=filter_none_collate,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=filter_none_collate,
        )

        # Training loop
        best_val_loss = float("inf")
        patience = 0

        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0

            for batch in tqdm(train_loader, desc=f"[{config_name}] Epoch {epoch+1}"):
                if batch is None:
                    continue

                pil_images = batch["image"]
                caption_lists = batch["captions"]
                B = len(pil_images)
                K = len(caption_lists[0])

                pixel_values = model.process_images(pil_images).to(args.device)
                all_captions = [c for cl in caption_lists for c in cl]
                text_inputs = model.process_text(all_captions)
                input_ids = text_inputs["input_ids"].view(B, K, -1).to(args.device)
                attn_mask = text_inputs["attention_mask"].view(B, K, -1).to(args.device)

                outputs = model(pixel_values, input_ids, attn_mask)

                loss, loss_dict = criterion(
                    outputs["img_mu"], outputs["img_logvar"], outputs["img_U"],
                    outputs["text_mu"], outputs["text_logvar"],
                    outputs["text_mus"], outputs["text_logvars"], outputs["text_Us"],
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss_dict["total"]

            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}")

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    pil_images = batch["image"]
                    caption_lists = batch["captions"]
                    B = len(pil_images)
                    K = len(caption_lists[0])

                    pixel_values = model.process_images(pil_images).to(args.device)
                    all_captions = [c for cl in caption_lists for c in cl]
                    text_inputs = model.process_text(all_captions)
                    input_ids = text_inputs["input_ids"].view(B, K, -1).to(args.device)
                    attn_mask = text_inputs["attention_mask"].view(B, K, -1).to(args.device)

                    outputs = model(pixel_values, input_ids, attn_mask)
                    loss, loss_dict = criterion(
                        outputs["img_mu"], outputs["img_logvar"], outputs["img_U"],
                        outputs["text_mu"], outputs["text_logvar"],
                        outputs["text_mus"], outputs["text_logvars"], outputs["text_Us"],
                    )
                    val_loss += loss_dict["total"]

            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Val loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model.save(str(best_ckpt_path))
                patience = 0
            else:
                patience += 1
                if patience >= 3:
                    logger.info("Early stopping")
                    break

    # Evaluation: Retrieval Recall@K
    results = {"config": config_name, "description": ablation_config.get("description", "")}

    if best_ckpt_path.exists():
        model = DistributionAlignmentModel(
            freeze_clip=True,
            distribution_merging=config.DIST_ALIGN_DISTRIBUTION_MERGING,
            cov_rank=ablation_config.get("cov_rank", config.MSDA_COV_RANK),
        )
        model.load(str(best_ckpt_path))
        model = model.to(args.device)
        model.eval()

        # Quick retrieval evaluation
        eval_dataset = ImageCaptionDataset(
            captions_path=config.CAPTIONS_PATH,
            images_dir=config.IMAGES_DIR,
            num_captions=5,
        )
        # Subsample for speed
        indices = list(range(min(500, len(eval_dataset))))
        from torch.utils.data import Subset
        eval_subset = Subset(eval_dataset, indices)
        eval_loader = DataLoader(
            eval_subset, batch_size=args.batch_size,
            shuffle=False, num_workers=0,
            collate_fn=filter_none_collate,
        )

        all_img_mu, all_text_mu = [], []
        with torch.no_grad():
            for batch in eval_loader:
                if batch is None:
                    continue
                pv = model.process_images(batch["image"]).to(args.device)
                caps = [c for cl in batch["captions"] for c in cl]
                B = len(batch["image"])
                K = len(batch["captions"][0])
                ti = model.process_text(caps)
                ids = ti["input_ids"].view(B, K, -1).to(args.device)
                am = ti["attention_mask"].view(B, K, -1).to(args.device)
                out = model(pv, ids, am)
                all_img_mu.append(out["img_mu"].cpu())
                all_text_mu.append(out["text_mu"].cpu())

        img_mu = torch.cat(all_img_mu)
        text_mu = torch.cat(all_text_mu)
        img_mu = torch.nn.functional.normalize(img_mu, dim=-1)
        text_mu = torch.nn.functional.normalize(text_mu, dim=-1)
        sim = torch.matmul(img_mu, text_mu.T)
        recalls = compute_recall_at_k(sim, config.RECALL_AT_K)
        results["retrieval"] = {f"R@{k}": v for k, v in recalls.items()}

    return results


def run_sensitivity_analysis(args, output_dir):
    """Run lambda_var, lambda_cover, and tau sensitivity analysis."""
    sensitivity_results = {}

    # lambda_var sensitivity
    for lam in config.ABLATION_LAMBDA_VAR_VALUES:
        name = f"lambda_var_{lam}"
        cfg = {**config.ABLATION_CONFIGS["full_model"],
               "lambda_var": lam, "description": f"lambda_var={lam}"}
        sensitivity_results[name] = train_ablation(name, cfg, args, output_dir)

    # lambda_cover sensitivity
    for lam in config.ABLATION_LAMBDA_COVER_VALUES:
        name = f"lambda_cover_{lam}"
        cfg = {**config.ABLATION_CONFIGS["full_model"],
               "lambda_cover": lam, "description": f"lambda_cover={lam}"}
        sensitivity_results[name] = train_ablation(name, cfg, args, output_dir)

    # tau sensitivity
    for tau in config.ABLATION_TAU_VALUES:
        name = f"tau_{tau}"
        cfg = {**config.ABLATION_CONFIGS["full_model"],
               "temperature": tau, "description": f"tau={tau}"}
        sensitivity_results[name] = train_ablation(name, cfg, args, output_dir)

    return sensitivity_results


def main():
    args = parse_args()
    set_seed(config.SEED)

    output_dir = Path(args.output_dir) if args.output_dir else config.ABLATION_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.config == "all":
        for name, cfg in config.ABLATION_CONFIGS.items():
            r = train_ablation(name, cfg, args, output_dir)
            all_results[name] = r
    elif args.config == "sensitivity":
        all_results = run_sensitivity_analysis(args, output_dir)
    else:
        cfg = config.ABLATION_CONFIGS[args.config]
        r = train_ablation(args.config, cfg, args, output_dir)
        all_results[args.config] = r

    # Save results
    output_path = output_dir / "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("Table 4: Ablation Study Results")
    print("=" * 70)
    print(f"{'Configuration':<45} {'R@1':>6} {'R@5':>6} {'R@10':>6}")
    print("-" * 70)
    for name, r in all_results.items():
        ret = r.get("retrieval", {})
        r1 = ret.get("R@1", 0)
        r5 = ret.get("R@5", 0)
        r10 = ret.get("R@10", 0)
        print(f"{r.get('description', name):<45} {r1:>6.4f} {r5:>6.4f} {r10:>6.4f}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Ablation study failed")
        raise
