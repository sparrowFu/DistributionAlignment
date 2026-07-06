"""
GaussianImageDistribution - MSDA Distribution Alignment Training Script

Trains the MSDA (Multi-caption Semantic Distribution Alignment) model, which
models image and text embeddings as general Gaussians N(mu, Sigma) with a
learned (non-diagonal) covariance, supervised so that the image variance
matches the multi-caption semantic spread.

A 3-stage schedule activates loss components progressively:
    Warm-up: L_set-NCE + L_mu
    Main:    + L_var + L_cover
    Full:    + L_cov

Usage:
    python scripts/train_dist_align.py
    python main.py --task train_dist_align
"""

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys
# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.caption_dataset import ImageCaptionDataset, filter_none_collate
from models.dist_align_model import DistributionAlignmentModel
from losses.dist_align_losses import MSDALoss
from utils.logger import get_logger, log_exception
from utils.seed import set_seed
from utils.retrieval import compute_recall_bidirectional


# Setup logger
logger = get_logger("train_dist_align", config.TRAIN_DIST_ALIGN_LOG_PATH)

# Exclude faulty CPU cores (e.g. unstable CPU 2) before DataLoader workers and
# torch threads are created. Inherited by forked worker processes.
from utils.cpu_affinity import apply_cpu_affinity
apply_cpu_affinity()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MSDA Distribution Alignment Model")

    # Data arguments
    parser.add_argument("--captions-path", type=str, default=None,
                        help="Path to captions parquet file (uses config default if None)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Path to images directory (uses config default if None)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=config.DIST_ALIGN_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.DIST_ALIGN_BATCH_SIZE,
                        help="Training batch size")
    parser.add_argument("--clip-lr", type=float, default=config.DIST_ALIGN_CLIP_LR,
                        help="Learning rate for CLIP (if fine-tuning)")
    parser.add_argument("--mlp-lr", type=float, default=config.DIST_ALIGN_MLP_LR,
                        help="Learning rate for MLP / covariance heads")
    parser.add_argument("--weight-decay", type=float, default=config.DIST_ALIGN_WEIGHT_DECAY,
                        help="Weight decay")
    parser.add_argument("--temperature", type=float, default=config.MSDA_TAU,
                        help="Temperature tau for L_set-NCE similarity")

    # MSDA loss arguments
    parser.add_argument("--lambda-ctr", type=float, default=config.MSDA_LAMBDA_CTR,
                        help="Weight for set-level contrastive loss")
    parser.add_argument("--lambda-mu", type=float, default=config.MSDA_LAMBDA_MU,
                        help="Weight for mean-center alignment loss")
    parser.add_argument("--lambda-var", type=float, default=config.MSDA_LAMBDA_VAR,
                        help="Weight for variance semantic consistency (core)")
    parser.add_argument("--lambda-cover", type=float, default=config.MSDA_LAMBDA_COVER,
                        help="Weight for multi-caption coverage loss")
    parser.add_argument("--lambda-cov", type=float, default=config.MSDA_LAMBDA_COV,
                        help="Weight for covariance direction alignment")
    parser.add_argument("--lambda-reg", type=float, default=config.MSDA_LAMBDA_REG,
                        help="Weight for variance regularization")
    parser.add_argument("--m-pos", type=float, default=config.MSDA_M_POS,
                        help="Per-dim-normalized positive coverage radius")
    parser.add_argument("--target-var", type=float, default=config.MSDA_TARGET_VAR,
                        help="Target variance sigma_0^2 for L_reg")
    parser.add_argument("--use-neg-cover", action="store_true", default=config.MSDA_USE_NEG_COVER,
                        help="Add negative coverage repulsion term")
    parser.add_argument("--no-uncertainty-sim", action="store_true", default=False,
                        help="Use standard cosine instead of uncertainty-discounted similarity")

    # MSDA model arguments
    parser.add_argument("--cov-rank", type=int, default=config.MSDA_COV_RANK,
                        help="Low-rank covariance rank r (0 = diagonal only)")
    parser.add_argument("--freeze-clip", action="store_true", default=config.DIST_ALIGN_FREEZE_CLIP,
                        help="Freeze CLIP parameters")
    parser.add_argument("--no-freeze-clip", action="store_false", dest="freeze_clip",
                        help="Don't freeze CLIP parameters")
    parser.add_argument("--distribution-merging", type=str, default=config.DIST_ALIGN_DISTRIBUTION_MERGING,
                        choices=["moment_matching", "poe", "simple"],
                        help="Method for merging multiple text distributions")
    parser.add_argument("--dropout-rate", type=float, default=config.DIST_ALIGN_DROPOUT_RATE,
                        help="Dropout rate for MLP heads")
    parser.add_argument("--no-staged", action="store_true",
                        help="Disable 3-stage schedule; use all losses from epoch 1")

    # System arguments
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    # Validation and early stopping arguments
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation set ratio (default: 0.1)")
    parser.add_argument("--early-stop-patience", type=int, default=3,
                        help="Early stopping patience in epochs (default: 3)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping")
    parser.add_argument("--select-by", type=str, default="recall",
                        choices=["recall", "loss"],
                        help="Best-checkpoint selection metric: 'recall' "
                             "(bidirectional R@1, higher better) or 'loss' "
                             "(val loss, lower better). Default: recall")

    # Output arguments
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint directory (uses config default if None)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (uses config default if None)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from "
                             "(e.g. checkpoints/dist_align_last.pt). "
                             "Restores model weights, optimizer state, epoch, and best_val_loss.")

    return parser.parse_args()


def stage_multipliers(epoch: int, total: int, no_staged: bool) -> Dict[str, float]:
    """Return per-loss multipliers for the 3-stage MSDA schedule.

    Warm-up: L_set-NCE + L_mu.  Main: + L_var + L_cover.  Full: + L_cov.

    In the full stage, L_cov is *linearly ramped* from 0 to 1 across the
    stage's epochs instead of a hard 0->1 step. The cov head (img_U) is
    essentially untrained before this stage, so L_cov starts near its maximum
    2*r; a hard step injects a gradient that dominates every other term and
    crashes Recall@1 the moment L_cov activates. The ramp lets the head warm
    up gently (together with the reduced MSDA_LAMBDA_COV and grad clipping).
    """
    base = {"ctr": 1.0, "mu": 1.0, "var": 1.0, "cover": 1.0, "cov": 1.0}
    if no_staged or total <= 0:
        return base
    warmup_end = max(1, int(round(total * config.MSDA_STAGE_WARMUP_FRAC)))
    main_end = max(warmup_end + 1,
                   int(round(total * (config.MSDA_STAGE_WARMUP_FRAC + config.MSDA_STAGE_MAIN_FRAC))))
    if epoch < warmup_end:
        base.update(var=0.0, cover=0.0, cov=0.0)
    elif epoch < main_end:
        base.update(cov=0.0)
    else:
        # Full stage: linearly ramp L_cov 0 -> 1 across the remaining epochs.
        full_len = max(1, total - main_end)
        j = epoch - main_end  # 0-based index within the full stage
        base["cov"] = min(1.0, (j + 1) / full_len)
    return base


def create_optimizer(model, args):
    """Create optimizer with different learning rates for CLIP and MLP/cov heads."""
    head_params = (
        list(model.img_mu_head.parameters())
        + list(model.img_logvar_head.parameters())
        + list(model.text_mu_head.parameters())
        + list(model.text_logvar_head.parameters())
    )
    if getattr(model, "cov_rank", 0) > 0:
        head_params += list(model.img_cov_head.parameters())
        head_params += list(model.text_cov_head.parameters())

    if args.freeze_clip:
        # Only train distribution + covariance heads
        optimizer = optim.Adam(head_params, lr=args.mlp_lr, weight_decay=args.weight_decay)
    else:
        # CLIP and heads with different learning rates
        optimizer = optim.Adam([
            {"params": model.clip_model.parameters(), "lr": args.clip_lr},
            {"params": head_params, "lr": args.mlp_lr},
        ], weight_decay=args.weight_decay)

    return optimizer


def train_epoch(
    model: DistributionAlignmentModel,
    dataloader: DataLoader,
    criterion: MSDALoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    if not model.freeze_clip:
        model.clip_model.train()

    total_loss = 0.0
    total_nce = 0.0
    total_var = 0.0
    total_cover = 0.0
    total_cov = 0.0
    total_img_var = 0.0
    processed_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue
        processed_batches += 1

        # Get data - PIL images and text lists
        pil_images = batch["image"]            # List[PIL.Image]
        caption_lists = batch["captions"]      # List[List[str]]

        # Process images with CLIP processor
        pixel_values = model.process_images(pil_images).to(device)  # [B, 3, 224, 224]

        # Process text captions: flatten B*K captions, then reshape to [B, K, max_len]
        batch_size = len(pil_images)
        num_captions = len(caption_lists[0])
        all_captions = []
        for caption_list in caption_lists:
            all_captions.extend(caption_list)
        text_inputs = model.process_text(all_captions)
        input_ids = text_inputs["input_ids"].view(batch_size, num_captions, -1).to(device)
        attention_mask = text_inputs["attention_mask"].view(batch_size, num_captions, -1).to(device)

        # Forward pass
        outputs = model(pixel_values, input_ids, attention_mask)

        # Compute MSDA loss
        loss, loss_dict = criterion(
            outputs['img_mu'], outputs['img_logvar'], outputs['img_U'],
            outputs['text_mu'], outputs['text_logvar'],
            outputs['text_mus'], outputs['text_logvars'], outputs['text_Us'],
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Clip global grad norm to protect against L_cov / cover spikes that can
        # destabilize the retrieval means (P0 stability fix).
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MSDA_GRAD_CLIP_NORM)
        optimizer.step()

        # Accumulate losses
        total_loss += loss_dict['total']
        total_nce += loss_dict['set_nce']
        total_var += loss_dict['var']
        total_cover += loss_dict['cover']
        total_cov += loss_dict['cov']
        total_img_var += loss_dict['img_var_avg']

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'NCE': f"{loss_dict['set_nce']:.4f}",
            'var': f"{loss_dict['var']:.4f}",
            'cov': f"{loss_dict['cov']:.4f}",
            'σ²i': f"{loss_dict['img_var_avg']:.3f}",
        })

        # Log every N batches
        if (batch_idx + 1) % 10 == 0:
            logger.debug(
                f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss_dict['total']:.4f}, "
                f"NCE: {loss_dict['set_nce']:.4f}, "
                f"Var: {loss_dict['var']:.4f}, "
                f"Cover: {loss_dict['cover']:.4f}, "
                f"Cov: {loss_dict['cov']:.4f}"
            )

    # Compute averages
    num_batches = max(processed_batches, 1)
    metrics = {
        'loss': total_loss / num_batches,
        'set_nce': total_nce / num_batches,
        'var': total_var / num_batches,
        'cover': total_cover / num_batches,
        'cov': total_cov / num_batches,
        'img_var_avg': total_img_var / num_batches,
    }

    return metrics


@torch.no_grad()
def evaluate(
    model: DistributionAlignmentModel,
    dataloader: DataLoader,
    criterion: MSDALoss,
    device: torch.device,
    compute_recall: bool = False,
    recall_k_values=None,
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()

    total_loss = 0.0
    total_nce = 0.0
    total_var = 0.0
    total_cover = 0.0
    total_cov = 0.0
    total_img_var = 0.0
    processed_batches = 0
    all_img_mu = [] if compute_recall else None
    all_text_mu = [] if compute_recall else None

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        if batch is None:
            continue
        processed_batches += 1

        # Get data - PIL images and text lists
        pil_images = batch["image"]
        caption_lists = batch["captions"]

        # Process images with CLIP processor
        pixel_values = model.process_images(pil_images).to(device)

        # Process text captions
        batch_size = len(pil_images)
        num_captions = len(caption_lists[0])
        all_captions = []
        for caption_list in caption_lists:
            all_captions.extend(caption_list)
        text_inputs = model.process_text(all_captions)
        input_ids = text_inputs["input_ids"].view(batch_size, num_captions, -1).to(device)
        attention_mask = text_inputs["attention_mask"].view(batch_size, num_captions, -1).to(device)

        # Forward pass
        outputs = model(pixel_values, input_ids, attention_mask)

        # Compute MSDA loss
        loss, loss_dict = criterion(
            outputs['img_mu'], outputs['img_logvar'], outputs['img_U'],
            outputs['text_mu'], outputs['text_logvar'],
            outputs['text_mus'], outputs['text_logvars'], outputs['text_Us'],
        )

        total_loss += loss_dict['total']
        total_nce += loss_dict['set_nce']
        total_var += loss_dict['var']
        total_cover += loss_dict['cover']
        total_cov += loss_dict['cov']
        total_img_var += loss_dict['img_var_avg']

        if all_img_mu is not None:
            all_img_mu.append(outputs['img_mu'].cpu())
            all_text_mu.append(outputs['text_mu'].cpu())

        pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

    # Compute averages
    num_batches = max(processed_batches, 1)
    metrics = {
        'loss': total_loss / num_batches,
        'set_nce': total_nce / num_batches,
        'var': total_var / num_batches,
        'cover': total_cover / num_batches,
        'cov': total_cov / num_batches,
        'img_var_avg': total_img_var / num_batches,
    }

    # Retrieval Recall@K (image<->text, diagonal pairing). The val loader uses
    # shuffle=False, so concatenated img_mu[i] stays aligned with its own caption-set
    # text_mu[i] -> the diagonal is the positive pair. Used for best selection.
    if compute_recall and recall_k_values and all_img_mu:
        img_mu_all = torch.cat(all_img_mu, dim=0).to(device)
        text_mu_all = torch.cat(all_text_mu, dim=0).to(device)
        recall = compute_recall_bidirectional(
            img_mu_all, text_mu_all, recall_k_values, chunk_size=1000, normalize=True
        )
        metrics.update(recall)

    return metrics


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Log configuration
    logger.info("=" * 60)
    logger.info("MSDA Distribution Alignment Training")
    logger.info("=" * 60)
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"CLIP LR: {args.clip_lr}")
    logger.info(f"MLP LR: {args.mlp_lr}")
    logger.info(f"Freeze CLIP: {args.freeze_clip}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Cov rank r: {args.cov_rank}")
    logger.info(f"Tau: {args.temperature}")
    logger.info(f"Lambda (ctr/mu/var/cover/cov/reg): "
                f"{args.lambda_ctr}/{args.lambda_mu}/{args.lambda_var}/"
                f"{args.lambda_cover}/{args.lambda_cov}/{args.lambda_reg}")
    logger.info(f"Staged schedule: {not args.no_staged}")
    logger.info("=" * 60)

    # Create model
    logger.info("Creating model...")
    model = DistributionAlignmentModel(
        freeze_clip=args.freeze_clip,
        distribution_merging=args.distribution_merging,
        dropout_rate=args.dropout_rate,
        cov_rank=args.cov_rank,
    )
    model = model.to(args.device)
    logger.info(f"Model created with {model.num_trainable_parameters():,} trainable parameters")

    # Create MSDA loss
    criterion = MSDALoss(
        lambda_ctr=args.lambda_ctr,
        lambda_mu=args.lambda_mu,
        lambda_var=args.lambda_var,
        lambda_cover=args.lambda_cover,
        lambda_cov=args.lambda_cov,
        lambda_reg=args.lambda_reg,
        tau=args.temperature,
        m_pos=args.m_pos,
        target_var=args.target_var,
        use_neg_cover=args.use_neg_cover,
        m_neg=config.MSDA_M_NEG,
        use_uncertainty_sim=not args.no_uncertainty_sim,
    )
    logger.info("Using MSDA loss")

    # Create optimizer
    optimizer = create_optimizer(model, args)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    best_recall = -float('inf')  # for --select-by recall
    patience_counter = 0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error(f"Resume checkpoint not found: {resume_path}")
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        logger.info(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(str(resume_path), map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Move optimizer state to correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(args.device)
        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        best_recall = checkpoint.get("best_recall", -float('inf'))
        patience_counter = checkpoint.get("patience_counter", 0)
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}, "
                     f"best_recall: {best_recall:.4f}, patience_counter: {patience_counter}")

    # Load dataset
    captions_path = args.captions_path or config.CAPTIONS_PATH
    images_dir = args.images_dir or config.IMAGES_DIR

    logger.info(f"Loading dataset from {captions_path}")
    logger.info(f"Images directory: {images_dir}")

    full_dataset = ImageCaptionDataset(
        captions_path=captions_path,
        images_dir=images_dir,
        num_captions=config.NUM_CAPTIONS
    )

    # Split into train and validation sets
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=filter_none_collate
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=filter_none_collate
    )

    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")
    logger.info(f"Batch size: {args.batch_size}, Train batches per epoch: {len(train_dataloader)}")

    # Training loop with early stopping
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else config.CHECKPOINT_DIR

    logger.info(f"Starting training from epoch {start_epoch + 1}...")
    last_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch

        # Apply 3-stage schedule: scale per-loss weights by stage multipliers
        mult = stage_multipliers(epoch, args.epochs, args.no_staged)
        criterion.lambda_ctr = args.lambda_ctr * mult["ctr"]
        criterion.lambda_mu = args.lambda_mu * mult["mu"]
        criterion.lambda_var = args.lambda_var * mult["var"]
        criterion.lambda_cover = args.lambda_cover * mult["cover"]
        criterion.lambda_cov = args.lambda_cov * mult["cov"]
        logger.info(f"Epoch {epoch + 1} stage multipliers: {mult}")

        # Train
        train_metrics = train_epoch(
            model, train_dataloader, criterion, optimizer, args.device, epoch
        )

        # Validate
        val_metrics = evaluate(
            model, val_dataloader, criterion, args.device,
            compute_recall=(args.select_by == "recall"),
            recall_k_values=config.RECALL_AT_K,
        )

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"NCE: {train_metrics['set_nce']:.4f}, "
            f"Var: {train_metrics['var']:.4f}, "
            f"Cover: {train_metrics['cover']:.4f}, "
            f"Cov: {train_metrics['cov']:.4f}, "
            f"σ²img: {train_metrics['img_var_avg']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val NCE: {val_metrics['set_nce']:.4f}, "
            f"σ²img: {val_metrics['img_var_avg']:.4f}, "
            f"R@1/5/10: {val_metrics.get('recall@1', 0):.3f}/"
            f"{val_metrics.get('recall@5', 0):.3f}/{val_metrics.get('recall@10', 0):.3f}"
        )

        # Save best checkpoint by the selected metric (recall@1 higher-better,
        # or val loss lower-better).
        if args.select_by == "recall" and "recall@1" in val_metrics:
            current_score = val_metrics["recall@1"]
            improved = current_score > best_recall
            if improved:
                best_recall = current_score
        else:
            current_score = val_metrics["loss"]
            improved = current_score < best_val_loss
            if improved:
                best_val_loss = current_score

        if improved:
            best_checkpoint_path = checkpoint_dir / "dist_align_best.pt"
            model.save(str(best_checkpoint_path))
            score_str = (f"recall@1: {best_recall:.4f}" if args.select_by == "recall"
                         else f"val_loss: {best_val_loss:.4f}")
            logger.info(f"Best model saved ({score_str}) -> {best_checkpoint_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{args.early_stop_patience}")

        # Early stopping
        if not args.no_early_stop and patience_counter >= args.early_stop_patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Save final model with full training state for resumption
    final_checkpoint_path = checkpoint_dir / "dist_align_last.pt"
    final_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": last_epoch + 1,
        "best_val_loss": best_val_loss,
        "best_recall": best_recall,
        "patience_counter": patience_counter,
        "select_by": args.select_by,
    }
    torch.save(final_state, str(final_checkpoint_path))
    logger.info(f"Final model saved to {final_checkpoint_path}")
    logger.info(f"Best val loss: {best_val_loss:.4f} | Best recall@1: {best_recall:.4f} "
                f"(selected by: {args.select_by})")
    logger.info("Training completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Training failed")
        raise
