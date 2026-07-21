"""
GaussianImageDistribution - MSDA Distribution Alignment Training Script

Trains the MSDA (Multi-caption Semantic Distribution Alignment) model, which
models image and text embeddings as Gaussians. The image uses a general
covariance Sigma_v = diag(sigma_v^2) + U_v U_v^T; text is diagonal-only (v1).
The image variance is supervised toward the multi-caption semantic spread, and
the image low-rank directions toward the caption deviation directions.

Total loss = lambda_ctr*L_set + lambda_mu*L_mu + lambda_var*L_var
           + lambda_cover*L_cover + lambda_cov*L_cov + lambda_reg*L_reg

A staged schedule activates loss components progressively:
    Warm-up: L_set + L_mu (+ L_reg always on)
    Main:    + L_var + L_cover
    Full:    linearly ramp L_cov 0 -> 1

Checkpoint selection is by the MSDA uncertainty-discounted cosine Recall@1
(the same score L_set optimizes), so the trained objective, the selection
metric and the reported metric all agree.

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
from utils.lr_scheduler import apply_lr_for_epoch
from utils.seed import set_seed
from utils.retrieval import (
    compute_recall_bidirectional,
    compute_recall_msda_chunked,
)


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
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=["coco", "flickr"],
                        help="Training dataset tag, embedded in the checkpoint filename as "
                             "{model}_{dataset}_best|last.pt (coco=MSCOCO, flickr=flickr30k)")

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
    parser.add_argument("--lr-scheduler", type=str, default=config.LR_SCHEDULER,
                        choices=["none", "cosine"],
                        help="LR schedule: 'cosine' (cosine + linear warmup) or 'none' "
                             "(constant LR). Scales both clip-lr and mlp-lr param groups.")
    parser.add_argument("--warmup-epochs", type=int, default=config.LR_WARMUP_EPOCHS,
                        help="Linear warmup epochs for the cosine schedule (0 disables warmup)")
    parser.add_argument("--min-lr-ratio", type=float, default=config.LR_MIN_LR_RATIO,
                        help="Cosine floor as a fraction of the base LR")

    # MSDA loss arguments (six terms)
    parser.add_argument("--lambda-ctr", type=float, default=config.MSDA_LAMBDA_CTR)
    parser.add_argument("--lambda-mu", type=float, default=config.MSDA_LAMBDA_MU)
    parser.add_argument("--lambda-var", type=float, default=config.MSDA_LAMBDA_VAR)
    parser.add_argument("--lambda-cover", type=float, default=config.MSDA_LAMBDA_COVER)
    parser.add_argument("--lambda-cov", type=float, default=config.MSDA_LAMBDA_COV)
    parser.add_argument("--lambda-reg", type=float, default=config.MSDA_LAMBDA_REG)
    parser.add_argument("--tau", type=float, default=config.MSDA_TAU,
                        help="Fixed temperature in the L_set similarity (not learnable)")
    parser.add_argument("--m-pos", type=float, default=config.MSDA_M_POS,
                        help="L_cover positive coverage margin (per-D normalized Mahalanobis)")
    parser.add_argument("--target-var", type=float, default=config.MSDA_TARGET_VAR,
                        help="L_reg variance prior sigma_0^2")
    parser.add_argument("--m-neg", type=float, default=config.MSDA_M_NEG,
                        help="L_cover negative repulsion margin")
    parser.add_argument("--use-uncertainty-sim", action="store_true",
                        default=config.MSDA_USE_UNCERTAINTY_SIM,
                        help="L_set uses the uncertainty-discounted score (default)")
    parser.add_argument("--no-uncertainty-sim", dest="use_uncertainty_sim",
                        action="store_false",
                        help="L_set uses plain cosine (ablation)")

    # MSDA model arguments
    parser.add_argument("--cov-rank", type=int, default=config.MSDA_COV_RANK,
                        help="Low-rank covariance rank r for the image side (0 = diagonal only)")
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
                             "(MSDA R@1, higher better) or 'loss' "
                             "(val loss, lower better). Default: recall")

    # Output arguments
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint directory (uses config default if None)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (uses config default if None)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from "
                             "(e.g. checkpoints/dist_align_coco_last.pt). "
                             "Restores model weights, optimizer state, epoch, and best_recall.")

    return parser.parse_args()


def stage_multipliers(epoch: int, total: int, no_staged: bool) -> Dict[str, float]:
    """Per-loss multipliers for the staged MSDA schedule.

    Warm-up: L_set + L_mu (+ L_reg always on). Main: + L_var + L_cover.
    Full: linearly ramp L_cov 0 -> 1 (the cov head is untrained before this;
    a hard step would dominate and destabilize the retrieval means).
    L_reg is always on (pure stabilizer).
    """
    base = {"ctr": 1.0, "mu": 1.0, "var": 1.0, "cover": 1.0, "cov": 1.0, "reg": 1.0}
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
        full_len = max(1, total - main_end)
        j = epoch - main_end
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

    if args.freeze_clip:
        # Only train distribution + image covariance heads
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

    totals = {k: 0.0 for k in
              ("loss", "set_nce", "mu", "var", "cover", "cov", "reg", "img_var_avg")}
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
            outputs['text_mus'], outputs['text_logvars'], outputs.get('text_Us'),
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Clip global grad norm to protect against L_cov / cover spikes that can
        # destabilize the retrieval means.
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MSDA_GRAD_CLIP_NORM)
        optimizer.step()

        # Accumulate losses
        for k in totals:
            kk = "loss" if k == "loss" else k
            src = "total" if k == "loss" else k
            totals[k] += loss_dict[src]

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'NCE': f"{loss_dict['set_nce']:.4f}",
            'mu': f"{loss_dict['mu']:.3f}",
            'var': f"{loss_dict['var']:.3f}",
            'cov': f"{loss_dict['cov']:.3f}",
            'σ²i': f"{loss_dict['img_var_avg']:.3f}",
        })

        # Log every N batches
        if (batch_idx + 1) % 10 == 0:
            logger.debug(
                f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss_dict['total']:.4f}, "
                f"NCE: {loss_dict['set_nce']:.4f}, mu: {loss_dict['mu']:.4f}, "
                f"var: {loss_dict['var']:.4f}, cover: {loss_dict['cover']:.4f}, "
                f"cov: {loss_dict['cov']:.4f}, reg: {loss_dict['reg']:.4f}"
            )

    num_batches = max(processed_batches, 1)
    return {k: v / num_batches for k, v in totals.items()}


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

    totals = {k: 0.0 for k in
              ("loss", "set_nce", "mu", "var", "cover", "cov", "reg", "img_var_avg")}
    processed_batches = 0
    feats = {k: [] for k in ("img_mu", "text_mu", "img_logvar", "text_logvar")} \
        if compute_recall else None

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        if batch is None:
            continue
        processed_batches += 1

        pil_images = batch["image"]
        caption_lists = batch["captions"]

        pixel_values = model.process_images(pil_images).to(device)
        batch_size = len(pil_images)
        num_captions = len(caption_lists[0])
        all_captions = []
        for caption_list in caption_lists:
            all_captions.extend(caption_list)
        text_inputs = model.process_text(all_captions)
        input_ids = text_inputs["input_ids"].view(batch_size, num_captions, -1).to(device)
        attention_mask = text_inputs["attention_mask"].view(batch_size, num_captions, -1).to(device)

        outputs = model(pixel_values, input_ids, attention_mask)

        loss, loss_dict = criterion(
            outputs['img_mu'], outputs['img_logvar'], outputs['img_U'],
            outputs['text_mu'], outputs['text_logvar'],
            outputs['text_mus'], outputs['text_logvars'], outputs.get('text_Us'),
        )

        for k in totals:
            totals[k] += loss_dict["total" if k == "loss" else k]

        if feats is not None:
            feats["img_mu"].append(outputs['img_mu'].cpu())
            feats["text_mu"].append(outputs['text_mu'].cpu())
            feats["img_logvar"].append(outputs['img_logvar'].cpu())
            feats["text_logvar"].append(outputs['text_logvar'].cpu())

        pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

    num_batches = max(processed_batches, 1)
    metrics = {k: v / num_batches for k, v in totals.items()}

    # Retrieval Recall@K (image<->text, diagonal pairing). The val loader uses
    # shuffle=False, so concatenated img_mu[i] stays aligned with its own
    # caption-set text_mu[i] -> the diagonal is the positive pair.
    # Primary score: MSDA uncertainty-discounted cosine (= what L_set optimizes).
    # Secondary: plain cosine-on-means (methodology's mean-only retrieval mode).
    if compute_recall and recall_k_values and feats and feats["img_mu"]:
        img_mu = torch.cat(feats["img_mu"], dim=0).to(device)
        text_mu = torch.cat(feats["text_mu"], dim=0).to(device)
        img_lv = torch.cat(feats["img_logvar"], dim=0).to(device)
        text_lv = torch.cat(feats["text_logvar"], dim=0).to(device)
        msda = compute_recall_msda_chunked(
            img_mu, img_lv, text_mu, text_lv, recall_k_values, tau=criterion.tau)
        metrics.update(msda)
        cos = compute_recall_bidirectional(img_mu, text_mu, recall_k_values, normalize=True)
        for k in recall_k_values:
            metrics[f"cos_recall@{k}"] = (cos[f"recall_i2t@{k}"] + cos[f"recall_t2i@{k}"]) / 2

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
    logger.info(f"Cov rank r (image side): {args.cov_rank}")
    logger.info(f"Tau (fixed): {args.tau}")
    logger.info(f"Loss weights: ctr={args.lambda_ctr} mu={args.lambda_mu} var={args.lambda_var} "
                f"cover={args.lambda_cover} cov={args.lambda_cov} reg={args.lambda_reg}")
    logger.info(f"Cover m_pos={args.m_pos} m_neg={args.m_neg}; reg target_var={args.target_var}; "
                f"uncertainty_sim={args.use_uncertainty_sim}")
    logger.info(f"Staged schedule: {not args.no_staged}")
    logger.info(f"LR scheduler: {args.lr_scheduler} (warmup {args.warmup_epochs}, "
                f"min_lr_ratio {args.min_lr_ratio})")
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

    # Create MSDA loss (no learnable parameters; tau is a fixed scalar)
    criterion = MSDALoss(
        lambda_ctr=args.lambda_ctr,
        lambda_mu=args.lambda_mu,
        lambda_var=args.lambda_var,
        lambda_cover=args.lambda_cover,
        lambda_cov=args.lambda_cov,
        lambda_reg=args.lambda_reg,
        tau=args.tau,
        m_pos=args.m_pos,
        target_var=args.target_var,
        m_neg=args.m_neg,
        use_uncertainty_sim=args.use_uncertainty_sim,
    )
    logger.info("Using MSDA loss (uncertainty-discounted cosine L_set)")
    criterion = criterion.to(args.device)

    # Create optimizer
    optimizer = create_optimizer(model, args)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

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
        base_lrs = checkpoint.get("base_lrs", base_lrs)
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

        # Apply staged schedule: scale per-loss weights by stage multipliers
        mult = stage_multipliers(epoch, args.epochs, args.no_staged)
        criterion.lambda_ctr = args.lambda_ctr * mult["ctr"]
        criterion.lambda_mu = args.lambda_mu * mult["mu"]
        criterion.lambda_var = args.lambda_var * mult["var"]
        criterion.lambda_cover = args.lambda_cover * mult["cover"]
        criterion.lambda_cov = args.lambda_cov * mult["cov"]
        criterion.lambda_reg = args.lambda_reg * mult["reg"]
        logger.info(f"Epoch {epoch + 1} stage multipliers: {mult}")

        # Apply LR schedule for this epoch (no-op when scheduler == "none")
        apply_lr_for_epoch(optimizer, base_lrs, epoch, args.epochs,
                           args.warmup_epochs, args.min_lr_ratio,
                           args.lr_scheduler, logger)

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
            f"mu: {train_metrics['mu']:.4f}, "
            f"Var: {train_metrics['var']:.4f}, "
            f"Cover: {train_metrics['cover']:.4f}, "
            f"Cov: {train_metrics['cov']:.4f}, "
            f"Reg: {train_metrics['reg']:.4f}, "
            f"σ²img: {train_metrics['img_var_avg']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val NCE: {val_metrics['set_nce']:.4f}, "
            f"σ²img: {val_metrics['img_var_avg']:.4f}, "
            f"MSDA R@1/5/10: {val_metrics.get('msda_recall@1', 0):.3f}/"
            f"{val_metrics.get('msda_recall@5', 0):.3f}/{val_metrics.get('msda_recall@10', 0):.3f}, "
            f"Cos R@1/5/10: {val_metrics.get('cos_recall@1', 0):.3f}/"
            f"{val_metrics.get('cos_recall@5', 0):.3f}/{val_metrics.get('cos_recall@10', 0):.3f}"
        )

        # Save best checkpoint by the selected metric (MSDA R@1 higher-better,
        # or val loss lower-better).
        if args.select_by == "recall" and "msda_recall@1" in val_metrics:
            current_score = val_metrics["msda_recall@1"]
            improved = current_score > best_recall
            if improved:
                best_recall = current_score
        else:
            current_score = val_metrics["loss"]
            improved = current_score < best_val_loss
            if improved:
                best_val_loss = current_score

        if improved:
            best_checkpoint_path = checkpoint_dir / f"dist_align_{args.dataset}_best.pt"
            model.save(str(best_checkpoint_path))
            score_str = (f"msda_recall@1: {best_recall:.4f}" if args.select_by == "recall"
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
    final_checkpoint_path = checkpoint_dir / f"dist_align_{args.dataset}_last.pt"
    final_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": last_epoch + 1,
        "best_val_loss": best_val_loss,
        "best_recall": best_recall,
        "patience_counter": patience_counter,
        "base_lrs": base_lrs,
        "select_by": args.select_by,
    }
    torch.save(final_state, str(final_checkpoint_path))
    logger.info(f"Final model saved to {final_checkpoint_path}")
    logger.info(f"Best val loss: {best_val_loss:.4f} | Best MSDA recall@1: {best_recall:.4f} "
                f"(selected by: {args.select_by})")
    logger.info("Training completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Training failed")
        raise
