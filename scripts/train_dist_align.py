"""
GaussianImageDistribution - Distribution Alignment Training Script

This script trains the distribution alignment model on image-caption pairs.
It models image and text embeddings as Gaussian distributions to address
modality gap and one-to-many relationships.

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
from losses.dist_align_losses import DistributionAlignmentLoss, CombinedDistributionLoss, DistributionalContrastiveLoss, UncertaintyCalibratedContrastiveLoss
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


# Setup logger
logger = get_logger("train_dist_align", config.TRAIN_DIST_ALIGN_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Distribution Alignment Model")

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
                        help="Learning rate for MLP distribution heads")
    parser.add_argument("--weight-decay", type=float, default=config.DIST_ALIGN_WEIGHT_DECAY,
                        help="Weight decay")
    parser.add_argument("--temperature", type=float, default=config.DIST_ALIGN_TEMPERATURE,
                        help="Temperature for contrastive loss")

    # Loss arguments
    parser.add_argument("--lambda-contrastive", type=float, default=config.DIST_ALIGN_LAMBDA_CONTRASTIVE,
                        help="Weight for contrastive loss")
    parser.add_argument("--lambda-kl", type=float, default=config.DIST_ALIGN_LAMBDA_KL,
                        help="Weight for KL divergence loss")
    parser.add_argument("--lambda-var", type=float, default=config.DIST_ALIGN_LAMBDA_VAR,
                        help="Weight for variance regularization loss")
    parser.add_argument("--kl-type", type=str, default=config.DIST_ALIGN_KL_TYPE,
                        choices=["symmetric", "forward", "reverse", "wasserstein"],
                        help="Type of KL divergence")

    # Model arguments
    parser.add_argument("--freeze-clip", action="store_true", default=config.DIST_ALIGN_FREEZE_CLIP,
                        help="Freeze CLIP parameters")
    parser.add_argument("--no-freeze-clip", action="store_false", dest="freeze_clip",
                        help="Don't freeze CLIP parameters")
    parser.add_argument("--distribution-merging", type=str, default=config.DIST_ALIGN_DISTRIBUTION_MERGING,
                        choices=["moment_matching", "poe", "simple"],
                        help="Method for merging multiple text distributions")
    parser.add_argument("--dropout-rate", type=float, default=config.DIST_ALIGN_DROPOUT_RATE,
                        help="Dropout rate for MLP heads")
    parser.add_argument("--use-variance-loss", action="store_true", default=config.DIST_ALIGN_USE_VARIANCE_LOSS,
                        help="Use variance regularization loss")

    # Distributional Contrastive Learning via OT arguments
    parser.add_argument("--use-ot-contrastive", action="store_true", default=config.DIST_ALIGN_USE_OT_CONTRASTIVE,
                        help="Use OT-based distributional contrastive loss")
    parser.add_argument("--no-ot-contrastive", action="store_false", dest="use_ot_contrastive",
                        help="Disable OT-based distributional contrastive loss")
    parser.add_argument("--ot-temperature", type=float, default=config.DIST_ALIGN_OT_TEMPERATURE,
                        help="Temperature for W2-based distributional similarity")
    parser.add_argument("--lambda-ot", type=float, default=config.DIST_ALIGN_LAMBDA_OT,
                        help="Weight for distributional contrastive loss")
    parser.add_argument("--lambda-var-ot", type=float, default=config.DIST_ALIGN_LAMBDA_VAR_OT,
                        help="Weight for variance regularization in OT mode")
    parser.add_argument("--min-sigma", type=float, default=config.DIST_ALIGN_MIN_SIGMA,
                        help="Minimum sigma to prevent numerical collapse")

    # Uncertainty-Calibrated Distributional Contrastive Learning arguments
    parser.add_argument("--use-uc-cl", action="store_true", default=config.DIST_ALIGN_USE_UC_CL,
                        help="Use Uncertainty-Calibrated Distributional Contrastive Learning")
    parser.add_argument("--no-uc-cl", action="store_false", dest="use_uc_cl",
                        help="Disable Uncertainty-Calibrated Distributional Contrastive Learning")
    parser.add_argument("--uc-temperature", type=float, default=config.DIST_ALIGN_UC_TEMPERATURE,
                        help="Temperature for uncertainty-calibrated similarity")
    parser.add_argument("--lambda-uc-cl", type=float, default=config.DIST_ALIGN_LAMBDA_UC_CL,
                        help="Weight for uncertainty-calibrated contrastive loss (λ_cl)")
    parser.add_argument("--lambda-consist", type=float, default=config.DIST_ALIGN_LAMBDA_CONSIST,
                        help="Weight for distributional consistency loss (λ_consist)")
    parser.add_argument("--lambda-uc-var", type=float, default=config.DIST_ALIGN_LAMBDA_UC_VAR,
                        help="Weight for variance regularization in UC-CL mode (λ_var)")

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


def create_optimizer(model, args):
    """Create optimizer with different learning rates for CLIP and MLP."""
    if args.freeze_clip:
        # Only train MLP heads
        optimizer = optim.Adam(
            model.trainable_parameters(),
            lr=args.mlp_lr,
            weight_decay=args.weight_decay
        )
    else:
        # CLIP and MLP with different learning rates
        optimizer = optim.Adam([
            {
                'params': model.clip_model.parameters(),
                'lr': args.clip_lr,
            },
            {
                'params': list(model.img_mu_head.parameters()) +
                          list(model.img_logvar_head.parameters()) +
                          list(model.text_mu_head.parameters()) +
                          list(model.text_logvar_head.parameters()),
                'lr': args.mlp_lr,
            }
        ], weight_decay=args.weight_decay)

    return optimizer


def train_epoch(
    model: DistributionAlignmentModel,
    dataloader: DataLoader,
    criterion,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_variance_loss: bool = False,
    use_ot: bool = False,
    use_uc_cl: bool = False
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    if not model.freeze_clip:
        model.clip_model.train()

    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_kl_loss = 0.0
    total_var_loss = 0.0
    total_consist_loss = 0.0
    total_w2_pos = 0.0
    total_w2_all = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue

        # Get data - PIL images and text lists
        pil_images = batch["image"]  # List[PIL.Image]
        caption_lists = batch["captions"]  # List[List[str]]

        # Process images with CLIP processor
        pixel_values = model.process_images(pil_images).to(device)  # [B, 3, 224, 224]

        # Process text captions
        # Each image has K captions, need to flatten and process
        batch_size = len(pil_images)
        num_captions = len(caption_lists[0])

        # Flatten all captions: [B*K, max_len]
        all_captions = []
        for caption_list in caption_lists:
            all_captions.extend(caption_list)  # Flatten the list

        # Process with CLIP processor
        text_inputs = model.process_text(all_captions)  # Returns dict with input_ids and attention_mask

        # Reshape to [B, K, max_len]
        input_ids = text_inputs["input_ids"].view(batch_size, num_captions, -1).to(device)
        attention_mask = text_inputs["attention_mask"].view(batch_size, num_captions, -1).to(device)

        # Forward pass
        outputs = model(pixel_values, input_ids, attention_mask)

        # Compute loss
        if use_uc_cl:
            loss, loss_dict = criterion(
                outputs['img_features'],
                outputs['text_features'],
                outputs['img_mu'],
                outputs['img_logvar'],
                outputs['text_mu'],
                outputs['text_logvar'],
                text_mus=outputs['text_mus'],
            )
        else:
            loss, loss_dict = criterion(
                outputs['img_features'],
                outputs['text_features'],
                outputs['img_mu'],
                outputs['img_logvar'],
                outputs['text_mu'],
                outputs['text_logvar']
            )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss_dict['total']
        total_contrastive_loss += loss_dict['contrastive']
        total_kl_loss += loss_dict['kl']

        if use_variance_loss or use_ot or use_uc_cl:
            total_var_loss += loss_dict.get('variance', 0.0)

        if use_uc_cl:
            total_consist_loss += loss_dict.get('consistency', 0.0)

        if use_ot:
            total_w2_pos += loss_dict.get('avg_w2_pos', 0.0)
            total_w2_all += loss_dict.get('avg_w2_all', 0.0)

        # Update progress bar
        if use_uc_cl:
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'CL': f"{loss_dict['contrastive']:.4f}",
                'Con': f"{loss_dict.get('consistency', 0):.4f}"
            })
        elif use_ot:
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'OT': f"{loss_dict['contrastive']:.4f}",
                'W2': f"{loss_dict.get('avg_w2_pos', 0):.2f}"
            })
        else:
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'contrastive': f"{loss_dict['contrastive']:.4f}",
                'kl': f"{loss_dict['kl']:.4f}"
            })

        # Log every N batches
        if (batch_idx + 1) % 10 == 0:
            if use_uc_cl:
                logger.debug(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, "
                    f"Loss: {loss_dict['total']:.4f}, "
                    f"CL: {loss_dict['contrastive']:.4f}, "
                    f"Consist: {loss_dict.get('consistency', 0):.4f}, "
                    f"Var: {loss_dict.get('variance', 0):.4f}"
                )
            elif use_ot:
                logger.debug(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, "
                    f"Loss: {loss_dict['total']:.4f}, "
                    f"OT: {loss_dict['contrastive']:.4f}, "
                    f"Var: {loss_dict['kl']:.4f}, "
                    f"W2_pos: {loss_dict.get('avg_w2_pos', 0):.2f}"
                )
            else:
                logger.debug(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, "
                    f"Loss: {loss_dict['total']:.4f}, "
                    f"Contrastive: {loss_dict['contrastive']:.4f}, "
                    f"KL: {loss_dict['kl']:.4f}"
                )

    # Compute averages
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'contrastive_loss': total_contrastive_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches,
    }

    if use_variance_loss:
        metrics['variance_loss'] = total_var_loss / num_batches

    if use_ot:
        metrics['variance_loss'] = total_var_loss / num_batches
        metrics['avg_w2_pos'] = total_w2_pos / num_batches
        metrics['avg_w2_all'] = total_w2_all / num_batches

    if use_uc_cl:
        metrics['variance_loss'] = total_var_loss / num_batches
        metrics['consistency_loss'] = total_consist_loss / num_batches

    return metrics


@torch.no_grad()
def evaluate(
    model: DistributionAlignmentModel,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
    use_variance_loss: bool = False,
    use_ot: bool = False,
    use_uc_cl: bool = False
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()

    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_kl_loss = 0.0
    total_var_loss = 0.0
    total_consist_loss = 0.0
    total_w2_pos = 0.0
    total_w2_all = 0.0

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        if batch is None:
            continue

        # Get data - PIL images and text lists
        pil_images = batch["image"]  # List[PIL.Image]
        caption_lists = batch["captions"]  # List[List[str]]

        # Process images with CLIP processor
        pixel_values = model.process_images(pil_images).to(device)  # [B, 3, 224, 224]

        # Process text captions
        batch_size = len(pil_images)
        num_captions = len(caption_lists[0])

        # Flatten all captions: [B*K]
        all_captions = []
        for caption_list in caption_lists:
            all_captions.extend(caption_list)

        # Process with CLIP processor
        text_inputs = model.process_text(all_captions)

        # Reshape to [B, K, max_len]
        input_ids = text_inputs["input_ids"].view(batch_size, num_captions, -1).to(device)
        attention_mask = text_inputs["attention_mask"].view(batch_size, num_captions, -1).to(device)

        # Forward pass
        outputs = model(pixel_values, input_ids, attention_mask)

        # Compute loss
        if use_uc_cl:
            loss, loss_dict = criterion(
                outputs['img_features'],
                outputs['text_features'],
                outputs['img_mu'],
                outputs['img_logvar'],
                outputs['text_mu'],
                outputs['text_logvar'],
                text_mus=outputs['text_mus'],
            )
        else:
            loss, loss_dict = criterion(
                outputs['img_features'],
                outputs['text_features'],
                outputs['img_mu'],
                outputs['img_logvar'],
                outputs['text_mu'],
                outputs['text_logvar']
            )

        total_loss += loss_dict['total']
        total_contrastive_loss += loss_dict['contrastive']
        total_kl_loss += loss_dict['kl']

        if use_variance_loss or use_ot or use_uc_cl:
            total_var_loss += loss_dict.get('variance', 0.0)

        if use_uc_cl:
            total_consist_loss += loss_dict.get('consistency', 0.0)

        if use_ot:
            total_w2_pos += loss_dict.get('avg_w2_pos', 0.0)
            total_w2_all += loss_dict.get('avg_w2_all', 0.0)

        pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

    # Compute averages
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'contrastive_loss': total_contrastive_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches,
    }

    if use_variance_loss:
        metrics['variance_loss'] = total_var_loss / num_batches

    if use_ot:
        metrics['variance_loss'] = total_var_loss / num_batches
        metrics['avg_w2_pos'] = total_w2_pos / num_batches
        metrics['avg_w2_all'] = total_w2_all / num_batches

    if use_uc_cl:
        metrics['variance_loss'] = total_var_loss / num_batches
        metrics['consistency_loss'] = total_consist_loss / num_batches

    return metrics


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Log configuration
    logger.info("=" * 60)
    if args.use_uc_cl:
        logger.info("Distribution Alignment Training (Uncertainty-Calibrated CL)")
    elif args.use_ot_contrastive:
        logger.info("Distribution Alignment Training (OT Contrastive)")
    else:
        logger.info("Distribution Alignment Training")
    logger.info("=" * 60)
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"CLIP LR: {args.clip_lr}")
    logger.info(f"MLP LR: {args.mlp_lr}")
    logger.info(f"Freeze CLIP: {args.freeze_clip}")
    logger.info(f"Device: {args.device}")
    if args.use_uc_cl:
        logger.info(f"UC Temperature: {args.uc_temperature}")
        logger.info(f"Lambda UC-CL: {args.lambda_uc_cl}")
        logger.info(f"Lambda Consist: {args.lambda_consist}")
        logger.info(f"Lambda UC Var: {args.lambda_uc_var}")
    elif args.use_ot_contrastive:
        logger.info(f"OT Temperature: {args.ot_temperature}")
        logger.info(f"Lambda OT: {args.lambda_ot}")
        logger.info(f"Lambda Var OT: {args.lambda_var_ot}")
        logger.info(f"Min Sigma: {args.min_sigma}")
    logger.info("=" * 60)

    # Create model
    logger.info("Creating model...")
    model = DistributionAlignmentModel(
        freeze_clip=args.freeze_clip,
        distribution_merging=args.distribution_merging,
        dropout_rate=args.dropout_rate
    )
    model = model.to(args.device)
    logger.info(f"Model created with {model.num_trainable_parameters():,} trainable parameters")

    # Create loss function
    if args.use_uc_cl:
        criterion = UncertaintyCalibratedContrastiveLoss(
            lambda_cl=args.lambda_uc_cl,
            lambda_consist=args.lambda_consist,
            lambda_var=args.lambda_uc_var,
            temperature=args.uc_temperature,
            target_variance=config.DIST_ALIGN_UC_TARGET_VARIANCE,
        )
        logger.info("Using Uncertainty-Calibrated Distributional Contrastive loss")
    elif args.use_ot_contrastive:
        criterion = DistributionalContrastiveLoss(
            lambda_ot=args.lambda_ot,
            temperature=args.ot_temperature,
            min_sigma=args.min_sigma,
            target_variance=config.DIST_ALIGN_TARGET_VARIANCE,
            lambda_var=args.lambda_var_ot,
        )
        logger.info("Using distributional contrastive loss (OT-based)")
    elif args.use_variance_loss:
        criterion = CombinedDistributionLoss(
            lambda_contrastive=args.lambda_contrastive,
            lambda_kl=args.lambda_kl,
            lambda_var=args.lambda_var,
            temperature=args.temperature,
            kl_type=args.kl_type
        )
        logger.info("Using combined loss with variance regularization")
    else:
        criterion = DistributionAlignmentLoss(
            lambda_contrastive=args.lambda_contrastive,
            lambda_kl=args.lambda_kl,
            temperature=args.temperature,
            kl_type=args.kl_type
        )
        logger.info("Using distribution alignment loss")

    # Create optimizer
    optimizer = create_optimizer(model, args)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
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
        patience_counter = checkpoint.get("patience_counter", 0)
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}, "
                     f"patience_counter: {patience_counter}")

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
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_dataloader, criterion, optimizer,
            args.device, epoch,
            use_variance_loss=args.use_variance_loss,
            use_ot=args.use_ot_contrastive,
            use_uc_cl=args.use_uc_cl
        )

        # Validate
        val_metrics = evaluate(
            model, val_dataloader, criterion,
            args.device,
            use_variance_loss=args.use_variance_loss,
            use_ot=args.use_ot_contrastive,
            use_uc_cl=args.use_uc_cl
        )

        if args.use_uc_cl:
            logger.info(
                f"Epoch {epoch + 1}/{args.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"CL: {train_metrics['contrastive_loss']:.4f}, "
                f"Consist: {train_metrics.get('consistency_loss', 0):.4f}, "
                f"Var: {train_metrics.get('variance_loss', 0):.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val CL: {val_metrics['contrastive_loss']:.4f}, "
                f"Val Consist: {val_metrics.get('consistency_loss', 0):.4f}"
            )
        elif args.use_ot_contrastive:
            logger.info(
                f"Epoch {epoch + 1}/{args.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"OT: {train_metrics['contrastive_loss']:.4f}, "
                f"Var: {train_metrics['kl_loss']:.4f}, "
                f"W2_pos: {train_metrics.get('avg_w2_pos', 0):.2f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val OT: {val_metrics['contrastive_loss']:.4f}, "
                f"Val W2_pos: {val_metrics.get('avg_w2_pos', 0):.2f}"
            )
        else:
            logger.info(
                f"Epoch {epoch + 1}/{args.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Contrastive: {train_metrics['contrastive_loss']:.4f}, "
                f"KL: {train_metrics['kl_loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Contrastive: {val_metrics['contrastive_loss']:.4f}, "
                f"Val KL: {val_metrics['kl_loss']:.4f}"
            )

        # Save best checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_checkpoint_path = checkpoint_dir / "dist_align_best.pt"
            model.save(str(best_checkpoint_path))
            logger.info(f"Best model saved (val_loss: {best_val_loss:.4f}) -> {best_checkpoint_path}")
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
        "epoch": epoch + 1,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
    }
    torch.save(final_state, str(final_checkpoint_path))
    logger.info(f"Final model saved to {final_checkpoint_path}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("Training completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Training failed")
        raise
