"""
GaussianImageDistribution - ProLIP Fine-tuning Script

Full fine-tunes the real ProLIP ViT-H/14 model (image + text encoders and the
learned mu / log(sigma^2) uncertainty heads) on image-caption pairs using
ProLIP's probabilistic inclusion loss (prolip.loss.ProLIPLoss): probabilistic
pairwise contrastive loss + inclusion loss + VIB regularization.

Usage:
    python scripts/train_prolip.py
    python main.py --task train_prolip
"""

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys
# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.caption_dataset import filter_none_collate
from models.prolip_model import ProLIPModel
from prolip.loss import ProLIPLoss
from utils.dataset_factory import build_train_dataset, VALID_DATASETS
from utils.logger import get_logger, log_exception
from utils.lr_scheduler import apply_lr_for_epoch
from utils.seed import set_seed


# Setup logger
logger = get_logger("train_prolip", config.TRAIN_PROLIP_LOG_PATH)

# Exclude faulty CPU cores (e.g. unstable CPU 2) before DataLoader workers and
# torch threads are created. Inherited by forked worker processes.
from utils.cpu_affinity import apply_cpu_affinity
apply_cpu_affinity()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ProLIP Baseline (B3)")

    # Data arguments
    parser.add_argument("--captions-path", type=str, default=None,
                        help="Path to captions parquet file (uses config default if None)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Path to images directory (uses config default if None)")
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=list(VALID_DATASETS),
                        help="Training dataset: selects both the training data and the "
                             "checkpoint-name tag (coco=MSCOCO, flickr=flickr30k)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=config.PROLIP_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.PROLIP_BATCH_SIZE,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=config.PROLIP_LR,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=config.PROLIP_WEIGHT_DECAY,
                        help="Weight decay")
    parser.add_argument("--lr-scheduler", type=str, default=config.LR_SCHEDULER,
                        choices=["none", "cosine"],
                        help="LR schedule: 'cosine' (cosine + linear warmup) or 'none' "
                             "(constant LR)")
    parser.add_argument("--warmup-epochs", type=int, default=config.LR_WARMUP_EPOCHS,
                        help="Linear warmup epochs for the cosine schedule (0 disables warmup)")
    parser.add_argument("--min-lr-ratio", type=float, default=config.LR_MIN_LR_RATIO,
                        help="Cosine floor as a fraction of the base LR")
    parser.add_argument("--temperature", type=float, default=config.PROLIP_TEMPERATURE,
                        help="Legacy alias (ProLIP loss uses the learned logit_scale)")

    # ProLIP inclusion-loss weights
    parser.add_argument("--ppcl-lambda", type=float, default=config.PROLIP_PPCL_LAMBDA,
                        help="Weight for the probabilistic pairwise contrastive loss")
    parser.add_argument("--inclusion-alpha", type=float, default=config.PROLIP_INCLUSION_ALPHA,
                        help="Weight for the inclusion loss (image subset text); 0 disables")
    parser.add_argument("--vib-beta", type=float, default=config.PROLIP_VIB_BETA,
                        help="Weight for the VIB (KL) regularization")

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
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from "
                             "(e.g. checkpoints/prolip_coco_last.pt). "
                             "Restores model weights, optimizer state, epoch, and best_val_loss.")

    return parser.parse_args()


def _forward_loss(
    model: ProLIPModel,
    criterion: ProLIPLoss,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Run the model and compute the ProLIP inclusion loss + mean-retrieval accuracy."""
    outputs = model(pixel_values, input_ids)

    loss = criterion(
        outputs["image_features"], outputs["text_features"],
        logit_scale=outputs["logit_scale"], logit_bias=outputs["logit_bias"],
    )

    # Mean-retrieval accuracy (normalized means), for logging only
    img_mean = outputs["image_features"]["mean"]
    text_mean = outputs["text_features"]["mean"]
    logits = img_mean @ text_mean.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    acc = (logits.argmax(dim=1) == labels).float().mean()

    return loss, {"loss": loss.item(), "acc": acc.item()}


def train_epoch(
    model: ProLIPModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: ProLIPLoss,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    processed_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue
        processed_batches += 1

        pil_images = batch["image"]
        captions_list = batch["captions"]

        # Randomly select one caption per image
        selected_captions = [random.choice(captions) for captions in captions_list]

        pixel_values = model.process_images(pil_images)
        text_inputs = model.process_text(selected_captions)
        input_ids = text_inputs["input_ids"].to(device)

        loss, loss_info = _forward_loss(model, criterion, pixel_values, input_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss_info["loss"]
        total_acc += loss_info["acc"]

        pbar.set_postfix({
            "loss": f"{loss_info['loss']:.4f}",
            "acc": f"{loss_info['acc']:.4f}",
        })

        if (batch_idx + 1) % 10 == 0:
            logger.debug(
                f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss_info['loss']:.4f}, Acc: {loss_info['acc']:.4f}"
            )

    num_batches = max(processed_batches, 1)
    return {"loss": total_loss / num_batches, "acc": total_acc / num_batches}


@torch.no_grad()
def evaluate(
    model: ProLIPModel,
    dataloader: DataLoader,
    criterion: ProLIPLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    processed_batches = 0

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        if batch is None:
            continue
        processed_batches += 1

        pil_images = batch["image"]
        captions_list = batch["captions"]

        # Use first caption for consistency
        selected_captions = [captions[0] for captions in captions_list]

        pixel_values = model.process_images(pil_images)
        text_inputs = model.process_text(selected_captions)
        input_ids = text_inputs["input_ids"].to(device)

        _, loss_info = _forward_loss(model, criterion, pixel_values, input_ids)

        total_loss += loss_info["loss"]
        total_acc += loss_info["acc"]

        pbar.set_postfix({"loss": f"{loss_info['loss']:.4f}"})

    num_batches = max(processed_batches, 1)
    return {"loss": total_loss / num_batches, "acc": total_acc / num_batches}


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Log configuration
    logger.info("=" * 60)
    logger.info("ProLIP (B3) Fine-tuning")
    logger.info("=" * 60)
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"LR scheduler: {args.lr_scheduler} (warmup {args.warmup_epochs}, "
                f"min_lr_ratio {args.min_lr_ratio})")
    logger.info(f"Inclusion loss: ppcl_lambda={args.ppcl_lambda}, "
                f"inclusion_alpha={args.inclusion_alpha}, vib_beta={args.vib_beta}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    # Create model (full fine-tuning, nothing frozen)
    logger.info("Creating model...")
    model = ProLIPModel(freeze=False)
    model = model.to(args.device)
    logger.info(f"Model created with {model.num_trainable_parameters():,} trainable parameters")

    # ProLIP inclusion loss
    criterion = ProLIPLoss(
        ppcl_lambda=args.ppcl_lambda,
        inclusion_alpha=args.inclusion_alpha,
        vib_beta=args.vib_beta,
    )

    # Create optimizer
    optimizer = optim.Adam(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    # Load dataset (selected by --dataset; coco=MSCOCO, flickr=flickr30k train split)
    logger.info(f"Loading training dataset (--dataset {args.dataset})")
    full_dataset = build_train_dataset(
        dataset=args.dataset,
        captions_path=args.captions_path,
        images_dir=args.images_dir,
    )

    # Split into train and validation sets
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=filter_none_collate,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=filter_none_collate,
    )

    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")
    logger.info(f"Batch size: {args.batch_size}, Train batches per epoch: {len(train_dataloader)}")

    # Training loop with early stopping
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else config.CHECKPOINT_DIR

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        logger.info(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=args.device, weights_only=False)
        model.load(resume_path)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(args.device)
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)
        base_lrs = ckpt.get("base_lrs", base_lrs)
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

    logger.info(f"Starting training from epoch {start_epoch + 1}...")
    last_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch

        # Apply LR schedule for this epoch (no-op when scheduler == "none")
        apply_lr_for_epoch(optimizer, base_lrs, epoch, args.epochs,
                           args.warmup_epochs, args.min_lr_ratio,
                           args.lr_scheduler, logger)

        # Train
        train_metrics = train_epoch(model, train_dataloader, optimizer, criterion, args.device, epoch)

        # Validate
        val_metrics = evaluate(model, val_dataloader, criterion, args.device)

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.4f}"
        )

        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_checkpoint_path = checkpoint_dir / f"prolip_{args.dataset}_best.pt"
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
    final_checkpoint_path = checkpoint_dir / f"prolip_{args.dataset}_last.pt"
    final_state = {
        "model_state_dict": model.prolip.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": last_epoch + 1,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "base_lrs": base_lrs,
    }
    torch.save(final_state, str(final_checkpoint_path))
    logger.info(f"Final model saved to {final_checkpoint_path}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("Training completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "ProLIP training failed")
        raise
