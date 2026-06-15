"""
GaussianImageDistribution - CLIP Baseline Training Script

This script trains the CLIP baseline model on image-caption pairs
using contrastive learning with full fine-tuning.

Usage:
    python scripts/train_clip_baseline.py
    python main.py --task train_clip_baseline
"""

import argparse
import random
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
from models.clip_baseline import CLIPFineTuneBaseline
from losses.clip_losses import clip_contrastive_loss
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


# Setup logger
logger = get_logger("train_clip_baseline", config.TRAIN_CLIP_BASELINE_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CLIP Baseline Model")

    # Data arguments
    parser.add_argument("--captions-path", type=str, default=None,
                        help="Path to captions parquet file (uses config default if None)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Path to images directory (uses config default if None)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=config.CLIP_BASELINE_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.CLIP_BASELINE_BATCH_SIZE,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=config.CLIP_BASELINE_LR,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=config.CLIP_BASELINE_WEIGHT_DECAY,
                        help="Weight decay")
    parser.add_argument("--temperature", type=float, default=config.CLIP_BASELINE_TEMPERATURE,
                        help="Temperature for contrastive loss")

    # Model arguments
    parser.add_argument("--freeze-image", action="store_true",
                        help="Freeze image encoder")
    parser.add_argument("--no-freeze-image", action="store_false", dest="freeze_image",
                        help="Don't freeze image encoder")
    parser.add_argument("--freeze-text", action="store_true",
                        help="Freeze text encoder")
    parser.add_argument("--no-freeze-text", action="store_false", dest="freeze_text",
                        help="Don't freeze text encoder")

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
                             "(e.g. checkpoints/clip_baseline_last.pt). "
                             "Restores model weights, optimizer state, epoch, and best_val_loss.")

    return parser.parse_args()


def train_epoch(
    model: CLIPFineTuneBaseline,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    temperature: float,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_acc = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue

        # Get data - PIL images and text lists
        pil_images = batch["image"]
        captions_list = batch["captions"]

        # Randomly select one caption per image
        selected_captions = [random.choice(captions) for captions in captions_list]

        # Process with CLIP processor
        pixel_values = model.process_images(pil_images).to(device)
        text_inputs = model.process_text(selected_captions)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # Forward pass
        image_features, text_features = model(
            images=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Compute loss
        loss, loss_info = clip_contrastive_loss(
            image_features, text_features,
            temperature=temperature
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss_info["loss"]
        total_acc += loss_info["acc"]

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_info['loss']:.4f}",
            'acc': f"{loss_info['acc']:.4f}"
        })

        # Log every N batches
        if (batch_idx + 1) % 10 == 0:
            logger.debug(
                f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss_info['loss']:.4f}, Acc: {loss_info['acc']:.4f}"
            )

    # Compute averages
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'acc': total_acc / num_batches,
    }

    return metrics


@torch.no_grad()
def evaluate(
    model: CLIPFineTuneBaseline,
    dataloader: DataLoader,
    temperature: float,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()

    total_loss = 0.0
    total_acc = 0.0

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        if batch is None:
            continue

        # Get data - PIL images and text lists
        pil_images = batch["image"]
        captions_list = batch["captions"]

        # Use first caption for consistency
        selected_captions = [captions[0] for captions in captions_list]

        # Process with CLIP processor
        pixel_values = model.process_images(pil_images).to(device)
        text_inputs = model.process_text(selected_captions)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # Forward pass
        image_features, text_features = model(
            images=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Compute loss
        loss, loss_info = clip_contrastive_loss(
            image_features, text_features,
            temperature=temperature
        )

        total_loss += loss_info["loss"]
        total_acc += loss_info["acc"]

        pbar.set_postfix({'loss': f"{loss_info['loss']:.4f}"})

    # Compute averages
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'acc': total_acc / num_batches,
    }

    return metrics


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Log configuration
    logger.info("=" * 60)
    logger.info("CLIP Baseline Training")
    logger.info("=" * 60)
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Freeze image: {args.freeze_image}")
    logger.info(f"Freeze text: {args.freeze_text}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    # Create model
    logger.info("Creating model...")
    model = CLIPFineTuneBaseline(
        freeze_image=args.freeze_image,
        freeze_text=args.freeze_text
    )
    model = model.to(args.device)
    logger.info(f"Model created with {model.num_trainable_parameters():,} trainable parameters")

    # Create optimizer
    optimizer = optim.Adam(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

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
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else config.CHECKPOINT_DIR

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        logger.info(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(args.device)
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float('inf'))
        patience_counter = ckpt.get("patience_counter", 0)
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

    logger.info(f"Starting training from epoch {start_epoch + 1}...")
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_dataloader, optimizer,
            args.temperature, args.device, epoch
        )

        # Validate
        val_metrics = evaluate(
            model, val_dataloader,
            args.temperature, args.device
        )

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.4f}"
        )

        # Save best checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_checkpoint_path = checkpoint_dir / "clip_baseline_best.pt"
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
    final_checkpoint_path = checkpoint_dir / "clip_baseline_last.pt"
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
