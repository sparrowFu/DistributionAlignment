"""
GaussianImageDistribution - Freeze-Align Training Script (Stage 1)

This script trains the Freeze-Align model on image-caption pairs.
Frozen CLIP encoders + trainable projectors with STRUCTURE regularization.

Stage 1: Train projectors on image-caption alignment task.
Stage 2: Load best checkpoint as frozen backbone for VQA fine-tuning.

Usage:
    python scripts/train_freeze_align.py
    python main.py --task train_freeze_align
"""

import argparse
import random
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.caption_dataset import ImageCaptionDataset
from models.freeze_align_model import FreezeAlignModel
from losses.clip_losses import clip_contrastive_loss
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


logger = get_logger("train_freeze_align", config.TRAIN_FREEZE_ALIGN_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Freeze-Align Model")

    # Data arguments
    parser.add_argument("--captions-path", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)

    # Training arguments
    parser.add_argument("--epochs", type=int, default=config.FREEZE_ALIGN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.FREEZE_ALIGN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.FREEZE_ALIGN_LR)
    parser.add_argument("--weight-decay", type=float, default=config.FREEZE_ALIGN_WEIGHT_DECAY)
    parser.add_argument("--temperature", type=float, default=config.FREEZE_ALIGN_TEMPERATURE)
    parser.add_argument("--structure-weight", type=float, default=config.FREEZE_ALIGN_STRUCTURE_WEIGHT)

    # Model arguments
    parser.add_argument("--proj-dim", type=int, default=config.FREEZE_ALIGN_PROJ_DIM)
    parser.add_argument("--dropout-rate", type=float, default=config.FREEZE_ALIGN_DROPOUT_RATE)

    # System arguments
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # Validation and early stopping
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--no-early-stop", action="store_true")

    # Output
    parser.add_argument("--checkpoint-dir", type=str, default=None)

    return parser.parse_args()


def filter_none_collate(batch):
    """Collate function that filters out None values."""
    filtered = [item for item in batch if item is not None]
    if not filtered:
        return None
    return {
        "image": [item["image"] for item in filtered],
        "captions": [item["captions"] for item in filtered],
    }


def train_epoch(
    model: FreezeAlignModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    temperature: float,
    structure_weight: float,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    model.clip_model.eval()  # Keep CLIP in eval mode

    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_structure_loss = 0.0
    total_acc = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue

        pil_images = batch["image"]
        captions_list = batch["captions"]
        selected_captions = [random.choice(captions) for captions in captions_list]

        pixel_values = model.process_images(pil_images).to(device)
        text_inputs = model.process_text(selected_captions)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # Forward: get projected features
        outputs = model(pixel_values, input_ids, attention_mask)
        proj_img = F.normalize(outputs["proj_img_features"], dim=-1)
        proj_text = F.normalize(outputs["proj_text_features"], dim=-1)

        # Contrastive loss on projected features
        clip_loss, loss_info = clip_contrastive_loss(proj_img, proj_text, temperature)

        # STRUCTURE regularization loss
        structure_loss = model.last_extra_loss

        # Total loss
        loss = clip_loss + structure_weight * structure_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_contrastive_loss += loss_info["loss"]
        total_structure_loss += structure_loss.item()
        total_acc += loss_info["acc"]

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "clip": f"{loss_info['loss']:.4f}",
            "struct": f"{structure_loss.item():.4f}",
            "acc": f"{loss_info['acc']:.4f}",
        })

        if (batch_idx + 1) % 10 == 0:
            logger.debug(
                f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.4f}, Acc: {loss_info['acc']:.4f}"
            )

    num_batches = len(dataloader)
    return {
        "loss": total_loss / num_batches,
        "contrastive_loss": total_contrastive_loss / num_batches,
        "structure_loss": total_structure_loss / num_batches,
        "acc": total_acc / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: FreezeAlignModel,
    dataloader: DataLoader,
    temperature: float,
    structure_weight: float,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()

    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_structure_loss = 0.0
    total_acc = 0.0

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        if batch is None:
            continue

        pil_images = batch["image"]
        captions_list = batch["captions"]
        selected_captions = [captions[0] for captions in captions_list]

        pixel_values = model.process_images(pil_images).to(device)
        text_inputs = model.process_text(selected_captions)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        outputs = model(pixel_values, input_ids, attention_mask)
        proj_img = F.normalize(outputs["proj_img_features"], dim=-1)
        proj_text = F.normalize(outputs["proj_text_features"], dim=-1)

        clip_loss, loss_info = clip_contrastive_loss(proj_img, proj_text, temperature)
        structure_loss = model.last_extra_loss
        loss = clip_loss + structure_weight * structure_loss

        total_loss += loss.item()
        total_contrastive_loss += loss_info["loss"]
        total_structure_loss += structure_loss.item()
        total_acc += loss_info["acc"]

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    num_batches = len(dataloader)
    return {
        "loss": total_loss / num_batches,
        "contrastive_loss": total_contrastive_loss / num_batches,
        "structure_loss": total_structure_loss / num_batches,
        "acc": total_acc / num_batches,
    }


def main():
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    logger.info("=" * 60)
    logger.info("Freeze-Align Training (Stage 1: image-caption alignment)")
    logger.info("=" * 60)
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Proj dim: {args.proj_dim}")
    logger.info(f"STRUCTURE weight: {args.structure_weight}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    # Create model
    model = FreezeAlignModel(
        proj_dim=args.proj_dim,
        dropout_rate=args.dropout_rate,
    )
    model = model.to(args.device)
    logger.info(f"Trainable parameters: {model.num_trainable_parameters():,}")

    # Optimizer (only projector params)
    optimizer = optim.Adam(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Load dataset
    captions_path = args.captions_path or config.CAPTIONS_PATH
    images_dir = args.images_dir or config.IMAGES_DIR
    full_dataset = ImageCaptionDataset(
        captions_path=captions_path,
        images_dir=images_dir,
        num_captions=config.NUM_CAPTIONS,
    )

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=filter_none_collate,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=filter_none_collate,
    )
    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else config.CHECKPOINT_DIR

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, train_dataloader, optimizer,
            args.temperature, args.structure_weight, args.device, epoch,
        )
        val_metrics = evaluate(
            model, val_dataloader, args.temperature, args.structure_weight, args.device,
        )

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Contrastive: {train_metrics['contrastive_loss']:.4f}, "
            f"STRUCTURE: {train_metrics['structure_loss']:.4f}, "
            f"Acc: {train_metrics['acc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            model.save(str(checkpoint_dir / "freeze_align_best.pt"))
            logger.info(f"Best model saved (val_loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{args.early_stop_patience}")

        if not args.no_early_stop and patience_counter >= args.early_stop_patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    model.save(str(checkpoint_dir / "freeze_align_last.pt"))
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("Training completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Training failed")
        raise
