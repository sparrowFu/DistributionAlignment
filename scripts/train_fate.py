"""
GaussianImageDistribution - FATE Training Script (Stage 1)

This script trains the FATE model on image-caption pairs.
FATE uses a small projector to inject vision features into text features.

Stage 1: Train projector on image-caption alignment task.
Stage 2: Load best checkpoint as frozen backbone for VQA fine-tuning.

Usage:
    python scripts/train_fate.py
    python main.py --task train_fate
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
from models.fate_model import FATEModel
from losses.clip_losses import clip_contrastive_loss
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


logger = get_logger("train_fate", config.TRAIN_FATE_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train FATE Model")

    # Data arguments
    parser.add_argument("--captions-path", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)

    # Training arguments
    parser.add_argument("--epochs", type=int, default=config.FATE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.FATE_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.FATE_LR)
    parser.add_argument("--weight-decay", type=float, default=config.FATE_WEIGHT_DECAY)
    parser.add_argument("--temperature", type=float, default=config.FATE_TEMPERATURE)

    # Model arguments
    parser.add_argument("--bottleneck-dim", type=int, default=config.FATE_BOTTLENECK_DIM)
    parser.add_argument("--alpha", type=float, default=config.FATE_ALPHA)

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
    model: FATEModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    temperature: float,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    model.clip_model.eval()  # Keep CLIP in eval mode

    total_loss = 0.0
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

        # Forward: get adapted text features
        img_feat = model.encode_image(pixel_values)
        adapted_text_feat = model.encode_text_adapted(input_ids, attention_mask, img_feat)

        # Normalize for contrastive loss
        img_feat_norm = F.normalize(img_feat, dim=-1)
        text_feat_norm = F.normalize(adapted_text_feat, dim=-1)

        # Contrastive loss
        loss, loss_info = clip_contrastive_loss(img_feat_norm, text_feat_norm, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss_info["loss"]
        total_acc += loss_info["acc"]

        pbar.set_postfix({"loss": f"{loss_info['loss']:.4f}", "acc": f"{loss_info['acc']:.4f}"})

        if (batch_idx + 1) % 10 == 0:
            logger.debug(
                f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss_info['loss']:.4f}, Acc: {loss_info['acc']:.4f}"
            )

    num_batches = len(dataloader)
    return {"loss": total_loss / num_batches, "acc": total_acc / num_batches}


@torch.no_grad()
def evaluate(
    model: FATEModel,
    dataloader: DataLoader,
    temperature: float,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()

    total_loss = 0.0
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

        img_feat = model.encode_image(pixel_values)
        adapted_text_feat = model.encode_text_adapted(input_ids, attention_mask, img_feat)

        img_feat_norm = F.normalize(img_feat, dim=-1)
        text_feat_norm = F.normalize(adapted_text_feat, dim=-1)

        loss, loss_info = clip_contrastive_loss(img_feat_norm, text_feat_norm, temperature)

        total_loss += loss_info["loss"]
        total_acc += loss_info["acc"]

        pbar.set_postfix({"loss": f"{loss_info['loss']:.4f}"})

    num_batches = len(dataloader)
    return {"loss": total_loss / num_batches, "acc": total_acc / num_batches}


def main():
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    logger.info("=" * 60)
    logger.info("FATE Training (Stage 1: image-caption alignment)")
    logger.info("=" * 60)
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Bottleneck dim: {args.bottleneck_dim}")
    logger.info(f"Alpha: {args.alpha}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    # Create model
    model = FATEModel(
        bottleneck_dim=args.bottleneck_dim,
        alpha=args.alpha,
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
            model, train_dataloader, optimizer, args.temperature, args.device, epoch,
        )
        val_metrics = evaluate(model, val_dataloader, args.temperature, args.device)

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            model.save(str(checkpoint_dir / "fate_best.pt"))
            logger.info(f"Best model saved (val_loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{args.early_stop_patience}")

        if not args.no_early_stop and patience_counter >= args.early_stop_patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    model.save(str(checkpoint_dir / "fate_last.pt"))
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("Training completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Training failed")
        raise
