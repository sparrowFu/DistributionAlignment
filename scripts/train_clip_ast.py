"""
GaussianImageDistribution - CLIP-AST Training Script (Stage 1)

This script trains the CLIP-AST model on image-caption pairs.
CLIP-AST adaptively selects critical CLIP parameters for fine-tuning.

Stage 1: Selective fine-tuning on image-caption alignment task.
Stage 2: Load best checkpoint as backbone for VQA fine-tuning.

Usage:
    python scripts/train_clip_ast.py
    python main.py --task train_clip_ast
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
from models.clip_ast_model import CLIPASTModel
from losses.clip_losses import clip_contrastive_loss
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


logger = get_logger("train_clip_ast", config.TRAIN_CLIP_AST_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CLIP-AST Model")

    # Data arguments
    parser.add_argument("--captions-path", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)

    # Training arguments
    parser.add_argument("--epochs", type=int, default=config.CLIP_AST_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.CLIP_AST_BATCH_SIZE)
    parser.add_argument("--clip-lr", type=float, default=config.CLIP_AST_CLIP_LR)
    parser.add_argument("--weight-decay", type=float, default=config.CLIP_AST_WEIGHT_DECAY)
    parser.add_argument("--temperature", type=float, default=config.CLIP_AST_TEMPERATURE)

    # Model arguments
    parser.add_argument("--select-ratio", type=float, default=config.CLIP_AST_SELECT_RATIO)
    parser.add_argument("--warmup-epochs", type=int, default=config.CLIP_AST_WARMUP_EPOCHS)

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
    model: CLIPASTModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    temperature: float,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

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

        # Forward
        img_feat = model.encode_image(pixel_values)
        text_feat = model.encode_text(input_ids, attention_mask)

        # Normalize for contrastive loss
        img_feat = F.normalize(img_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        # Contrastive loss
        loss, loss_info = clip_contrastive_loss(img_feat, text_feat, temperature)

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
    model: CLIPASTModel,
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
        text_feat = model.encode_text(input_ids, attention_mask)

        img_feat = F.normalize(img_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        loss, loss_info = clip_contrastive_loss(img_feat, text_feat, temperature)

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
    logger.info("CLIP-AST Training (Stage 1: image-caption alignment)")
    logger.info("=" * 60)
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"CLIP LR: {args.clip_lr}")
    logger.info(f"Select ratio: {args.select_ratio}")
    logger.info(f"Warmup epochs: {args.warmup_epochs}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    # Create model
    model = CLIPASTModel(select_ratio=args.select_ratio)
    model = model.to(args.device)

    # Unfreeze all CLIP params for warmup phase
    model.unfreeze_all_clip()
    logger.info(f"Trainable parameters: {model.num_trainable_parameters():,}")

    # Optimizer (all CLIP params with small LR)
    optimizer = optim.AdamW(
        model.trainable_parameters(),
        lr=args.clip_lr,
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
        # After warmup: select parameters
        if epoch == args.warmup_epochs and not model._param_selected:
            logger.info(f"Selecting parameters after warmup epoch {epoch}...")
            model.select_parameters()
            # Recreate optimizer with only selected params
            optimizer = optim.AdamW(
                model.trainable_parameters(),
                lr=args.clip_lr,
                weight_decay=args.weight_decay,
            )
            logger.info(f"Trainable after selection: {model.num_trainable_parameters():,}")

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
            model.save(str(checkpoint_dir / "clip_ast_best.pt"))
            logger.info(f"Best model saved (val_loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{args.early_stop_patience}")

        if not args.no_early_stop and patience_counter >= args.early_stop_patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    model.save(str(checkpoint_dir / "clip_ast_last.pt"))
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("Training completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Training failed")
        raise
