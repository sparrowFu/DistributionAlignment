"""
GaussianImageDistribution - VQA Fine-tuning Training Script

This script fine-tunes a VQA classification head on top of a frozen (or
partially frozen) base model. Supports multiple model types with unified
training and evaluation interfaces.

Usage:
    python scripts/train_vqa.py --model-type dist_align
    python scripts/train_vqa.py --model-type freeze_align
    python scripts/train_vqa.py --model-type fate
    python scripts/train_vqa.py --model-type clip_ast
    python scripts/train_vqa.py --model-type clip_zero_shot
    python main.py --task train_vqa --model-type dist_align
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.vqa_dataset import VQADataset, vqa_collate_fn
from models.vqa_model import VQAModel, TRAINABLE_MODEL_TYPES, ALL_MODEL_TYPES
from models.clip_zero_shot import CLIPZeroShotVQA
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


logger = get_logger("train_vqa", config.VQA_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train VQA Classification Head")

    # Model type
    parser.add_argument("--model-type", type=str, required=True,
                        choices=ALL_MODEL_TYPES,
                        help="Base model type to use as backbone")

    # Base model checkpoint
    parser.add_argument("--base-ckpt", type=str, default=None,
                        help="Path to base model checkpoint. "
                             "Defaults to config checkpoint if not specified.")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=config.VQA_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.VQA_BATCH_SIZE,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=config.VQA_LR,
                        help="Learning rate for classification head")
    parser.add_argument("--weight-decay", type=float, default=config.VQA_WEIGHT_DECAY,
                        help="Weight decay")
    parser.add_argument("--hidden-dim", type=int, default=config.VQA_HIDDEN_DIM,
                        help="Hidden dimension of classification head")
    parser.add_argument("--dropout", type=float, default=config.VQA_DROPOUT,
                        help="Dropout rate in classification head")

    # System arguments
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed")
    parser.add_argument("--num-workers", type=int, default=config.VQA_NUM_WORKERS,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    # Validation and early stopping
    parser.add_argument("--val-split", type=float, default=config.VQA_VAL_SPLIT,
                        help="Validation set ratio")
    parser.add_argument("--early-stop-patience", type=int,
                        default=config.VQA_EARLY_STOP_PATIENCE,
                        help="Early stopping patience in epochs")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping")

    # Data paths (defaults from config)
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Path to images directory")

    # Output
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint save directory")

    # CLIP-AST specific
    parser.add_argument("--clip-lr", type=float, default=1e-6,
                        help="Learning rate for CLIP params (clip_ast only)")
    parser.add_argument("--warmup-epochs", type=int, default=1,
                        help="Warmup epochs before parameter selection (clip_ast only)")

    # Freeze-Align specific
    parser.add_argument("--structure-weight", type=float, default=0.1,
                        help="Weight for STRUCTURE regularization loss (freeze_align only)")

    return parser.parse_args()


def get_default_base_ckpt(model_type: str) -> str:
    """Get default base model checkpoint path for the given model type."""
    ckpt_map = {
        "dist_align": str(config.DIST_ALIGN_BEST_CKPT),
        "clip_baseline": str(config.CLIP_BASELINE_BEST_CKPT),
        "freeze_align": str(config.FREEZE_ALIGN_BEST_CKPT),
        "fate": str(config.FATE_BEST_CKPT),
        "clip_ast": str(config.CLIP_AST_BEST_CKPT),
    }
    return ckpt_map.get(model_type, None)


def get_vqa_ckpt_name(model_type: str) -> str:
    """Get VQA checkpoint filename for the given model type."""
    return f"vqa_{model_type}_best.pt"


def get_optimizer_for_model(
    model: VQAModel,
    model_type: str,
    lr: float,
    clip_lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    """
    Create optimizer appropriate for the model type.

    Different model types have different parameter groups:
        - dist_align, clip_baseline, freeze_align, fate:
            Only classifier params (and adapter params) are trainable
        - clip_ast: classifier params + selected CLIP params (differential LR)
    """
    if model_type == "clip_ast":
        # CLIP-AST: differential learning rates
        clip_params = []
        classifier_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name:
                classifier_params.append(param)
            else:
                clip_params.append(param)

        param_groups = [
            {"params": classifier_params, "lr": lr},
            {"params": clip_params, "lr": clip_lr},
        ]
        logger.info(
            f"CLIP-AST optimizer: classifier LR={lr}, CLIP LR={clip_lr}, "
            f"classifier params={sum(p.numel() for p in classifier_params):,}, "
            f"CLIP params={sum(p.numel() for p in clip_params):,}"
        )
        return optim.AdamW(param_groups, weight_decay=weight_decay)

    else:
        # Standard: only classifier and adapter params
        trainable = model.trainable_parameters()
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
        return optim.Adam(trainable, lr=lr, weight_decay=weight_decay)


def train_epoch(
    model: VQAModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    structure_weight: float = 0.1,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    # Ensure base model stays in eval mode (for dropout/batchnorm in frozen parts)
    if model.model_type != "clip_ast":
        model.base_model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Per-type accuracy tracking
    type_correct = defaultdict(int)
    type_total = defaultdict(int)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue

        # Get data
        pil_images = batch["images"]          # List[PIL.Image]
        questions = batch["questions"]         # List[str]
        answer_indices = batch["answer_indices"]  # List[int]
        question_types = batch["question_types"]  # List[int]

        # Process with CLIP processor (on CPU, then move to device)
        pixel_values = model.process_images(pil_images).to(device)
        text_inputs = model.process_text(questions)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        labels = torch.tensor(answer_indices, dtype=torch.long, device=device)

        # Forward
        logits = model(pixel_values, input_ids, attention_mask)

        # Classification loss
        ce_loss = criterion(logits, labels)

        # Extra loss (e.g., STRUCTURE regularization for Freeze-Align)
        extra_loss = model.extra_loss
        if extra_loss is not None:
            total_loss_val = ce_loss + structure_weight * extra_loss
        else:
            total_loss_val = ce_loss

        # Backward
        optimizer.zero_grad()
        total_loss_val.backward()
        optimizer.step()

        # Metrics
        total_loss += ce_loss.item()
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

        # Per-type tracking
        for i, qt in enumerate(question_types):
            type_total[qt] += 1
            if preds[i].item() == labels[i].item():
                type_correct[qt] += 1

        # Progress bar
        acc = total_correct / max(total_samples, 1)
        pbar.set_postfix({"loss": f"{ce_loss.item():.4f}", "acc": f"{acc:.4f}"})

        if (batch_idx + 1) % 50 == 0:
            logger.debug(
                f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {ce_loss.item():.4f}, Acc: {acc:.4f}"
            )

    num_batches = max(len(dataloader), 1)
    metrics = {
        "loss": total_loss / num_batches,
        "accuracy": total_correct / max(total_samples, 1),
        "total_samples": total_samples,
    }

    # Per-type accuracy
    for qt in sorted(type_total.keys()):
        metrics[f"acc_type_{qt}"] = type_correct[qt] / max(type_total[qt], 1)

    return metrics


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model (works for both VQAModel and CLIPZeroShotVQA)."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    type_correct = defaultdict(int)
    type_total = defaultdict(int)

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        if batch is None:
            continue

        pil_images = batch["images"]
        questions = batch["questions"]
        answer_indices = batch["answer_indices"]
        question_types = batch["question_types"]

        pixel_values = model.process_images(pil_images).to(device)
        text_inputs = model.process_text(questions)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        labels = torch.tensor(answer_indices, dtype=torch.long, device=device)

        logits = model(pixel_values, input_ids, attention_mask)

        # For zero-shot, logits may not align with answer_vocab directly
        # But the interface is the same
        total_loss += criterion(logits, labels).item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        for i, qt in enumerate(question_types):
            type_total[qt] += 1
            if preds[i].item() == labels[i].item():
                type_correct[qt] += 1

        acc = total_correct / max(total_samples, 1)
        pbar.set_postfix({"loss": f"{total_loss / max(len(dataloader), 1):.4f}", "acc": f"{acc:.4f}"})

    num_batches = max(len(dataloader), 1)
    metrics = {
        "loss": total_loss / num_batches,
        "accuracy": total_correct / max(total_samples, 1),
        "total_samples": total_samples,
    }

    for qt in sorted(type_total.keys()):
        metrics[f"acc_type_{qt}"] = type_correct[qt] / max(type_total[qt], 1)

    return metrics


def evaluate_zero_shot(
    answer_vocab: Dict[str, int],
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate CLIP Zero-Shot on VQA test set.

    Args:
        answer_vocab: Answer → index mapping
        dataloader: Test dataloader
        device: Device

    Returns:
        Evaluation metrics
    """
    # Build answer list from vocab (index → answer)
    idx_to_answer = {v: k for k, v in answer_vocab.items()}
    answer_list = [idx_to_answer[i] for i in range(len(answer_vocab))]

    model = CLIPZeroShotVQA(answer_list=answer_list)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    logger.info("Evaluating CLIP Zero-Shot on test set...")
    metrics = evaluate(model, dataloader, criterion, device)

    logger.info(f"CLIP Zero-Shot Accuracy: {metrics['accuracy']:.4f}")
    return metrics


def main():
    """Main training function."""
    args = parse_args()

    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Determine paths
    base_ckpt = args.base_ckpt or get_default_base_ckpt(args.model_type)
    images_dir = Path(args.images_dir) if args.images_dir else config.VQA_IMAGES_DIR
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else config.CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info("=" * 60)
    logger.info("VQA Fine-tuning Training")
    logger.info("=" * 60)
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Base checkpoint: {base_ckpt}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Hidden dim: {args.hidden_dim}")
    logger.info(f"Dropout: {args.dropout}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Val split: {args.val_split}")
    logger.info(f"Images dir: {images_dir}")
    if args.model_type == "clip_ast":
        logger.info(f"CLIP LR: {args.clip_lr}")
        logger.info(f"Warmup epochs: {args.warmup_epochs}")
    if args.model_type == "freeze_align":
        logger.info(f"STRUCTURE weight: {args.structure_weight}")
    logger.info("=" * 60)

    # Load full training dataset to build answer vocabulary
    logger.info("Loading training dataset to build answer vocabulary...")
    train_full = VQADataset(
        questions_path=config.VQA_TRAIN_QUESTIONS,
        img_filenames_path=config.VQA_TRAIN_IMG_FILENAMES,
        types_path=config.VQA_TRAIN_TYPES,
        answers_path=config.VQA_TRAIN_ANSWERS,
        images_dir=images_dir,
    )
    answer_vocab = train_full.get_answer_vocab()
    num_classes = len(answer_vocab)
    logger.info(f"Answer vocabulary size: {num_classes}")

    # === CLIP Zero-Shot: no training, evaluate directly ===
    if args.model_type == "clip_zero_shot":
        logger.info("CLIP Zero-Shot mode: skipping training, evaluating directly...")

        test_dataset = VQADataset(
            questions_path=config.VQA_TEST_QUESTIONS,
            img_filenames_path=config.VQA_TEST_IMG_FILENAMES,
            types_path=config.VQA_TEST_TYPES,
            answers_path=config.VQA_TEST_ANSWERS,
            images_dir=images_dir,
            answer_vocab=answer_vocab,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=vqa_collate_fn,
        )

        test_metrics = evaluate_zero_shot(answer_vocab, test_dataloader, args.device)

        logger.info("=" * 60)
        logger.info("CLIP Zero-Shot Test Set Results")
        logger.info("=" * 60)
        logger.info(f"Overall Accuracy: {test_metrics['accuracy']:.4f}")
        for k in range(4):
            acc_key = f"acc_type_{k}"
            if acc_key in test_metrics:
                logger.info(f"  Type {k} Accuracy: {test_metrics[acc_key]:.4f}")
        logger.info(f"Total test samples: {test_metrics['total_samples']}")
        logger.info("=" * 60)
        return

    # === Trainable models ===

    # Split into train and validation
    val_size = int(len(train_full) * args.val_split)
    train_size = len(train_full) - val_size
    train_dataset, val_dataset = random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=vqa_collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=vqa_collate_fn,
    )

    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")

    # Create model
    logger.info(f"Creating VQA model with {args.model_type} backbone...")
    model = VQAModel(
        model_type=args.model_type,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        answer_vocab=answer_vocab,
        base_ckpt_path=base_ckpt,
        device=args.device,
    )
    model = model.to(args.device)

    # CLIP-AST: unfreeze all CLIP params for warmup phase
    if args.model_type == "clip_ast":
        logger.info("CLIP-AST: Unfreezing all CLIP params for warmup phase...")
        model.base_model.unfreeze_all_clip()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.num_trainable_parameters()
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {total_params - trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer_for_model(
        model, args.model_type, args.lr, args.clip_lr, args.weight_decay
    )

    # Training loop
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    vqa_ckpt_path = checkpoint_dir / get_vqa_ckpt_name(args.model_type)

    logger.info("Starting training...")
    for epoch in range(args.epochs):
        # CLIP-AST: after warmup, select parameters
        if (args.model_type == "clip_ast"
                and epoch == args.warmup_epochs
                and not model.base_model._param_selected):
            logger.info(f"CLIP-AST: Selecting parameters after warmup epoch {epoch}...")
            model.base_model.select_parameters()
            # Recreate optimizer with updated parameter groups
            optimizer = get_optimizer_for_model(
                model, args.model_type, args.lr, args.clip_lr, args.weight_decay
            )

        # Train
        train_metrics = train_epoch(
            model, train_dataloader, criterion, optimizer,
            args.device, epoch,
            structure_weight=args.structure_weight,
        )

        # Validate
        val_metrics = evaluate(
            model, val_dataloader, criterion, args.device,
        )

        # Log metrics
        type_acc_str = " | ".join(
            f"Type {k}: {train_metrics.get(f'acc_type_{k}', 0):.4f}"
            for k in range(4)
        )
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f} "
            f"({type_acc_str}) | "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        # Save best checkpoint (by validation loss)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_val_acc = val_metrics["accuracy"]
            model.save(str(vqa_ckpt_path))
            logger.info(
                f"Best model saved (val_loss: {best_val_loss:.4f}, "
                f"val_acc: {best_val_acc:.4f}) -> {vqa_ckpt_path}"
            )
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{args.early_stop_patience}"
            )

        # Early stopping
        if not args.no_early_stop and patience_counter >= args.early_stop_patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Save last checkpoint
    last_ckpt_path = checkpoint_dir / get_vqa_ckpt_name(args.model_type).replace(
        "_best.pt", "_last.pt"
    )
    model.save(str(last_ckpt_path))
    logger.info(f"Last model saved to {last_ckpt_path}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}, accuracy: {best_val_acc:.4f}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_dataset = VQADataset(
        questions_path=config.VQA_TEST_QUESTIONS,
        img_filenames_path=config.VQA_TEST_IMG_FILENAMES,
        types_path=config.VQA_TEST_TYPES,
        answers_path=config.VQA_TEST_ANSWERS,
        images_dir=images_dir,
        answer_vocab=answer_vocab,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=vqa_collate_fn,
    )

    # Load best model for test evaluation
    model.load_classifier(str(vqa_ckpt_path))
    test_metrics = evaluate(model, test_dataloader, criterion, args.device)

    logger.info("=" * 60)
    logger.info("Test Set Results")
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    for k in range(4):
        acc_key = f"acc_type_{k}"
        if acc_key in test_metrics:
            logger.info(f"  Type {k} Accuracy: {test_metrics[acc_key]:.4f}")
    logger.info(f"Total test samples: {test_metrics['total_samples']}")
    logger.info("=" * 60)
    logger.info("Training completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "VQA training failed")
        raise
