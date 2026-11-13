"""
CLIP-L/14 Set-FT Training Script (rebuttal exp 1)

Trains the same CLIPFineTuneBaseline as train_clip_baseline.py from the same
pretrained weights, but replaces the per-step single random caption with the
full five-caption set:

    B images x K captions -> encode all BK captions -> average per image into
    one set vector -> L2-normalize -> BxB cosine matrix with the image
    features -> cross-entropy on the matrix and its transpose (diagonal =
    positives), averaged.

Everything else -- freeze scope, image batch size, optimizer, LR schedule,
update budget, early stopping and the validation criterion (first caption +
single-caption contrastive loss) -- is kept identical to train_clip_baseline.py
so that "original vs Set-FT" isolates only the text-supervision change.
Hyperparameter defaults therefore reuse the config.CLIP_BASELINE_* values.
"""

import argparse
import gc
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
from data.caption_dataset import filter_none_collate
from models.clip_baseline import CLIPFineTuneBaseline
from utils.dataset_factory import build_train_dataset, VALID_DATASETS
from losses.clip_losses import clip_contrastive_loss
from utils.logger import get_logger, log_exception
from utils.lr_scheduler import apply_lr_for_epoch
from utils.seed import set_seed


logger = get_logger("train_clip_setft", config.TRAIN_CLIP_SETFT_LOG_PATH)

# Exclude faulty CPU cores (e.g. unstable CPU 2) before DataLoader workers and
# torch threads are created. Inherited by forked worker processes.
from utils.cpu_affinity import apply_cpu_affinity
apply_cpu_affinity()


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP Set-FT (rebuttal exp 1)")

    parser.add_argument("--captions-path", type=str, default=None,
                        help="Path to captions parquet file (uses config default if None)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Path to images directory (uses config default if None)")
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=list(VALID_DATASETS),
                        help="Training dataset: selects both the training data and the "
                             "checkpoint-name tag (coco=MSCOCO, flickr=flickr30k)")

    # identical to train_clip_baseline.py by design (see module docstring)
    parser.add_argument("--epochs", type=int, default=config.CLIP_BASELINE_EPOCHS,
                        help="Number of training epochs (same as CLIP baseline)")
    parser.add_argument("--batch-size", type=int, default=config.CLIP_BASELINE_BATCH_SIZE,
                        help="Image batch size; each image carries its full K captions")
    parser.add_argument("--lr", type=float, default=config.CLIP_BASELINE_LR,
                        help="Learning rate (same as CLIP baseline)")
    parser.add_argument("--weight-decay", type=float, default=config.CLIP_BASELINE_WEIGHT_DECAY,
                        help="Weight decay (same as CLIP baseline)")
    parser.add_argument("--lr-scheduler", type=str, default=config.LR_SCHEDULER,
                        choices=["none", "cosine"],
                        help="LR schedule (same as CLIP baseline)")
    parser.add_argument("--warmup-epochs", type=int, default=config.LR_WARMUP_EPOCHS,
                        help="Linear warmup epochs for the cosine schedule (0 disables warmup)")
    parser.add_argument("--min-lr-ratio", type=float, default=config.LR_MIN_LR_RATIO,
                        help="Cosine floor as a fraction of the base LR")
    parser.add_argument("--temperature", type=float, default=config.CLIP_BASELINE_TEMPERATURE,
                        help="Temperature for contrastive loss (same as CLIP baseline)")

    parser.add_argument("--freeze-image", action="store_true",
                        help="Freeze image encoder")
    parser.add_argument("--no-freeze-image", action="store_false", dest="freeze_image",
                        help="Don't freeze image encoder")
    parser.add_argument("--freeze-text", action="store_true",
                        help="Freeze text encoder")
    parser.add_argument("--no-freeze-text", action="store_false", dest="freeze_text",
                        help="Don't freeze text encoder")

    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed (same value as the CLIP baseline run "
                             "reproduces the same train/val split)")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    # Validation and early stopping arguments (identical protocol to the baseline)
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation set ratio (default: 0.1)")
    parser.add_argument("--early-stop-patience", type=int, default=3,
                        help="Early stopping patience in epochs (default: 3)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping")

    parser.add_argument("--max-batches-per-epoch", type=int, default=0,
                        help="Cap on processed batches per epoch (<=0: no cap; "
                             "smoke-run convenience)")

    # Output arguments
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint directory (uses config default if None)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from "
                             "(e.g. checkpoints/clip_setft_coco_last.pt). "
                             "Restores model weights, optimizer state, epoch, and best_val_loss.")

    return parser.parse_args()


def _release_loader(loader: DataLoader) -> None:
    """Release a finished DataLoader's worker processes now.

    torch shuts the workers down itself when the iterator runs to normal
    exhaustion, but not when the loop exits early (--max-batches-per-epoch
    break) or raises; the workers then linger until a GC pass collects the
    abandoned iterator. Call this after each epoch so no worker processes
    are carried across the epoch boundary.
    """
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:  # persistent_workers-style loader (future-proof)
        iterator._shutdown_workers()
        loader._iterator = None
    gc.collect()  # drop the abandoned iterator (early break/raise) -> workers exit


def _save_last_checkpoint(checkpoint_dir: Path, model, optimizer, dataset_tag: str,
                          epoch: int, best_val_loss: float, patience_counter: int,
                          base_lrs) -> Path:
    """Full training state for --resume, rewritten after every epoch (atomic
    tmp+replace) so a crash only costs the current epoch -- the PCME++ run
    already lost a multi-day training to an end-of-training-only last.pt."""
    last_path = checkpoint_dir / f"clip_setft_{dataset_tag}_last.pt"
    tmp_path = checkpoint_dir / f"clip_setft_{dataset_tag}_last.pt.tmp"
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch + 1,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "base_lrs": base_lrs,
    }
    torch.save(state, str(tmp_path))
    tmp_path.replace(last_path)
    return last_path


def _set_text_features(model, captions_list, device):
    """Encode every caption of every image and pool per image.

    captions_list: list of B per-image caption lists (K captions each).
    Returns L2-normalized set vectors of shape (B, D): raw caption features
    are averaged first, the set vector is normalized once afterwards.
    """
    k_counts = {len(cl) for cl in captions_list}
    if len(k_counts) != 1:
        raise ValueError(
            f"Set-FT needs the same caption count for every image in a batch, "
            f"got {sorted(k_counts)}; drop or re-filter the offending items")
    K = k_counts.pop()

    flat = [c for cl in captions_list for c in cl]  # image-major: (B*K,)
    text_inputs = model.process_text(flat)
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    # normalize=False: keep raw features so the average happens before the
    # single normalization of the set vector
    text_features = model.encode_text(
        input_ids, attention_mask, normalize=False)
    set_features = text_features.view(len(captions_list), K, -1).mean(dim=1)
    return F.normalize(set_features, dim=-1)


def train_epoch(
    model: CLIPFineTuneBaseline,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    temperature: float,
    device: torch.device,
    epoch: int,
    max_batches: int = 0,
) -> Dict[str, float]:
    """Train for one epoch on the five-caption set objective."""
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    processed_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for batch_idx, batch in enumerate(pbar):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        if batch is None:
            continue
        processed_batches += 1

        pil_images = batch["image"]
        captions_list = batch["captions"]

        pixel_values = model.process_images(pil_images).to(device)
        image_features = model.encode_image(pixel_values, normalize=True)

        set_features = _set_text_features(model, captions_list, device)

        loss, loss_info = clip_contrastive_loss(
            image_features, set_features,
            temperature=temperature
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss_info["loss"]
        total_acc += loss_info["acc"]

        pbar.set_postfix({
            'loss': f"{loss_info['loss']:.4f}",
            'acc': f"{loss_info['acc']:.4f}"
        })

        if (batch_idx + 1) % 10 == 0:
            logger.debug(
                f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss_info['loss']:.4f}, Acc: {loss_info['acc']:.4f}"
            )

    num_batches = max(processed_batches, 1)
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
    """Validation, deliberately identical to train_clip_baseline.evaluate:
    first caption per image + single-caption contrastive loss, so the
    best-model selection criterion matches the original-CLIP control."""
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

        selected_captions = [captions[0] for captions in captions_list]

        pixel_values = model.process_images(pil_images).to(device)
        text_inputs = model.process_text(selected_captions)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        image_features, text_features = model(
            images=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss, loss_info = clip_contrastive_loss(
            image_features, text_features,
            temperature=temperature
        )

        total_loss += loss_info["loss"]
        total_acc += loss_info["acc"]

        pbar.set_postfix({'loss': f"{loss_info['loss']:.4f}"})

    num_batches = max(processed_batches, 1)
    metrics = {
        'loss': total_loss / num_batches,
        'acc': total_acc / num_batches,
    }

    return metrics


def main():
    args = parse_args()

    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    logger.info("=" * 60)
    logger.info("CLIP Set-FT Training (rebuttal exp 1)")
    logger.info("=" * 60)
    logger.info(f"Text supervision: all K captions averaged into a set vector")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Image batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"LR scheduler: {args.lr_scheduler} (warmup {args.warmup_epochs}, "
                f"min_lr_ratio {args.min_lr_ratio})")
    logger.info(f"Freeze image: {args.freeze_image}")
    logger.info(f"Freeze text: {args.freeze_text}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    logger.info("Creating model...")
    model = CLIPFineTuneBaseline(
        freeze_image=args.freeze_image,
        freeze_text=args.freeze_text
    )
    model = model.to(args.device)
    logger.info(f"Model created with {model.num_trainable_parameters():,} trainable parameters")

    optimizer = optim.Adam(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    # coco=MSCOCO, flickr=flickr30k train split; same factory and split seed
    # as the CLIP baseline run
    logger.info(f"Loading training dataset (--dataset {args.dataset})")
    full_dataset = build_train_dataset(
        dataset=args.dataset,
        captions_path=args.captions_path,
        images_dir=args.images_dir,
    )

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
    logger.info(f"Image batch size: {args.batch_size}, Train batches per epoch: {len(train_dataloader)}")

    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else config.CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

        train_metrics = train_epoch(
            model, train_dataloader, optimizer,
            args.temperature, args.device, epoch,
            max_batches=args.max_batches_per_epoch,
        )
        _release_loader(train_dataloader)

        val_metrics = evaluate(
            model, val_dataloader,
            args.temperature, args.device
        )
        _release_loader(val_dataloader)

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.4f}"
        )

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = checkpoint_dir / f"clip_setft_{args.dataset}_best.pt"
            tmp_path = checkpoint_dir / f"clip_setft_{args.dataset}_best.pt.tmp"
            model.save(str(tmp_path))
            tmp_path.replace(best_path)  # atomic: a crash mid-write cannot corrupt best.pt
            logger.info(f"Best model saved (val_loss: {best_val_loss:.4f}) -> {best_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{args.early_stop_patience}")

        # crash-safe resume point: overwrite every epoch (atomic)
        last_path = _save_last_checkpoint(
            checkpoint_dir, model, optimizer, args.dataset,
            epoch, best_val_loss, patience_counter, base_lrs)
        logger.info(f"Last checkpoint saved (epoch {epoch + 1}) -> {last_path}")

        if not args.no_early_stop and patience_counter >= args.early_stop_patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("Training completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Training failed")
        raise
