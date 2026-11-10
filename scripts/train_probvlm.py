"""
ProbVLM Training Script (B4)

Trains the ProbVLM post-hoc probabilistic adapter (Upadhyay et al., ICCV
2023) on image-caption pairs: two BayesCap MLP adapters on the FROZEN CLIP
ViT-L/14 projected embeddings, optimized with the official TempComb loss
(T1*L1 + T2*GenGauss NLL) against the frozen CLIP features, plus the
cross-modal terms weighted by cross_modal_lambda.

Fidelity notes w.r.t. the official train_probVLM.py:
- The official per-sample dataloader yields one (image, caption) pair per
  item; selecting one random caption of K=5 per image per step samples the
  same uniform distribution over captions.
- The official code caps each epoch at 2000 batches (its `if idx > 2000:
  break` is an off-by-one; we cap at exactly --max-batches-per-epoch).
- The official CosineAnnealingLR is stepped OUTSIDE the epoch loop
  (indentation bug -> effectively constant LR). We implement the intent:
  per-epoch cosine annealing with warmup 0 and min_lr_ratio 0.0 (defaults
  below; the config-wide LR_WARMUP_EPOCHS/LR_MIN_LR_RATIO do not apply).
- The official selects the best checkpoint by val MAE(mu, frozen feats);
  we follow the project pattern (best val loss) and log the val MAE
  alongside as a diagnostic.
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
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.caption_dataset import filter_none_collate
from losses.probvlm_losses import TempCombLoss, probvlm_total_loss
from models.probvlm_model import ProbVLMModel
from utils.dataset_factory import build_train_dataset, VALID_DATASETS
from utils.logger import get_logger, log_exception
from utils.lr_scheduler import apply_lr_for_epoch
from utils.seed import set_seed


logger = get_logger("train_probvlm", config.TRAIN_PROBVLM_LOG_PATH)

# Exclude faulty CPU cores (e.g. unstable CPU 2) before DataLoader workers and
# torch threads are created. Inherited by forked worker processes.
from utils.cpu_affinity import apply_cpu_affinity
apply_cpu_affinity()


def parse_args():
    parser = argparse.ArgumentParser(description="Train ProbVLM Baseline (B4)")

    parser.add_argument("--captions-path", type=str, default=None,
                        help="Path to captions parquet file (uses config default if None)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Path to images directory (uses config default if None)")
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=list(VALID_DATASETS),
                        help="Training dataset: selects both the training data and the "
                             "checkpoint-name tag (coco=MSCOCO, flickr=flickr30k)")

    parser.add_argument("--epochs", type=int, default=config.PROBVLM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.PROBVLM_BATCH_SIZE,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=config.PROBVLM_LR,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=config.PROBVLM_WEIGHT_DECAY,
                        help="Weight decay")
    parser.add_argument("--lr-scheduler", type=str, default=config.LR_SCHEDULER,
                        choices=["none", "cosine"],
                        help="LR schedule: 'cosine' or 'none' (constant LR)")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Linear warmup epochs (ProbVLM default 0: the official "
                             "schedule is pure cosine, unlike the config-wide default)")
    parser.add_argument("--min-lr-ratio", type=float, default=0.0,
                        help="Cosine floor as a fraction of the base LR (ProbVLM default "
                             "0.0: the official CosineAnnealingLR uses eta_min=0)")
    parser.add_argument("--max-batches-per-epoch", type=int,
                        default=config.PROBVLM_MAX_BATCHES_PER_EPOCH,
                        help="Cap on processed batches per epoch (official: 2000; "
                             "<=0 disables the cap)")

    # Adapter architecture (defaults from config; checkpoint meta overrides on load)
    parser.add_argument("--hidden-dim", type=int, default=config.PROBVLM_HIDDEN_DIM,
                        help="BayesCap adapter hidden dim")
    parser.add_argument("--num-layers", type=int, default=config.PROBVLM_NUM_LAYERS,
                        help="BayesCap adapter trunk depth (official: 3)")
    parser.add_argument("--dropout", type=float, default=config.PROBVLM_DROPOUT,
                        help="BayesCap adapter dropout rate")

    # ProbVLM loss weights
    parser.add_argument("--t1", type=float, default=config.PROBVLM_T1,
                        help="TempComb weight for the L1 term")
    parser.add_argument("--t2", type=float, default=config.PROBVLM_T2,
                        help="TempComb weight for the GenGauss NLL term")
    parser.add_argument("--cross-modal-lambda", type=float,
                        default=config.PROBVLM_CROSS_MODAL_LAMBDA,
                        help="Weight of the cross-modal reconstruction terms")

    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    # Validation and early stopping arguments
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation set ratio (default: 0.1)")
    parser.add_argument("--early-stop-patience", type=int,
                        default=config.PROBVLM_EARLY_STOP_PATIENCE,
                        help="Early stopping patience in epochs "
                             f"(default: {config.PROBVLM_EARLY_STOP_PATIENCE})")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping (reproduces the official "
                             "full 100-epoch schedule)")

    # Output arguments
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint directory (uses config default if None)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from "
                             "(e.g. checkpoints/probvlm_coco_last.pt). "
                             "Restores model weights, optimizer state, epoch, and best_val_loss.")

    return parser.parse_args()


def _forward_loss(
    model: ProbVLMModel,
    criterion: TempCombLoss,
    args: argparse.Namespace,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Run the model and compute the ProbVLM loss + the official val-MAE diagnostic."""
    outputs = model(pixel_values, input_ids, attention_mask)

    loss, info = probvlm_total_loss(
        outputs, criterion,
        t1=args.t1, t2=args.t2, cross_modal_lambda=args.cross_modal_lambda,
    )

    # Official selection metric (informational): mean |mu - frozen feature|
    with torch.no_grad():
        mae = 0.5 * (
            (outputs["img_mu"] - outputs["i_features"]).abs().mean()
            + (outputs["text_mu"] - outputs["t_features"]).abs().mean()
        )
    info["mae"] = mae.item()

    return loss, info


def train_epoch(
    model: ProbVLMModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: TempCombLoss,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train the adapters for one epoch (<= args.max_batches_per_epoch batches)."""
    model.train()
    # model.train() flips every submodule to train mode; the frozen backbone
    # must stay in eval mode. Dropout stays active in the adapters only.
    model.clip_model.eval()

    total_loss = 0.0
    total_mae = 0.0
    processed_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for batch_idx, batch in enumerate(pbar):
        if args.max_batches_per_epoch > 0 and batch_idx >= args.max_batches_per_epoch:
            break
        if batch is None:
            continue
        processed_batches += 1

        pil_images = batch["image"]
        captions_list = batch["captions"]

        # One random caption per image per step (uniform over the K=5 captions,
        # matching the official per-caption-item dataloader)
        selected_captions = [random.choice(captions) for captions in captions_list]

        pixel_values = model.process_images(pil_images)
        text_inputs = model.process_text(selected_captions)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        loss, loss_info = _forward_loss(
            model, criterion, args, pixel_values, input_ids, attention_mask)

        if not torch.isfinite(loss):
            logger.warning(
                f"Epoch {epoch + 1}, Batch {batch_idx + 1}: non-finite loss "
                f"({loss.item()}); skipping step"
            )
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss_info["loss"]
        total_mae += loss_info["mae"]

        pbar.set_postfix({
            "loss": f"{loss_info['loss']:.4f}",
            "mae": f"{loss_info['mae']:.4f}",
        })

        if (batch_idx + 1) % 10 == 0:
            logger.debug(
                f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss_info['loss']:.4f}, MAE: {loss_info['mae']:.4f}"
            )

    num_batches = max(processed_batches, 1)
    return {"loss": total_loss / num_batches, "mae": total_mae / num_batches}


@torch.no_grad()
def evaluate(
    model: ProbVLMModel,
    dataloader: DataLoader,
    criterion: TempCombLoss,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    processed_batches = 0

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        if batch is None:
            continue
        processed_batches += 1

        pil_images = batch["image"]
        captions_list = batch["captions"]

        selected_captions = [captions[0] for captions in captions_list]

        pixel_values = model.process_images(pil_images)
        text_inputs = model.process_text(selected_captions)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        _, loss_info = _forward_loss(
            model, criterion, args, pixel_values, input_ids, attention_mask)

        total_loss += loss_info["loss"]
        total_mae += loss_info["mae"]

        pbar.set_postfix({"loss": f"{loss_info['loss']:.4f}"})

    num_batches = max(processed_batches, 1)
    return {"loss": total_loss / num_batches, "mae": total_mae / num_batches}


def main():
    args = parse_args()

    # Direct script runs bypass main.py's ensure_project_dirs()
    config.ensure_project_dirs()

    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    logger.info("=" * 60)
    logger.info("ProbVLM (B4) Training")
    logger.info("=" * 60)
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"LR scheduler: {args.lr_scheduler} (warmup {args.warmup_epochs}, "
                f"min_lr_ratio {args.min_lr_ratio})")
    logger.info(f"Max batches per epoch: {args.max_batches_per_epoch}")
    logger.info(f"TempComb loss: t1={args.t1}, t2={args.t2}, "
                f"cross_modal_lambda={args.cross_modal_lambda}")
    logger.info(f"Adapter: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, "
                f"dropout={args.dropout}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    logger.info("Creating model...")
    model = ProbVLMModel(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = model.to(args.device)
    logger.info(f"Model created with {model.num_trainable_parameters():,} trainable parameters "
                f"(CLIP ViT-L/14 frozen)")

    criterion = TempCombLoss()

    optimizer = optim.Adam(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    # coco=MSCOCO, flickr=flickr30k train split
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

        train_metrics = train_epoch(model, train_dataloader, optimizer, criterion,
                                    args, args.device, epoch)

        val_metrics = evaluate(model, val_dataloader, criterion, args, args.device)

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val MAE: {val_metrics['mae']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_checkpoint_path = checkpoint_dir / f"probvlm_{args.dataset}_best.pt"
            model.save(str(best_checkpoint_path))
            logger.info(f"Best model saved (val_loss: {best_val_loss:.4f}) -> {best_checkpoint_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{args.early_stop_patience}")

        if not args.no_early_stop and patience_counter >= args.early_stop_patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Save final model with full training state for resumption (heads-only
    # model_state_dict + resume keys; model.load ignores the extra keys)
    final_checkpoint_path = checkpoint_dir / f"probvlm_{args.dataset}_last.pt"
    heads_sd = {k: v for k, v in model.state_dict().items()
                if not k.startswith("clip_model.")}
    final_state = {
        "model_state_dict": heads_sd,
        "embed_dim": model.embed_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
        "backbone_excluded": True,
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
        log_exception(logger, e, "ProbVLM training failed")
        raise
