"""
GaussianImageDistribution - ProLIP (B3) Training Script

ProLIP uses the same 4-MLP-head architecture as dist_align, but sigma has
no explicit semantic constraint (no consistency loss). Trained with
contrastive loss + variance regularization.

Usage:
    python scripts/train_prolip.py
    python main.py --task train_prolip
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.caption_dataset import ImageCaptionDataset, filter_none_collate
from losses.dist_align_losses import CombinedDistributionLoss
from models.prolip_model import ProLIPModel
from utils.logger import get_logger, log_exception
from utils.lr_scheduler import apply_lr_for_epoch
from utils.seed import set_seed


logger = get_logger("train_prolip", config.TRAIN_PROLIP_LOG_PATH)

# Exclude faulty CPU cores (e.g. unstable CPU 2) before DataLoader workers and
# torch threads are created. Inherited by forked worker processes.
from utils.cpu_affinity import apply_cpu_affinity
apply_cpu_affinity()


def parse_args():
    parser = argparse.ArgumentParser(description="Train ProLIP Baseline (B3)")
    parser.add_argument("--epochs", type=int, default=config.DIST_ALIGN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DIST_ALIGN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.DIST_ALIGN_MLP_LR)
    parser.add_argument("--weight-decay", type=float, default=config.DIST_ALIGN_WEIGHT_DECAY)
    parser.add_argument("--lr-scheduler", type=str, default=config.LR_SCHEDULER,
                        choices=["none", "cosine"],
                        help="LR schedule: 'cosine' (cosine + linear warmup) or 'none' (constant LR)")
    parser.add_argument("--warmup-epochs", type=int, default=config.LR_WARMUP_EPOCHS,
                        help="Linear warmup epochs for the cosine schedule (0 disables warmup)")
    parser.add_argument("--min-lr-ratio", type=float, default=config.LR_MIN_LR_RATIO,
                        help="Cosine floor as a fraction of the base LR")
    parser.add_argument("--temperature", type=float, default=config.DIST_ALIGN_TEMPERATURE)
    parser.add_argument("--captions-path", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("ProLIP (B3) Training")
    logger.info("=" * 60)
    logger.info(f"LR scheduler: {args.lr_scheduler} (warmup {args.warmup_epochs}, "
                f"min_lr_ratio {args.min_lr_ratio})")

    # Dataset
    captions_path = args.captions_path or config.CAPTIONS_PATH
    images_dir = args.images_dir or config.IMAGES_DIR
    dataset = ImageCaptionDataset(
        captions_path=captions_path, images_dir=images_dir,
        num_captions=config.NUM_CAPTIONS,
    )

    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=config.NUM_WORKERS, collate_fn=filter_none_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=config.NUM_WORKERS, collate_fn=filter_none_collate)

    logger.info(f"Train: {train_size}, Val: {val_size}")

    # Model
    model = ProLIPModel(freeze_clip=True, dropout_rate=config.DIST_ALIGN_DROPOUT_RATE)
    model = model.to(args.device)

    trainable = model.trainable_parameters()
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    # Loss: contrastive + variance reg, NO consistency (lambda_consist=0)
    criterion = CombinedDistributionLoss(
        temperature=args.temperature,
        lambda_contrastive=config.DIST_ALIGN_LAMBDA_CONTRASTIVE,
        lambda_kl=0.0,  # ProLIP does not use consistency loss
        lambda_var=config.DIST_ALIGN_LAMBDA_VAR,
        target_variance=config.DIST_ALIGN_TARGET_VARIANCE,
        kl_type=config.DIST_ALIGN_KL_TYPE,
    )

    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    start_epoch = 0
    best_val_loss = float("inf")

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
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        base_lrs = ckpt.get("base_lrs", base_lrs)
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

    last_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch

        # Apply LR schedule for this epoch (no-op when scheduler == "none")
        apply_lr_for_epoch(optimizer, base_lrs, epoch, args.epochs,
                           args.warmup_epochs, args.min_lr_ratio,
                           args.lr_scheduler, logger)

        model.train()
        model.clip_model.eval()

        epoch_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            if batch is None:
                continue

            pil_images = batch["image"]
            caption_lists = batch["captions"]
            batch_size = len(pil_images)
            num_captions = len(caption_lists[0])

            pixel_values = model.process_images(pil_images).to(args.device)

            all_captions = []
            for cl in caption_lists:
                all_captions.extend(cl)
            text_inputs = model.process_text(all_captions)
            input_ids = text_inputs["input_ids"].view(batch_size, num_captions, -1).to(args.device)
            attention_mask = text_inputs["attention_mask"].view(batch_size, num_captions, -1).to(args.device)

            outputs = model(pixel_values, input_ids, attention_mask)

            loss, loss_dict = criterion(
                outputs['img_features'], outputs['text_features'],
                outputs['img_mu'], outputs['img_logvar'],
                outputs['text_mu'], outputs['text_logvar'],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}")

        # Validation
        val_loss = 0.0
        val_batches = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                pil_images = batch["image"]
                caption_lists = batch["captions"]
                bs = len(pil_images)
                nc = len(caption_lists[0])
                pv = model.process_images(pil_images).to(args.device)
                caps = []
                for cl in caption_lists:
                    caps.extend(cl)
                ti = model.process_text(caps)
                ids = ti["input_ids"].view(bs, nc, -1).to(args.device)
                mask = ti["attention_mask"].view(bs, nc, -1).to(args.device)
                out = model(pv, ids, mask)
                l, _ = criterion(
                    out['img_features'], out['text_features'],
                    out['img_mu'], out['img_logvar'],
                    out['text_mu'], out['text_logvar'],
                )
                val_loss += l.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        logger.info(f"Epoch {epoch+1}: val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(str(config.PROLIP_BEST_CKPT))
            logger.info(f"Best model saved (val_loss: {best_val_loss:.4f})")

    # Save last checkpoint with full training state for resumption
    last_ckpt_path = str(config.PROLIP_BEST_CKPT).replace("_best.pt", "_last.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": last_epoch + 1,
        "best_val_loss": best_val_loss,
        "base_lrs": base_lrs,
    }, last_ckpt_path)
    logger.info(f"Training completed. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "ProLIP training failed")
        raise
