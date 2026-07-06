"""
GaussianImageDistribution - GroVE (B4) Training Script

GroVE adds a GP posterior on top of frozen CLIP features. Training learns
the inducing points via contrastive loss on posterior mu.

Usage:
    python scripts/train_grove.py
    python main.py --task train_grove
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.caption_dataset import ImageCaptionDataset, filter_none_collate
from losses.clip_losses import clip_contrastive_loss
from models.grove_model import GroVEModel
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


logger = get_logger("train_grove", config.TRAIN_GROVE_LOG_PATH)

# Exclude faulty CPU cores (e.g. unstable CPU 2) before DataLoader workers and
# torch threads are created. Inherited by forked worker processes.
from utils.cpu_affinity import apply_cpu_affinity
apply_cpu_affinity()


def parse_args():
    parser = argparse.ArgumentParser(description="Train GroVE Baseline (B4)")
    parser.add_argument("--epochs", type=int, default=config.GROVE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.GROVE_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.GROVE_LR)
    parser.add_argument("--weight-decay", type=float, default=config.GROVE_WEIGHT_DECAY)
    parser.add_argument("--temperature", type=float, default=config.GROVE_TEMPERATURE)
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
    logger.info("GroVE (B4) Training")
    logger.info("=" * 60)

    # Dataset
    captions_path = args.captions_path or config.CAPTIONS_PATH
    images_dir = args.images_dir or config.IMAGES_DIR
    dataset = ImageCaptionDataset(
        captions_path=captions_path, images_dir=images_dir,
        num_captions=config.NUM_CAPTIONS,
    )

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
    model = GroVEModel(num_inducing=config.GROVE_NUM_INDUCING, freeze_clip=True)
    model = model.to(args.device)

    trainable = model.trainable_parameters()
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

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
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

    last_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
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

            # Contrastive loss on posterior mu
            img_mu = outputs['img_mu']
            text_mu = outputs['text_mu']
            loss = clip_contrastive_loss(img_mu, text_mu, temperature=args.temperature)

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
                l = clip_contrastive_loss(out['img_mu'], out['text_mu'], temperature=args.temperature)
                val_loss += l.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        logger.info(f"Epoch {epoch+1}: val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(str(config.GROVE_BEST_CKPT))
            logger.info(f"Best model saved (val_loss: {best_val_loss:.4f})")

    last_ckpt_path = str(config.GROVE_BEST_CKPT).replace("_best.pt", "_last.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": last_epoch + 1,
        "best_val_loss": best_val_loss,
    }, last_ckpt_path)
    logger.info(f"Training completed. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "GroVE training failed")
        raise
