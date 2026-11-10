"""
PCME++ Training Script (B5)

Trains PCME++ (Chun, ICLR 2024) from OpenAI CLIP ViT-L/14 initialization on
the project's MSCOCO pool, replicating the official configs/pcmepp.yaml +
README ViT-L/14 recipe on a single GPU: caption-level sampling (5 captions
per image per epoch), batch 128 (== the official 8 GPUs x 16), AdamP with
layerwise LR decay + backbone LR multipliers, image-backbone freeze for the
first 2 epochs, loss-multiplied intra-epoch warmup at epoch 2, lr x0.1 from
epoch 15 (official recipe; see the epochs deviation below), grad clip 2.0,
MSDA (mixup/cutmix, elem mode) and the token-level SizeAugment /
caption-token perturbation.

Deviations from the official setup (documented in experiments.md):
- precision: official fp32 ("32-true", the default since 2026-11-10). The
  earlier bf16-autocast default was reverted: in the officially unbounded
  sigma branch, occasional mid-training runaways (a structural property of
  the recipe, see experiments.md section 10) produced non-finite gradients
  under bf16 backward, while official fp32 survives them via grad clipping.
  To fit fp32 batch 128 on one 80GB GPU the two CLIP towers run under
  torch.utils.checkpoint (numerically identical; pure-torch modules, no RNG
  inside, exact fp32 recompute) -- --no-grad-checkpoint disables it.
- no torch.compile: the official engine calls torch.compile and discards
  the result, so its runs were effectively uncompiled too.
- no SWA (official swa_start_epoch 16; not part of the paper's main-table
  protocol).
- epochs: 5 (official 25); the epoch-15 lr milestone is therefore never
  reached.
- non-finite loss handling: official prints 'Killed due to NaN!!' and
  exits; we log and raise (same outcome, visible in main.py).
- data: the project's MSCOCO train2017 pool (118,287 images) instead of the
  official COCO-2014 Karpathy split; PIL bilinear instead of cv2.INTER_LINEAR;
  validation uses caption[0] per image.
"""

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F  # noqa: F401 (kept for parity with the other train scripts)
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.pcmepp_dataset import PCMEPPCaptionView, pcmepp_collate, pcmepp_worker_init_fn
from losses.pcmepp_losses import (
    build_matched_matrix,
    build_param_groups,
    label_smooth_matched,
    milestone_lr_factor,
    warmup_alpha,
)
from models.pcmepp_model import PCMEPPModel
from pcmepp.adamp import AdamP
from pcmepp.augmentation import Mixup
from utils.dataset_factory import build_train_dataset, VALID_DATASETS
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


logger = get_logger("train_pcmepp", config.TRAIN_PCMEPP_LOG_PATH)

# Exclude faulty CPU cores (e.g. unstable CPU 2) before DataLoader workers and
# torch threads are created. Inherited by forked worker processes.
from utils.cpu_affinity import apply_cpu_affinity
apply_cpu_affinity()


def parse_args():
    parser = argparse.ArgumentParser(description="Train PCME++ Baseline (B5)")

    parser.add_argument("--captions-path", type=str, default=None,
                        help="Path to captions parquet file (uses config default if None)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Path to images directory (uses config default if None)")
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=list(VALID_DATASETS),
                        help="Training dataset: selects both the training data and the "
                             "checkpoint-name tag (coco=MSCOCO, flickr=flickr30k)")

    parser.add_argument("--epochs", type=int, default=config.PCMEPP_EPOCHS,
                        help="Number of training epochs (official: 25)")
    parser.add_argument("--batch-size", type=int, default=config.PCMEPP_BATCH_SIZE,
                        help="Training batch size (official effective batch 128; MSDA "
                             "requires an even size)")
    parser.add_argument("--lr", type=float, default=config.PCMEPP_LR,
                        help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=config.PCMEPP_WEIGHT_DECAY,
                        help="Weight decay")
    parser.add_argument("--precision", type=str, default=config.PCMEPP_PRECISION,
                        choices=["bf16-mixed", "32-true"],
                        help="32-true: official full fp32 (default); bf16-mixed: "
                             "autocast encoders (numerically riskier in the sigma "
                             "branch -- see module docstring)")
    parser.add_argument("--no-grad-checkpoint", action="store_true",
                        help="Disable torch.utils.checkpoint over the CLIP towers "
                             "(enabled by default; numerically identical, halves the "
                             "activation memory so fp32 batch 128 fits one 80GB GPU)")
    parser.add_argument("--max-batches-per-epoch", type=int, default=0,
                        help="Cap on processed batches per epoch (<=0: no cap; "
                             "smoke-run convenience)")

    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (the vendored CLIP loader requires cuda)")

    # Validation split (project convention; official uses its own dev split)
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation set ratio (default: 0.1)")

    # Output arguments
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint directory (uses config default if None)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from "
                             "(e.g. checkpoints/pcmepp_coco_last.pt). "
                             "Restores model weights, optimizer state, epoch, and best_val_loss.")

    return parser.parse_args()


def _forward_embeddings(model: PCMEPPModel, images: torch.Tensor,
                        input_ids: torch.Tensor, lengths: torch.Tensor,
                        use_autocast: bool):
    """Encode a batch; under bf16 autocast for the towers, fp32 outputs for
    the criterion (distances / BCE are fp32; GPO's GRU already runs fp32 on
    its self-built positional encodings)."""
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if use_autocast else torch.autocast("cuda", enabled=False))
    with ctx:
        img_emb = model.img_enc(images)
        cap_emb = model.txt_enc(input_ids, lengths)
    if use_autocast:
        img_emb = {k: v.float() for k, v in img_emb.items()}
        cap_emb = {k: v.float() for k, v in cap_emb.items()}
    return img_emb, cap_emb


def train_epoch(model: PCMEPPModel, dataloader: DataLoader, optimizer,
                mixup_fn, args, device: torch.device, epoch: int) -> Dict[str, float]:
    """One official training epoch (engine.py training_step, single-GPU path)."""
    model.train()
    # official freeze schedule (uncertainty transformer stays trainable)
    model.apply_epoch_freeze(epoch)

    use_autocast = args.precision == "bf16-mixed"
    num_batches = len(dataloader)

    total_loss = 0.0
    processed = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for batch_idx, batch in enumerate(pbar):
        if args.max_batches_per_epoch > 0 and batch_idx >= args.max_batches_per_epoch:
            break
        if batch is None:
            continue
        processed += 1

        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        lengths = torch.Tensor(batch["lengths"]).to(device)
        img_ids = batch["img_ids"]

        # --- matched matrix + MSDA (engine.py forward_step) ---
        matched = build_matched_matrix(img_ids, device)
        if mixup_fn is not None:
            batch_size = len(images)
            if batch_size % 2 != 0:
                # NOTE: mixup only allows n % 2 == 0
                images = images[:batch_size - 1]
                input_ids = input_ids[:batch_size - 1]
                lengths = lengths[:batch_size - 1]
                batch_size = len(images)
            images, matched, _ = mixup_fn(
                images, torch.arange(batch_size).to(images.device))
        else:
            matched = label_smooth_matched(matched, config.PCMEPP_LABEL_SMOOTHING)

        img_emb, cap_emb = _forward_embeddings(
            model, images, input_ids, lengths, use_autocast)
        loss, loss_dict = model.criterion(img_emb, cap_emb, matched=matched)

        # official loss-multiplied warmup (epoch 2 intra-epoch ramp)
        alpha = warmup_alpha(epoch, batch_idx, num_batches,
                             embedding_warmup_epochs=config.PCMEPP_EMBEDDING_WARMUP_EPOCHS)
        loss = loss * alpha

        # Official engine.py: `if torch.isnan(loss) or torch.isinf(loss):
        # print('Killed due to NaN!!'); exit(0)`. Finite-but-huge losses
        # (sigma-branch runaways are a structural property of the recipe)
        # are STEPPED: grad clip 2.0 bounds the update and pulls sigma back,
        # which is how the official fp32 recipe self-heals them.
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Epoch {epoch + 1}, batch {batch_idx + 1}: non-finite loss "
                f"({loss.item()}) -- terminating, as the official engine does "
                f"('Killed due to NaN')."
            )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.PCMEPP_GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        if (batch_idx + 1) % 50 == 0:
            logger.debug(f"Epoch {epoch + 1}, batch {batch_idx + 1}/{num_batches}, "
                         f"loss: {loss.item():.4f}, mu_pdist: "
                         f"{float(loss_dict.get('loss/mu_pdist', 0)):.4f}")

    n = max(processed, 1)
    return {"loss": total_loss / n}


def _save_last_checkpoint(checkpoint_dir: Path, model: PCMEPPModel, optimizer,
                          args, epoch: int, best_val_loss: float,
                          base_group_lrs) -> None:
    """Full training state for --resume (model.load ignores the extras);
    overwritten after every epoch so a crash only costs the current epoch."""
    last_state = {
        "model_state_dict": model.state_dict(),
        "embed_size": model.embed_size,
        "img_dim": model.img_dim,
        "n_unc_layers": model.n_unc_layers,
        "clip_model_name": model.clip_model_name,
        "backbone_excluded": False,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch + 1,
        "best_val_loss": best_val_loss,
        "base_group_lrs": base_group_lrs,
    }
    torch.save(last_state, str(checkpoint_dir / f"pcmepp_{args.dataset}_last.pt"))


@torch.no_grad()
def evaluate(model: PCMEPPModel, dataloader: DataLoader, args,
             device: torch.device) -> Dict[str, float]:
    """Official validation: label-smoothed matched matrix (no MSDA), first
    caption per image, deterministic eval transform."""
    model.eval()
    use_autocast = args.precision == "bf16-mixed"

    total_loss = 0.0
    processed = 0
    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        if batch is None:
            continue
        processed += 1

        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        lengths = torch.Tensor(batch["lengths"]).to(device)

        matched = build_matched_matrix(batch["img_ids"], device)
        matched = label_smooth_matched(matched, config.PCMEPP_LABEL_SMOOTHING)

        img_emb, cap_emb = _forward_embeddings(
            model, images, input_ids, lengths, use_autocast)
        loss, _ = model.criterion(img_emb, cap_emb, matched=matched)

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    n = max(processed, 1)
    return {"loss": total_loss / n}


def main():
    args = parse_args()
    if args.batch_size % 2 != 0:
        raise ValueError(f"batch size must be even for MSDA, got {args.batch_size}")

    config.ensure_project_dirs()
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("PCME++ (B5) Training")
    logger.info("=" * 60)
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Base LR: {args.lr}, weight decay: {args.weight_decay}")
    logger.info(f"Layerwise LR decay: {config.PCMEPP_LAYERWISE_LR_DECAY} "
                f"(img backbone x{config.PCMEPP_IMG_BACKBONE_LR_DECAY}, "
                f"txt backbone x{config.PCMEPP_TXT_BACKBONE_LR_DECAY})")
    logger.info(f"LR milestone: x0.1 from epoch {config.PCMEPP_LR_MILESTONE_EPOCH}")
    logger.info(f"Img backbone freeze: first {config.PCMEPP_IMG_BACKBONE_FREEZE_EPOCHS} "
                f"epochs; warmup ramp at epoch {config.PCMEPP_EMBEDDING_WARMUP_EPOCHS}")
    logger.info(f"Precision: {args.precision}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    logger.info("Creating model (requires the converted OpenAI-format CLIP weights)...")
    model = PCMEPPModel().to(args.device)
    # official trainer uses benchmark=True (fixed 224x224 inputs); set_seed
    # disables it, re-enable after seeding
    torch.backends.cudnn.benchmark = True
    model.apply_epoch_freeze(0)
    if not args.no_grad_checkpoint:
        model.enable_grad_checkpoint()
    logger.info(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters, "
                f"{model.num_trainable_parameters():,} trainable under the epoch-0 freeze "
                f"(img backbone frozen, unc branch + text tower + heads trainable)")

    optimizer = AdamP(
        build_param_groups(
            model.named_parameters(),
            lr=args.lr,
            layerwise_lr_decay=config.PCMEPP_LAYERWISE_LR_DECAY,
            img_backbone_lr_decay=config.PCMEPP_IMG_BACKBONE_LR_DECAY,
            txt_backbone_lr_decay=config.PCMEPP_TXT_BACKBONE_LR_DECAY,
            n_unc_layers=config.PCMEPP_N_UNC_LAYERS,
            total_params=len(list(model.parameters())),
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    base_group_lrs = [g["lr"] for g in optimizer.param_groups]

    # coco=MSCOCO; image-level 90/10 split identical to the other baselines,
    # then wrapped by the caption-level view (5 items per image).
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

    train_loader = DataLoader(
        PCMEPPCaptionView(train_dataset, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pcmepp_collate,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=pcmepp_worker_init_fn,
    )
    val_loader = DataLoader(
        PCMEPPCaptionView(val_dataset, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pcmepp_collate,
        pin_memory=True,
    )
    logger.info(f"Train images: {train_size} (x5 = {train_size * 5} caption items), "
                f"Val images: {val_size}")
    logger.info(f"Batch size: {args.batch_size}, "
                f"train batches per epoch: {len(train_loader)}")

    start_epoch = 0
    best_val_loss = float("inf")
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
        base_group_lrs = ckpt.get("base_group_lrs", base_group_lrs)
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

    logger.info(f"Starting training from epoch {start_epoch + 1}...")
    for epoch in range(start_epoch, args.epochs):

        # per-epoch LR milestone (scale every group from its base LR)
        factor = milestone_lr_factor(epoch, config.PCMEPP_LR_MILESTONE_EPOCH)
        for group, base_lr in zip(optimizer.param_groups, base_group_lrs):
            group["lr"] = base_lr * factor
        logger.info(f"Epoch {epoch + 1}: LR factor {factor}, "
                    f"img-backbone-freeze={'on' if epoch < config.PCMEPP_IMG_BACKBONE_FREEZE_EPOCHS else 'off'}")

        # MSDA is rebuilt every epoch (official on_train_epoch_start)
        mixup_fn = Mixup(
            mixup_alpha=config.PCMEPP_MIXUP,
            cutmix_alpha=config.PCMEPP_CUTMIX,
            cutmix_minmax=None,
            prob=config.PCMEPP_MIXUP_PROB,
            switch_prob=config.PCMEPP_MIXUP_SWITCH_PROB,
            mode=config.PCMEPP_MIXUP_MODE,
            label_smoothing=config.PCMEPP_MIXUP_SMOOTHING,
            mix_batch_ratio=config.PCMEPP_MIX_BATCH_RATIO,
        )

        train_metrics = train_epoch(model, train_loader, optimizer, mixup_fn,
                                    args, torch.device(args.device), epoch)
        val_metrics = evaluate(model, val_loader, args, torch.device(args.device))

        peak_mem = (torch.cuda.max_memory_allocated() / 2**30
                    if torch.cuda.is_available() else 0.0)
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | peak mem: {peak_mem:.1f} GiB"
        )
        torch.cuda.reset_peak_memory_stats()

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = checkpoint_dir / f"pcmepp_{args.dataset}_best.pt"
            model.save(str(best_path))
            logger.info(f"Best model saved (val_loss: {best_val_loss:.4f}) -> {best_path}")

        # crash-safe resume point: overwrite every epoch (the previous run
        # crashed at epoch 10 and lost everything because last.pt was only
        # written after all epochs)
        _save_last_checkpoint(checkpoint_dir, model, optimizer, args,
                              epoch, best_val_loss, base_group_lrs)
        logger.info(f"Last checkpoint saved (epoch {epoch + 1}) -> "
                    f"{checkpoint_dir / f'pcmepp_{args.dataset}_last.pt'}")

    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("Training completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "PCME++ training failed")
        raise
