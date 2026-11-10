"""Round 2: scan WITH the full MSDA mixup/cutmix chain (as in train_epoch) to
find a batch that pushes the epoch-2 model's img logsigma into the runaway
tail. Stops early once found and runs grad forensics bf16 vs fp32 on it.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.pcmepp_dataset import PCMEPPCaptionView, pcmepp_collate
from losses.pcmepp_losses import build_matched_matrix, label_smooth_matched
from models.pcmepp_model import PCMEPPModel
from pcmepp.augmentation import Mixup
from utils.dataset_factory import build_train_dataset
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger("repro_explosion", config.LOG_DIR / "repro_pcmepp_explosion.log")
DEVICE = torch.device("cuda")
CKPT = config.CHECKPOINT_DIR / "pcmepp_coco_last_epoch2_clean.pt"


def encode(model, images, input_ids, lengths, use_bf16):
    ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)
    with ctx:
        img_emb = model.img_enc(images)
        cap_emb = model.txt_enc(input_ids, lengths)
    return ({k: v.float() for k, v in img_emb.items()},
            {k: v.float() for k, v in cap_emb.items()})


def main():
    set_seed(config.SEED)
    model = PCMEPPModel()
    model.load(str(CKPT))
    model.to(DEVICE).train()
    model.apply_epoch_freeze(2)

    mixup_fn = Mixup(
        mixup_alpha=config.PCMEPP_MIXUP, cutmix_alpha=config.PCMEPP_CUTMIX,
        cutmix_minmax=None, prob=config.PCMEPP_MIXUP_PROB,
        switch_prob=config.PCMEPP_MIXUP_SWITCH_PROB, mode=config.PCMEPP_MIXUP_MODE,
        label_smoothing=config.PCMEPP_MIXUP_SMOOTHING,
        mix_batch_ratio=config.PCMEPP_MIX_BATCH_RATIO)

    full = build_train_dataset(dataset="coco", captions_path=None, images_dir=None)
    val_size = int(len(full) * 0.1)
    train, _ = torch.utils.data.random_split(
        full, [len(full) - val_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED))
    loader = torch.utils.data.DataLoader(
        PCMEPPCaptionView(train, train=True), batch_size=128, shuffle=True,
        num_workers=8, collate_fn=pcmepp_collate, drop_last=True)

    logger.info("scanning WITH mixup for sigma runaway...")
    hits = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 200 or len(hits) >= 3:
                break
            images = batch["images"].to(DEVICE, non_blocking=True)
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            lengths = torch.Tensor(batch["lengths"]).to(DEVICE)
            matched = build_matched_matrix(batch["img_ids"], DEVICE)
            images, matched, _ = mixup_fn(images, torch.arange(len(images)).to(images.device))
            img_emb, _ = encode(model, images, input_ids, lengths, use_bf16=True)
            mx = img_emb["std"].max().item()
            if i % 20 == 0:
                logger.info(f"  batch {i}: img_std_max={mx:.3f}")
            if mx > 0:
                logger.info(f"  !! batch {i}: img_std_max={mx:.3f} <-- RUNAWAY")
                hits.append((i, mx, batch, images, matched))
    if not hits:
        logger.info("no runaway found in 200 mixup batches from the epoch-2 weights")
        return

    # grad forensics on the first hit
    i, mx, batch, images, matched = hits[0]
    matched = label_smooth_matched(matched, config.PCMEPP_LABEL_SMOOTHING)
    for tag, use_bf16 in (("bf16-mixed", True), ("32-true", False)):
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        img_emb, cap_emb = encode(model, images, batch["input_ids"].to(DEVICE),
                                  torch.Tensor(batch["lengths"]).to(DEVICE), use_bf16)
        loss, _ = model.criterion(img_emb, cap_emb, matched=matched)
        logger.info(f"[{tag}] loss={loss.item():.6e} img_std_max="
                    f"{img_emb['std'].max().item():.3f}")
        loss.backward()
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not p.grad.isfinite().all()]
        total = sum(p.grad.norm() for _, p in model.named_parameters()
                    if p.grad is not None)
        logger.info(f"[{tag}] grad_norm={total.item():.4e} non_finite={len(bad)} {bad[:8]}")
    logger.info("done")


if __name__ == "__main__":
    main()
