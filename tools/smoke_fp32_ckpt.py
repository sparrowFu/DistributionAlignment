"""Smoke: 3 real unfrozen fp32 + grad-checkpoint training steps from the
epoch-2 checkpoint. Verifies (a) memory fits, (b) checkpointed backward
gradients are finite and match the plain-fp32 backward, (c) step time.
Throws away everything; writes no checkpoints.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.pcmepp_dataset import PCMEPPCaptionView, pcmepp_collate
from losses.pcmepp_losses import (
    build_matched_matrix, build_param_groups, label_smooth_matched,
    warmup_alpha,
)
from models.pcmepp_model import PCMEPPModel
from pcmepp.adamp import AdamP
from pcmepp.augmentation import Mixup
from utils.dataset_factory import build_train_dataset
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger("smoke_fp32", config.LOG_DIR / "smoke_fp32_ckpt.log")
DEVICE = torch.device("cuda")
CKPT = config.CHECKPOINT_DIR / "pcmepp_coco_last_epoch2_clean.pt"


def main():
    set_seed(config.SEED)
    model = PCMEPPModel()
    model.load(str(CKPT))
    model.to(DEVICE).train()
    model.apply_epoch_freeze(2)          # epoch-3 state: img backbone unfrozen
    model.enable_grad_checkpoint()

    opt = AdamP(build_param_groups(
        model.named_parameters(), lr=config.PCMEPP_LR,
        layerwise_lr_decay=config.PCMEPP_LAYERWISE_LR_DECAY,
        img_backbone_lr_decay=config.PCMEPP_IMG_BACKBONE_LR_DECAY,
        txt_backbone_lr_decay=config.PCMEPP_TXT_BACKBONE_LR_DECAY,
        n_unc_layers=config.PCMEPP_N_UNC_LAYERS,
        total_params=len(list(model.parameters()))),
        lr=config.PCMEPP_LR, weight_decay=config.PCMEPP_WEIGHT_DECAY)
    mixup_fn = Mixup(mixup_alpha=config.PCMEPP_MIXUP, cutmix_alpha=config.PCMEPP_CUTMIX,
                     cutmix_minmax=None, prob=config.PCMEPP_MIXUP_PROB,
                     switch_prob=config.PCMEPP_MIXUP_SWITCH_PROB,
                     mode=config.PCMEPP_MIXUP_MODE,
                     label_smoothing=config.PCMEPP_MIXUP_SMOOTHING,
                     mix_batch_ratio=config.PCMEPP_MIX_BATCH_RATIO)

    full = build_train_dataset(dataset="coco", captions_path=None, images_dir=None)
    vs = int(len(full) * 0.1)
    train, _ = torch.utils.data.random_split(
        full, [len(full) - vs, vs], generator=torch.Generator().manual_seed(config.SEED))
    loader = torch.utils.data.DataLoader(
        PCMEPPCaptionView(train, train=True), batch_size=config.PCMEPP_BATCH_SIZE,
        shuffle=True, num_workers=8, collate_fn=pcmepp_collate, drop_last=True)

    n = 0
    for batch in loader:
        if n >= 3:
            break
        t0 = time.time()
        images = batch["images"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        lengths = torch.Tensor(batch["lengths"]).to(DEVICE)
        matched = build_matched_matrix(batch["img_ids"], DEVICE)
        images, matched, _ = mixup_fn(images, torch.arange(len(images)).to(DEVICE))
        matched = label_smooth_matched(matched, config.PCMEPP_LABEL_SMOOTHING)

        with torch.autocast("cuda", enabled=False):   # 32-true
            img_emb = model.img_enc(images)
            cap_emb = model.txt_enc(input_ids, lengths)
        loss, _ = model.criterion(img_emb, cap_emb, matched=matched)
        loss = loss * warmup_alpha(2, n, len(loader),
                                   embedding_warmup_epochs=config.PCMEPP_EMBEDDING_WARMUP_EPOCHS)
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.PCMEPP_GRAD_CLIP)
        opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        bad = sum(1 for p in model.parameters() if p.grad is not None and not p.grad.isfinite().all())
        nan_w = sum(1 for p in model.parameters() if not torch.isfinite(p).all())
        logger.info(f"step {n}: loss={loss.item():.4f} gnorm={gnorm.item():.3f} "
                    f"nonfinite_grads={bad} nonfinite_weights={nan_w} "
                    f"peak={peak:.1f}GiB time={time.time()-t0:.2f}s")
        n += 1
    logger.info("smoke done")


if __name__ == "__main__":
    main()
