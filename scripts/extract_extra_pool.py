#!/usr/bin/env python
"""Extra fusion-pool features: last-checkpoints + flip-TTA + backbone readout.

Adds to outputs/feat_cache:
  - {tag}_last_{dataset}.pt   for unfrozen_kl / frozen_kl / std (seeds 42-44)
  - {tag}_flip_{dataset}.pt   unfrozen_kl seeds (horizontal-flip image TTA)
  - {tag}_bb_{dataset}.pt     unfrozen_kl seeds: RAW backbone-projected
                              features (pre-head readout of the fine-tuned
                              encoders) -- extra diversity scorers
"""

import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.eval_common import build_eval_dataloader
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger("extract_extra", config.LOG_DIR / "extract_extra_pool.log")


@torch.no_grad()
def extract_variant(model, loader, device, n, flip=False, backbone=False):
    """mcdisp features; flip: horizontal-flip image TTA; backbone: raw proj."""
    model.eval()
    imgs, caps = [], []
    got = 0
    for batch in loader:
        if batch is None:
            continue
        pil, cl = batch["image"], batch["captions"]
        B, K = len(pil), len(cl[0])
        flat = [c for l in cl for c in l]
        ti = model.process_text(flat)
        pv = model.process_images(pil).to(device)
        if flip:
            pv = pv.flip(-1)
        if backbone:
            # raw backbone readout: per-caption text features require a
            # direct CLIP call (forward()'s text_features is K-averaged)
            imgs.append(model.clip_model.get_image_features(pv).pooler_output.float())
            tf = model.clip_model.get_text_features(
                input_ids=ti["input_ids"].to(device).view(B * K, -1),
                attention_mask=ti["attention_mask"].to(device).view(B * K, -1),
            ).pooler_output.float()
            caps.append(tf.view(B, K, -1))
        else:
            out = model(pv,
                        ti["input_ids"].to(device).view(B, K, -1),
                        ti["attention_mask"].to(device).view(B, K, -1))
            imgs.append(out["img_mu"].float())
            caps.append(out["text_mus"].float())
        got += B
        if got >= n:
            break
    return {"img_mu": torch.cat(imgs).cpu(), "text_mus": torch.cat(caps).cpu()}


def jobs(dataset):
    out = []
    for s in (42, 43, 44):
        out.append((f"mcdisp_kl_unfrozen_s{s}_last_{dataset}",
                    f"checkpoints/unfrozen_seed{s}/mcdisp_align_kl_{dataset}_last.pt",
                    {}))
        out.append((f"mcdisp_kl_frozen_s{s}_last_{dataset}",
                    f"checkpoints/seed{s}/mcdisp_align_kl_{dataset}_last.pt",
                    {}))
        out.append((f"mcdisp_std_s{s}_last_{dataset}",
                    f"checkpoints/seed{s}/mcdisp_align_{dataset}_last.pt",
                    {}))
        out.append((f"mcdisp_kl_unfrozen_s{s}_flip_{dataset}",
                    f"checkpoints/unfrozen_seed{s}/mcdisp_align_kl_{dataset}_best.pt",
                    {"flip": True}))
        out.append((f"mcdisp_kl_unfrozen_s{s}_bb_{dataset}",
                    f"checkpoints/unfrozen_seed{s}/mcdisp_align_kl_{dataset}_best.pt",
                    {"backbone": True}))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["coco", "flickr"], default="coco")
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    set_seed(config.SEED)

    from models.mcdisp_align_model import MCDispAlignModel
    loader, _ = build_eval_dataloader(args.dataset, batch_size=args.batch_size,
                                      num_workers=4, num_samples=args.num_samples)
    cache = Path("outputs/feat_cache")
    cache.mkdir(parents=True, exist_ok=True)

    for key, ckpt, kw in jobs(args.dataset):
        out = cache / f"{key}.pt"
        if out.exists():
            logger.info("skip %s (cached)", key)
            continue
        model = MCDispAlignModel()
        model.load(ckpt)
        model = model.to(args.device).eval()
        f = extract_variant(model, loader, args.device, args.num_samples, **kw)
        torch.save(f, out)
        logger.info("cached %s from %s %s", key, ckpt, kw)
        del model, f
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
