#!/usr/bin/env python
"""Extract + cache test-split features for every trained MCDisp checkpoint.

Populates outputs/feat_cache/{key}_{dataset}.pt with the same dict format as
score_variant_sweep.extract (img_mu / text_mus / img_logvar / text_logvars /
img_U), for the fusion pool:
  - frozen KL  x seeds 42/43/44  (checkpoints/seed{S})
  - std        x seeds 42/43/44  (checkpoints/seed{S}, mcdisp_align_*)
  - capNCE     seed 42 unfrozen + frozen (COCO only)
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
from scripts.score_variant_sweep import extract

logger = get_logger("extract_all", config.LOG_DIR / "extract_all_mcdisp.log")


def jobs_for(dataset):
    jobs = []
    for s in (42, 43, 44):
        jobs.append((f"mcdisp_kl_unfrozen_s{s}_{dataset}",
                     f"checkpoints/unfrozen_seed{s}/mcdisp_align_kl_{dataset}_best.pt"))
    for s in (42, 43, 44):
        jobs.append((f"mcdisp_kl_frozen_s{s}_{dataset}",
                     f"checkpoints/seed{s}/mcdisp_align_kl_{dataset}_best.pt"))
        jobs.append((f"mcdisp_std_s{s}_{dataset}",
                     f"checkpoints/seed{s}/mcdisp_align_{dataset}_best.pt"))
    if dataset == "coco":
        jobs.append(("mcdisp_kl_capnce_s42_coco",
                     "checkpoints/capnce_seed42/mcdisp_align_kl_capnce_coco_best.pt"))
        jobs.append(("mcdisp_kl_capnce_frozen_s42_coco",
                     "checkpoints/capnce_frozen_seed42/mcdisp_align_kl_capnce_coco_best.pt"))
        jobs.append(("mcdisp_res_s42_coco",
                     "checkpoints/res_seed42/mcdisp_align_kl_capnce_coco_best.pt"))
        jobs.append(("mcdisp_res_s42_last_coco",
                     "checkpoints/res_seed42/mcdisp_align_kl_capnce_coco_last.pt"))
        jobs.append(("mcdisp_bal_s42_coco",
                     "checkpoints/bal_seed42/mcdisp_align_kl_coco_best.pt"))
        jobs.append(("mcdisp_bal_s42_last_coco",
                     "checkpoints/bal_seed42/mcdisp_align_kl_coco_last.pt"))
    return jobs


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

    for key, ckpt in jobs_for(args.dataset):
        out = cache / f"{key}.pt"
        if out.exists():
            logger.info("skip %s (cached)", key)
            continue
        model = MCDispAlignModel()
        model.load(ckpt)
        model = model.to(args.device).eval()
        f = extract(model, "mcdisp", loader, args.device, args.num_samples)
        torch.save(f, out)
        logger.info("cached %s from %s", key, ckpt)
        del model, f
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
