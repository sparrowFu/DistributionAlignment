#!/usr/bin/env python
"""Full-gallery evaluation of MCDisp-Align's own scoring vs baselines.

Protocol: unified multi-caption retrieval on the FULL eval split (N images vs
N*K captions; any-hit I2T / per-caption T2I; cosine unless stated). This is
the same protocol as outputs/eval_multiseed/summary.md Table 1.

Variants (per model, per dataset; seeds 42/43/44):
  V0        plain cosine per seed
  CSLS      2*C - r_img - r_cap with k=5 (hubness fix; chosen on the COCO/
            Flickr dev halves in score_sweep_objective.py)
  ENS       mean of the 3 per-seed cosine matrices (score-level seed ensemble)
  ENS+CSLS  CSLS applied to ENS

Models: mcdisp_kl_unfrozen (primary), clip fine-tuned and ProLIP fine-tuned
as controls (they get the same variants, so the CSLS/ensemble gains are
directly comparable).
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.eval_common import build_eval_dataloader
from utils.logger import get_logger
from utils.seed import set_seed
from scripts.score_variant_sweep import extract, recalls_from_scores

logger = get_logger("eval_full_obj", config.LOG_DIR / "eval_full_objective.log")

SEEDS = [42, 43, 44]
CSLS_K = 5


def cosine_matrix(img_mu, cap_mus):
    N, K, D = cap_mus.shape
    img_n = F.normalize(img_mu, dim=-1)
    cap_n = F.normalize(cap_mus.reshape(N * K, D), dim=-1)
    return img_n @ cap_n.T


def csls(C, k=CSLS_K):
    r_img = C.topk(k, dim=1).values.mean(1, keepdim=True)
    r_cap = C.topk(k, dim=0).values.mean(0, keepdim=True)
    return 2 * C - r_img - r_cap


def extract_prolip(model, loader, device, n):
    """ProLIP per-caption means: img_mu (N,D), text_mus (N,K,D)."""
    imgs, caps = [], []
    got = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            pil, cl = batch["image"], batch["captions"]
            B, K = len(pil), len(cl[0])
            flat = [c for l in cl for c in l]
            ti = model.process_text(flat)
            out = model(model.process_images(pil).to(device),
                        ti["input_ids"].to(device))
            imgs.append(out["img_mu"].float())
            caps.append(out["text_mu"].float().view(B, K, -1))
            got += B
            if got >= n:
                break
    return torch.cat(imgs).cpu(), torch.cat(caps).cpu()


def eval_matrices(mats, K, device, tag, results):
    """mats: dict variant -> full cosine/score matrix; recalls + log."""
    N = next(iter(mats.values())).shape[0]
    for name, S in mats.items():
        r = recalls_from_scores(S, N, K, device)
        results.setdefault(tag, {})[name] = r
        logger.info("%s | %-9s i2t %.4f/%.4f/%.4f  t2i %.4f/%.4f/%.4f",
                    tag, name, r["i2t@1"], r["i2t@5"], r["i2t@10"],
                    r["t2i@1"], r["t2i@5"], r["t2i@10"])
        del S
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["coco", "flickr"], default="coco")
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--models", nargs="+",
                   default=["mcdisp", "clip", "prolip"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    set_seed(config.SEED)

    from models.mcdisp_align_model import MCDispAlignModel
    from models.clip_baseline import CLIPFineTuneBaseline
    from models.prolip_model import ProLIPModel

    loader, n_avail = build_eval_dataloader(args.dataset, batch_size=args.batch_size,
                                            num_workers=4, num_samples=args.num_samples)
    n = min(args.num_samples, n_avail)
    results = {"dataset": args.dataset, "num_samples": n}

    for mdl in args.models:
        per_seed = {}
        for seed in SEEDS:
            if mdl == "mcdisp":
                ckpt = f"checkpoints/unfrozen_seed{seed}/mcdisp_align_kl_{args.dataset}_best.pt"
                model = MCDispAlignModel()
                model.load(ckpt)
                model = model.to(args.device).eval()
                f = extract(model, "mcdisp", loader, args.device, n)
                img_mu, cap_mus = f["img_mu"], f["text_mus"]
            elif mdl == "clip":
                ckpt = f"checkpoints/seed{seed}/clip_baseline_{args.dataset}_best.pt"
                model = CLIPFineTuneBaseline()
                model.load(ckpt)
                model = model.to(args.device).eval()
                f = extract(model, "clip", loader, args.device, n)
                img_mu, cap_mus = f["img_mu"], f["text_mus"]
            else:
                ckpt = f"checkpoints/seed{seed}/prolip_{args.dataset}_best.pt"
                model = ProLIPModel()
                model.load(ckpt)
                model = model.to(args.device).eval()
                img_mu, cap_mus = extract_prolip(model, loader, args.device, n)
            del model
            torch.cuda.empty_cache()
            per_seed[seed] = cosine_matrix(img_mu.to(args.device),
                                           cap_mus.to(args.device))
            del img_mu, cap_mus
            torch.cuda.empty_cache()

        # K from any stored matrix: M = N*K
        first = per_seed[SEEDS[0]]
        N = first.shape[0]
        K = first.shape[1] // N

        mats = {}
        for seed in SEEDS:
            mats[f"V0_s{seed}"] = per_seed[seed]
        ens = sum(per_seed.values()) / len(per_seed)
        mats["ENS"] = ens
        for seed in SEEDS:
            mats[f"CSLS_s{seed}"] = csls(per_seed[seed])
        mats["ENS+CSLS"] = csls(ens)
        eval_matrices(mats, K, args.device, f"{mdl}_{args.dataset}", results)

        del per_seed, mats, ens
        torch.cuda.empty_cache()

    out = Path("outputs/score_sweep")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"results_full_{args.dataset}.json").write_text(json.dumps(results, indent=2))
    logger.info("saved outputs/score_sweep/results_full_%s.json", args.dataset)


if __name__ == "__main__":
    main()
