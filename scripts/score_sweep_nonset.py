#!/usr/bin/env python
"""Path-1b: NON-set scoring variants only (no caption grouping used).

  V0  cos(img, cap)                                  baseline
  T1  V0 - eta * sigma_bar_text(cap)                 MCDisp-only: candidate's own uncertainty
  T2  V0 - eta * sigma_bar_img (row-const for I2T; moves T2I only)
  N1  raw dot product mu_v . mu_t                    norms as confidence
  W1  cosine on dev-whitened space (per-dim center+scale)   all models, fair
  W2  W1 - eta * sigma_bar_text                      MCDisp-only combo

Whitening statistics come from the DEV half only (no test leakage). Per-image
caption grouping is never read; every score depends only on (img, cap) pair.
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

logger = get_logger("score_sweep2", config.LOG_DIR / "score_sweep2.log")

ETAS = [0.05, 0.2, 0.5, 1.0]


def variants_nonset(f, kind, device):
    img_mu, cap_mus = f["img_mu"].to(device), f["text_mus"].to(device)
    N, K, D = cap_mus.shape
    cap_mu = cap_mus.reshape(N * K, D)
    img_n = F.normalize(img_mu, dim=-1)
    cap_n = F.normalize(cap_mu, dim=-1)
    C = img_n @ cap_n.T

    out = {"V0": C, "N1:dot": img_mu @ cap_mu.T}

    sig_t = sig_i = None
    if kind == "mcdisp":
        sig_t = torch.exp(f["text_logvars"].to(device)).mean(-1).reshape(N * K)
        sig_i = torch.exp(f["img_logvar"].to(device)).mean(-1)
        for e in ETAS:
            out[f"T1:eta={e}"] = C - e * sig_t.unsqueeze(0)
            out[f"T2:eta={e}"] = C - e * sig_i.unsqueeze(1)

    # dev-half whitening statistics (both modalities pooled, per-dim)
    dev_rows = torch.arange(0, N, 2, device=device)
    dev_cols = (dev_rows * K).repeat_interleave(K) + torch.arange(K, device=device).repeat(len(dev_rows))
    pool = torch.cat([img_mu[dev_rows], cap_mu[dev_cols]], dim=0)
    mu, sd = pool.mean(0, keepdim=True), pool.std(0, keepdim=True).clamp_min(1e-6)
    iw = (img_mu - mu) / sd
    cw = (cap_mu - mu) / sd
    Cw = F.normalize(iw, dim=-1) @ F.normalize(cw, dim=-1).T
    out["W1:whiten"] = Cw
    if sig_t is not None:
        for e in ETAS:
            out[f"W2:eta={e}"] = Cw - e * sig_t.unsqueeze(0)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    set_seed(config.SEED)

    jobs = [
        ("mcdisp_kl_unfrozen_s42", "mcdisp", "checkpoints/unfrozen_seed42/mcdisp_align_kl_coco_best.pt"),
        ("mcdisp_kl_frozen_s42", "mcdisp", "checkpoints/seed42/mcdisp_align_kl_coco_best.pt"),
        ("clip_ft_s42", "clip", "checkpoints/seed42/clip_baseline_coco_best.pt"),
    ]
    from models.mcdisp_align_model import MCDispAlignModel
    from models.clip_baseline import CLIPFineTuneBaseline

    loader, _ = build_eval_dataloader("coco", batch_size=args.batch_size,
                                      num_workers=4, num_samples=args.num_samples)
    results = {}
    for key, kind, ckpt in jobs:
        model = (MCDispAlignModel() if kind == "mcdisp" else CLIPFineTuneBaseline())
        model.load(ckpt)
        model = model.to(args.device).eval()
        f = extract(model, kind, loader, args.device, args.num_samples)
        del model
        torch.cuda.empty_cache()

        N, K = f["img_mu"].shape[0], f["text_mus"].shape[1]
        vs = variants_nonset(f, kind, args.device)
        dev_idx = torch.arange(0, N, 2, device=args.device)
        test_idx = torch.arange(1, N, 2, device=args.device)

        def _cols(img_idx):
            return (img_idx * K).repeat_interleave(K) + torch.arange(K, device=args.device).repeat(len(img_idx))

        report, best = {}, None
        for name, S in vs.items():          # compute recalls then FREE each matrix
            r_dev = recalls_from_scores(S[dev_idx][:, _cols(dev_idx)], len(dev_idx), K, args.device)
            r_test = recalls_from_scores(S[test_idx][:, _cols(test_idx)], len(test_idx), K, args.device)
            del S
            report[name] = {"dev_mr": r_dev["mr"], "test": r_test}
            if best is None or r_dev["mr"] > best[1]:
                best = (name, r_dev["mr"], r_test)
        del vs
        torch.cuda.empty_cache()
        results[key] = {"variants": report,
                        "dev_selected": {"name": best[0], "test": best[2]}}
        logger.info(f"{key}: dev-selected {best[0]} -> test mr={best[2]['mr']:.4f} "
                    f"(V0 {report['V0']['test']['mr']:.4f})")

    out = Path("outputs/score_sweep")
    (out / "results_nonset.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
