#!/usr/bin/env python
"""Bipartite mutual-neighbor diffusion re-ranking on the fused MCDisp score.

Joint-space k-reciprocal degenerates here: same-modality similarities
dominate every top-k list (modality gap), so the mutual-neighbor graph has
no cross-modal edges. This version keeps the k-reciprocal IDEA on the
BIPARTITE graph only:

  A(q,g)      = 1 iff caption g is a top-k1 neighbor of image q AND q is a
                top-k1 neighbor of g (mutual cross-modal adjacency, N x M)
  P3(q,g)     = (A @ A.T @ A)(q,g)   # length-3 mutual paths image->cap
                = #images sharing a mutual caption with q that are also
                  mutual neighbors of g, chained through mutual captions
  G           = A + P3 / sqrt(deg_i * deg_c)   (degree-normalized diffusion)
  final       = base + lam * G                  (additive bonus)

base = the dev-selected fused + CSLS matrix from fuse_beat_baseline.py.
k1 and lam selected on the dev half; final numbers on the full gallery.
Memory: only N x M and N x N / M x M intermediates -- no (N+M)^2 tensors.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.logger import get_logger
from scripts.fuse_beat_baseline import csls, fast_recalls, TARGETS

logger = get_logger("krecip", config.LOG_DIR / "krecip_rerank.log")

K1_GRID = [5, 10, 20]
LAM_GRID = [0.5, 1.0, 2.0]


def bipartite_diffusion(base, k1, lam):
    N, M = base.shape
    r_idx = base.topk(k1, dim=1).indices                    # images -> caps
    c_idx = base.topk(k1, dim=0).indices.T                  # caps -> images
    A = torch.zeros(N, M, dtype=torch.bool, device=base.device)
    A.scatter_(1, r_idx, True)
    col = torch.zeros(M, N, dtype=torch.bool, device=base.device)
    col.scatter_(1, c_idx, True)
    A = A & col.T                                           # mutual only
    del col
    Af = A.float()
    deg_i = Af.sum(1, keepdim=True)                         # (N,1)
    deg_c = Af.sum(0, keepdim=True)                         # (1,M)
    P3 = (Af @ Af.T) @ Af                                   # (N, M)
    G = Af + P3 / (deg_i.sqrt() @ deg_c.sqrt()).clamp_min(1e-6)
    return base + lam * G


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["coco", "flickr"], default="coco")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = args.device

    cfg = json.loads((Path("outputs/score_sweep") / f"results_fuse_{args.dataset}.json").read_text())
    fusion, csls_k = cfg["fusion"], cfg["csls_k"]
    logger.info("fusion=%s | csls k=%d", fusion, csls_k)

    cache = Path("outputs/feat_cache")
    feats_by_base = {}
    any_base = None
    for name in fusion:
        base = name[:-4] if name.endswith("+ctr") else name
        any_base = any_base or base
        if base not in feats_by_base:
            feats_by_base[base] = torch.load(cache / f"{base}_{args.dataset}.pt",
                                             weights_only=False)

    def build(idx):
        """Fused (+CSLS) cross-modal matrix for the image subset idx."""
        img, cap = None, None
        for name, w in fusion.items():
            b = name[:-4] if name.endswith("+ctr") else name
            f = feats_by_base[b]
            N_all, K, D = f["text_mus"].shape
            i = f["img_mu"].float()[idx]
            c = f["text_mus"].reshape(N_all * K, D).float()[
                (idx * K).repeat_interleave(K)
                + torch.arange(K).repeat(len(idx))]
            if name.endswith("+ctr"):
                i = i - i.mean(0, keepdim=True)
                c = c - c.mean(0, keepdim=True)
            i = F.normalize(i, dim=-1) * w
            c = F.normalize(c, dim=-1) * w
            img = i if img is None else img + i
            cap = c if cap is None else cap + c
        return csls(img.to(device) @ cap.to(device).T, csls_k)

    N_full = feats_by_base[any_base]["img_mu"].shape[0]
    K = feats_by_base[any_base]["text_mus"].shape[1]
    dev_idx = torch.arange(0, N_full, 2)

    best = None
    base_dev = build(dev_idx)
    for k1 in K1_GRID:
        for lam in LAM_GRID:
            r = fast_recalls(bipartite_diffusion(base_dev, k1, lam), len(dev_idx), K)
            logger.info("dev k1=%d lam=%.1f -> i2t@1=%.4f t2i@1=%.4f mr=%.4f",
                        k1, lam, r["i2t@1"], r["t2i@1"], r["mr"])
            if best is None or r["mr"] > best[0]:
                best = (r["mr"], k1, lam)
    logger.info("dev-selected k1=%d lam=%.1f (mr=%.4f)", best[1], best[2], best[0])
    del base_dev
    torch.cuda.empty_cache()

    base_full = build(torch.arange(N_full))
    r_full = fast_recalls(bipartite_diffusion(base_full, best[1], best[2]), N_full, K)
    tgt = TARGETS[args.dataset]
    logger.info("%s FULL bipartite diffusion (k1=%d, lam=%.1f):",
                args.dataset, best[1], best[2])
    for k in ["i2t@1", "i2t@5", "i2t@10", "t2i@1", "t2i@5", "t2i@10"]:
        logger.info("  %-7s %.4f   target %.4f   %s", k, r_full[k], tgt[k],
                    "PASS" if r_full[k] > tgt[k] else "gap %+.4f" % (r_full[k] - tgt[k]))

    out = Path("outputs/score_sweep") / f"results_krecip_{args.dataset}.json"
    out.write_text(json.dumps({
        "dataset": args.dataset, "k1": best[1], "lam": best[2],
        "full": r_full, "targets": tgt,
    }, indent=2))
    logger.info("saved %s", out)


if __name__ == "__main__":
    main()
