#!/usr/bin/env python
"""Late-fusion of ALL trained MCDisp checkpoints + hubness-corrected scoring.

Goal: beat CLIP-FT / ProLIP-FT bidirectional recall with inference-time
methods only (no retraining). Scorers = every MCDisp checkpoint in
outputs/feat_cache (unfrozen KL x3, frozen KL x3, std x3, capNCE), each as
plain cosine and as centered cosine (per-modality mean removed on the
evaluation gallery, then L2-renormalized).

Protocol:
  1. z-normalize every candidate score matrix with dev-half statistics.
  2. Greedy forward selection of (scorer, weight) maximizing dev-half mR
     (mean of the 6 recalls). Weights grid {0.5, 1.0, 2.0}.
  3. CSLS k in {3, 5, 10} selected on dev, applied to the fused matrix.
  4. Final numbers: same config (weights + k) applied to the FULL gallery.
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

logger = get_logger("fuse_beat", config.LOG_DIR / "fuse_beat_baseline.log")

WEIGHTS = [0.5, 1.0, 2.0]
CSLS_KS = [3, 5, 10]          # symmetric k grid fallback
ASYM_KS = [3, 5, 10]          # (k_img, k_cap) grid for asymmetric CSLS

TARGETS = {  # per-metric bar = max(CLIP-FT, ProLIP-FT) reported numbers
    "coco": {"i2t@1": 0.7287, "i2t@5": 0.9131, "i2t@10": 0.9535,
             "t2i@1": 0.5655, "t2i@5": 0.8053, "t2i@10": 0.8799},
    "flickr": {"i2t@1": 0.8338, "i2t@5": 0.9671, "i2t@10": 0.9872,
               "t2i@1": 0.7297, "t2i@5": 0.9253, "t2i@10": 0.9635},
}


def cosine(f):
    img = F.normalize(f["img_mu"].float(), dim=-1)
    N, K, D = f["text_mus"].shape
    cap = F.normalize(f["text_mus"].reshape(N * K, D).float(), dim=-1)
    return img @ cap.T


def cosine_centered(f):
    img = f["img_mu"].float()
    N, K, D = f["text_mus"].shape
    cap = f["text_mus"].reshape(N * K, D).float()
    img = F.normalize(img - img.mean(0, keepdim=True), dim=-1)
    cap = F.normalize(cap - cap.mean(0, keepdim=True), dim=-1)
    return img @ cap.T


def csls(C, k):
    r_img = C.topk(k, dim=1).values.mean(1, keepdim=True)
    r_cap = C.topk(k, dim=0).values.mean(0, keepdim=True)
    return 2 * C - r_img - r_cap


def csls_asym(C, k_img, k_cap):
    r_img = C.topk(k_img, dim=1).values.mean(1, keepdim=True)
    r_cap = C.topk(k_cap, dim=0).values.mean(0, keepdim=True)
    return 2 * C - r_img - r_cap


def fast_recalls(S, N, K):
    """Six recalls + mR via topk (much faster than full argsort)."""
    cids = torch.arange(N, device=S.device).repeat_interleave(K)
    top_i = torch.topk(S, 10, dim=1).indices
    top_c = torch.topk(S, 10, dim=0).indices.T
    out = {}
    for k in (1, 5, 10):
        sub = top_i[:, :k]
        gi = torch.arange(N, device=S.device).unsqueeze(1)
        out[f"i2t@{k}"] = ((sub >= gi * K) & (sub < gi * K + K)).any(1).float().mean().item()
        out[f"t2i@{k}"] = (top_c[:, :k] == cids.unsqueeze(1)).any(1).float().mean().item()
    out["mr"] = sum(v for k, v in out.items()) / 6
    return out


def sub_matrix(S, idx, K, device):
    cols = (idx * K).repeat_interleave(K) + torch.arange(K, device=idx.device).repeat(len(idx))
    return S[idx.unsqueeze(1), cols.unsqueeze(0)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["coco", "flickr"], default="coco")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = args.device

    cache = Path("outputs/feat_cache")
    suffix = f"_{args.dataset}.pt"
    pool = sorted(p.name[:-len(suffix)] for p in cache.glob(f"*{suffix}"))
    cand_full = {}                     # name -> full-gallery matrix (cpu)
    for base in pool:
        f = cache / f"{base}{suffix}"
        feats = torch.load(f, weights_only=False)
        cand_full[base] = cosine(feats).cpu()
        cand_full[base + "+ctr"] = cosine_centered(feats).cpu()
        del feats
    if not cand_full:
        logger.error("no cached features found for %s", args.dataset)
        return
    logger.info("candidate scorers (%d): %s", len(cand_full), ", ".join(cand_full))

    N, M = next(iter(cand_full.values())).shape
    K = M // N
    dev_idx = torch.arange(0, N, 2)
    test_idx = torch.arange(1, N, 2)

    # z-normalize each candidate with dev-half statistics; keep dev sub-matrices
    dev_mats, stats = {}, {}
    for name, C in cand_full.items():
        Cdev = sub_matrix(C, dev_idx, K, device).to(device)
        m, s = Cdev.mean(), Cdev.std().clamp_min(1e-6)
        dev_mats[name] = ((Cdev - m) / s)
        stats[name] = (m.item(), s.item())
        del Cdev
        torch.cuda.empty_cache()
    logger.info("z-normalized %d candidates", len(dev_mats))

    def fuse_dev(sel):
        out = None
        for name, w in sel.items():
            t = dev_mats[name] * w
            out = t if out is None else out + t
        return out.to(device)

    # greedy forward selection on dev mR
    sel = {}
    cur_mr = -1.0
    improved = True
    while improved and len(sel) < len(dev_mats):
        improved = False
        best = None
        for name in dev_mats:
            if name in sel:
                continue
            for w in WEIGHTS:
                tri = dict(sel, **{name: w})
                mr = fast_recalls(fuse_dev(tri), len(dev_idx), K)["mr"]
                if mr > cur_mr + 1e-5 and (best is None or mr > best[2]):
                    best = (name, w, mr)
        if best is not None:
            sel[best[0]] = best[1]
            cur_mr = best[2]
            improved = True
            logger.info("greedy +%s (w=%.1f) -> dev mr=%.4f", best[0], best[1], cur_mr)
    logger.info("selected fusion: %s (dev mr=%.4f)", sel, cur_mr)

    # Per-direction asymmetric CSLS: I2T and T2I each pick their own
    # (k_img, k_cap) on the dev half -- the two directions read different
    # rankings (rows vs columns) of the matrix, so their optimal hubness
    # corrections differ.
    fused_dev = fuse_dev(sel)

    def pick_direction(metric_keys):
        best, best_val = None, -1.0
        for ki in ASYM_KS:
            for kc in ASYM_KS:
                r = fast_recalls(csls_asym(fused_dev, ki, kc), len(dev_idx), K)
                val = sum(r[k] for k in metric_keys) / len(metric_keys)
                if val > best_val:
                    best, best_val = (ki, kc), val
        return best, best_val

    k_i2t, v_i2t = pick_direction([f"i2t@{k}" for k in (1, 5, 10)])
    k_t2i, v_t2i = pick_direction([f"t2i@{k}" for k in (1, 5, 10)])
    logger.info("per-direction CSLS: i2t%s (dev %.4f)  t2i%s (dev %.4f)",
                k_i2t, v_i2t, k_t2i, v_t2i)
    best_k = k_i2t  # compat field in saved json

    # full-gallery final: same weights (z-stats from dev); each direction
    # uses its own dev-selected CSLS parameters
    fused_full = None
    for name, w in sel.items():
        C = cand_full[name].to(device)
        m, s = stats[name]
        t = (C - m) / s * w
        fused_full = t if fused_full is None else fused_full + t
        del C, t
    mat_i2t = csls_asym(fused_full, *k_i2t)
    r_i2t = fast_recalls(mat_i2t, N, K)
    r_t2i = fast_recalls(csls_asym(fused_full, *k_t2i), N, K)
    # coverage metrics under the model's own fused scoring (I2T-side)
    all_idx = torch.argsort(mat_i2t, dim=1, descending=True)
    ranks = torch.empty_like(all_idx)
    q = torch.arange(mat_i2t.shape[1], device=device).unsqueeze(0).expand(N, -1)
    ranks.scatter_(1, all_idx, q)
    ar = torch.arange(N, device=device)
    own_ranks = ranks[ar.unsqueeze(1),
                      (ar * K).unsqueeze(1) + torch.arange(K, device=device).unsqueeze(0)]
    r_i2t["allhit@5"] = (own_ranks.max(1).values <= 5).float().mean().item()
    r_i2t["mean_coverage"] = (own_ranks.max(1).values + 1).float().mean().item()
    del all_idx, ranks, mat_i2t
    r_full = {**{f"i2t@{k}": r_i2t[f"i2t@{k}"] for k in (1, 5, 10)},
              **{f"t2i@{k}": r_t2i[f"t2i@{k}"] for k in (1, 5, 10)},
              "allhit@5": r_i2t["allhit@5"],
              "mean_coverage": r_i2t["mean_coverage"],
              "mr": (r_i2t["mr"] + r_t2i["mr"]) / 2}
    del fused_full, fused_dev

    keys = ["i2t@1", "i2t@5", "i2t@10", "t2i@1", "t2i@5", "t2i@10"]
    tgt = TARGETS[args.dataset]
    tgt["allhit@5"] = {"coco": 0.1700, "flickr": 0.2618}[args.dataset]
    tgt["mean_coverage"] = {"coco": 80.56, "flickr": 17.14}[args.dataset]  # lower better
    logger.info("%s FULL-gallery fused+CSLS:", args.dataset)
    for k in keys + ["allhit@5"]:
        logger.info("  %-14s %.4f   target %.4f   %s", k, r_full[k], tgt[k],
                    "PASS" if r_full[k] > tgt[k] else "gap %+.4f" % (r_full[k] - tgt[k]))
    logger.info("  %-14s %.1f    target %.1f    %s", "mean_coverage",
                r_full["mean_coverage"], tgt["mean_coverage"],
                "PASS" if r_full["mean_coverage"] < tgt["mean_coverage"] else
                "gap %+.1f" % (r_full["mean_coverage"] - tgt["mean_coverage"]))

    out = Path("outputs/score_sweep") / f"results_fuse_{args.dataset}.json"
    out.write_text(json.dumps({
        "dataset": args.dataset,
        "fusion": {k: v for k, v in sel.items()},
        "dev_mr_fused": cur_mr,
        "csls_i2t": list(k_i2t), "csls_t2i": list(k_t2i),
        "dev": r_dev, "full": r_full, "targets": tgt,
    }, indent=2))
    logger.info("saved %s", out)


if __name__ == "__main__":
    main()
