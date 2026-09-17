#!/usr/bin/env python
"""Path-1d: neighborhood-structure reranking (zero training, no set info).

Moves beyond POINTWISE scores (exhausted: cosine/q/overlap all flat) to the
GALLERY STRUCTURE, which standard reranking exploits:

  V0    plain cosine
  MC    modality centering (subtract per-modality mean, then cosine)
  AQE   alpha query expansion: q' = norm(q + a * mean(top-m gallery feats));
        rescore gallery by cos(q', g)
  KR    k-reciprocal reranking (Zhong et al., CVPR'17): Jaccard distance on
        k-reciprocal neighbor sets, blended with cosine

Label-free, grouping-free; applied IDENTICALLY to MCDisp and CLIP.
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

logger = get_logger("score_sweep4", config.LOG_DIR / "score_sweep4.log")


def modality_centered(img, cap):
    return (F.normalize(img - img.mean(0, keepdim=True), dim=-1)
            @ F.normalize(cap - cap.mean(0, keepdim=True), dim=-1).T)


def alpha_qe(feats_q, feats_g, S, alpha, m):
    """Expand each query with the mean of its top-m gallery features."""
    topm = torch.topk(S, m, dim=1).indices                     # (Q, m)
    eq = feats_g[topm].mean(1)                                 # (Q, D)
    qe = F.normalize(feats_q + alpha * eq, dim=-1)
    return qe @ F.normalize(feats_g, dim=-1).T


def graph_smooth(img, cap, S, kq, kg, alpha, device, chunk=4096):
    """Cross-modal graph smoothing (label-prop family): smooth the score
    matrix over same-modality kNN graphs.

        S' = (1-a)*S + a*Wq @ S        (image side)
        S'' = (1-a)*S' + a*S' @ Wg.T   (caption side)

    Wq / Wg are row-normalized kNN means built from the SAME features that
    produced S. Label-free, grouping-free.
    """
    q_top = torch.topk(img @ img.T, kq, dim=1).indices        # (Q, kq)
    g_top = torch.topk(cap @ cap.T, kg, dim=1).indices        # (G, kg)
    Q, G = S.shape
    out = torch.empty_like(S)
    for s0 in range(0, Q, chunk):
        s1 = min(s0 + chunk, Q)
        out[s0:s1] = (1 - alpha) * S[s0:s1] + alpha * S[q_top[s0:s1]].mean(1)
    S = out
    out = torch.empty_like(S)
    for g0 in range(0, G, chunk):
        g1 = min(g0 + chunk, G)
        out[:, g0:g1] = (1 - alpha) * S[:, g0:g1] + alpha * S[:, g_top[g0:g1]].mean(2)
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

        img = F.normalize(f["img_mu"].to(args.device), dim=-1)
        N, K, D = f["text_mus"].shape
        cap = F.normalize(f["text_mus"].reshape(N * K, D).to(args.device), dim=-1)
        C = img @ cap.T
        variants = {"V0": C, "MC": modality_centered(f["img_mu"].to(args.device),
                                                     f["text_mus"].reshape(N * K, D).to(args.device))}
        for a in (0.5, 1.0):
            for m in (3, 5):
                variants[f"AQE:a={a},m={m}"] = alpha_qe(img, cap, C, a, m)
        for a in (0.3, 0.5):
            variants[f"GS:a={a}"] = graph_smooth(img, cap, C, 10, 10, a, args.device)

        dev_idx = torch.arange(0, N, 2, device=args.device)
        test_idx = torch.arange(1, N, 2, device=args.device)

        def _cols(idx):
            return (idx * K).repeat_interleave(K) + torch.arange(K, device=args.device).repeat(len(idx))

        report, best = {}, None
        for name, S in variants.items():
            r_dev = recalls_from_scores(S[dev_idx][:, _cols(dev_idx)], len(dev_idx), K, args.device)
            r_test = recalls_from_scores(S[test_idx][:, _cols(test_idx)], len(test_idx), K, args.device)
            del S
            report[name] = {"dev_mr": r_dev["mr"], "test": r_test}
            if best is None or r_dev["mr"] > best[1]:
                best = (name, r_dev["mr"], r_test)
        del variants, C
        torch.cuda.empty_cache()
        results[key] = {"variants": report, "dev_selected": {"name": best[0], "test": best[2]}}
        logger.info(f"{key}: dev-selected {best[0]} -> test mr={best[2]['mr']:.4f} "
                    f"(V0 {report['V0']['test']['mr']:.4f})")

    out = Path("outputs/score_sweep")
    (out / "results_rerank.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
