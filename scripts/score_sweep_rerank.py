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


def k_reciprocal(feats_q, feats_g, S, k1, k2, lam, device, chunk=2048):
    """Simplified k-reciprocal: reciprocal sets + k2-neighbor expansion +
    Jaccard similarity, blended with the cosine."""
    Q, G = S.shape
    # adjacency: query->topk gallery, gallery->topk query (as bool mats)
    kq = min(k1, G)
    kg = min(k1, Q)
    A_idx = torch.topk(S, kq, dim=1).indices                    # (Q, kq)
    B_idx = torch.topk(S.T, kg, dim=1).indices                  # (G, kg)
    # reciprocal: R(q) = {g: g in A(q), q in B(g)}
    Bt = torch.zeros(Q, G, dtype=torch.bool, device=device)     # (Q, G) = B^T
    g_arange = torch.arange(G, device=device).unsqueeze(1).expand_as(B_idx)
    Bt[B_idx.reshape(-1), g_arange.reshape(-1)] = True
    A = torch.zeros(Q, G, dtype=torch.bool, device=device)
    q_arange = torch.arange(Q, device=device).unsqueeze(1).expand_as(A_idx)
    A[q_arange.reshape(-1), A_idx.reshape(-1)] = True
    R = A & Bt                                                 # (Q, G)
    # local expansion: R'(q) = R(q) ∪ N(R(q), k2) -- via Bt rows of members
    if k2 > 0:
        extra = torch.zeros_like(R)
        for s in range(0, Q, chunk):
            e = min(s + chunk, Q)
            members = R[s:e]                                    # (c, G) bool
            # union of gallery->topk(k2) queries for each g in R(q):
            nb = B_idx[:, :k2]                                 # (G, k2) queries
            # count contributions via scatter: for row q, |{q' in nb(g)}| >= 1
            acc = torch.zeros(e - s, Q, dtype=torch.float, device=device)
            g_sel = members.nonzero(as_tuple=False)             # (nnz, 2)
            if g_sel.numel():
                rows, gs = g_sel[:, 0], g_sel[:, 1]
                acc.index_put_(
                    (rows.repeat_interleave(k2), nb[gs].reshape(-1)),
                    torch.ones(rows.numel() * k2, device=device),
                    accumulate=True)
            extra[s:e] = acc > 0
        R = R | extra
    del A, Bt, extra
    # Jaccard via float matmul on the (Q,G)x(G,G)... use R (Q,G) with
    # gallery-side reciprocal sets RG (G,G sparse): approximate RG by
    # gallery self-reciprocity through queries: RG = B_rows & A_cols style is
    # too big; instead use the query-membership trick: inter(q,g) counts
    # shared QUERIES in R? Faithful Jaccard needs gallery sets; approximate
    # with the standard simplification: sim(q,g) from R row vs B(g) columns.
    Bq = torch.zeros(G, Q, dtype=torch.float, device=device)   # B as float
    Bq[g_arange.reshape(-1), B_idx.reshape(-1)] = 1.0
    Rf = R.float()
    card_r = Rf.sum(1, keepdim=True)                           # (Q, 1)
    card_b = Bq.sum(1, keepdim=True).T                         # (1, G)
    inter = Rf @ Bq                                            # (Q, G)
    jacc = inter / (card_r + card_b - inter + 1e-8)
    return lam * S + (1 - lam) * jacc


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
        for lam in (0.3, 0.5):
            variants[f"KR:k1=20,lam={lam}"] = k_reciprocal(
                img, cap, C, 20, 6, lam, args.device)

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
