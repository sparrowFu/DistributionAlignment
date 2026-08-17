"""Shared ranking core for the ablation metric modules.

Every retrieval-flavored metric in the plan (mR, PairCount, Worst-caption Rank,
stratified retrieval, scorer interventions, likelihood retrieval) reduces to the
same object: a similarity matrix between N images and the N*K per-caption
gallery, plus the per-image positive index range ``[i*K, (i+1)*K)``.

This module materializes that matrix once per scorer (float16, bounded memory)
and derives from it:

  * positive caption ranks (N, K), 1-based, under the full N*K gallery
  * I2T any-hit Recall@K (plan §3.3) and per-image hit vectors (for the
    paired bootstrap and stratified retrieval)
  * T2I Recall@K (each caption query's single positive = its own image)
  * PairCount@K (own captions in top-K) and Worst-caption Rank (max own rank)

Scorers: ``mc`` (uncertainty-discounted cosine -- the training-consistent
primary), ``cos`` (plain cosine baseline). The full Gaussian likelihood scorer
lives in ``gaussian_scorer.py`` and feeds the same rank-derivation helper.
"""

from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F


def mc_sim_rows(
    img_mu: torch.Tensor, img_logvar: torch.Tensor,
    cap_mu_flat: torch.Tensor, cap_lv_flat: torch.Tensor,
    tau: float = 0.07, chunk: int = 256, dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """(N, M) uncertainty-discounted cosine scores, M = N*K captions.

    ``dtype`` controls storage (float16 halves memory on large galleries);
    pass ``torch.float32`` when exact ranking invariances must survive
    rounding (e.g. the §7.2 query-side invariance check).
    """
    q = F.normalize(img_mu, dim=-1)
    g = F.normalize(cap_mu_flat, dim=-1)
    qs = torch.sqrt(1.0 + torch.exp(img_logvar).mean(dim=-1))
    gs = torch.sqrt(1.0 + torch.exp(cap_lv_flat).mean(dim=-1))
    rows = []
    for s in range(0, q.shape[0], chunk):
        e = min(s + chunk, q.shape[0])
        sim = q[s:e] @ g.T / (tau * qs[s:e].unsqueeze(1) * gs.unsqueeze(0))
        rows.append(sim.to(dtype).cpu())
    return torch.cat(rows, dim=0)


def cos_sim_rows(
    img_mu: torch.Tensor, _img_logvar: torch.Tensor,
    cap_mu_flat: torch.Tensor, _cap_lv_flat: torch.Tensor,
    tau: float = 0.07, chunk: int = 256, dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """(N, M) plain cosine scores."""
    q = F.normalize(img_mu, dim=-1)
    g = F.normalize(cap_mu_flat, dim=-1)
    rows = []
    for s in range(0, q.shape[0], chunk):
        e = min(s + chunk, q.shape[0])
        rows.append((q[s:e] @ g.T).to(dtype).cpu())
    return torch.cat(rows, dim=0)


def _positive_ranks(sim: torch.Tensor, K: int, chunk: int = 64) -> torch.Tensor:
    """Exact 1-based ranks of each image's K own captions in the full gallery.

    rank(i, k) = 1 + #{j : sim[i, j] > sim[i, own_k]} (ties resolve favorably,
    standard retrieval convention). Chunked so peak memory stays ~(chunk, K, M).
    """
    N, M = sim.shape
    sf = sim.float()
    ranks = torch.empty(N, K, dtype=torch.long)
    k_off = torch.arange(K).unsqueeze(0)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        rows = torch.arange(s, e)
        idx = (rows * K).unsqueeze(1) + k_off                       # (c, K)
        pv = sf[rows.unsqueeze(1), idx]                             # (c, K)
        gt = sf[s:e].unsqueeze(1) > pv.unsqueeze(2)                 # (c, K, M)
        ranks[s:e] = gt.sum(dim=2) + 1
    return ranks


def ranks_from_sim(sim: torch.Tensor, K: int, k_values=(1, 5, 10)) -> Dict[str, object]:
    """Derive the plan's retrieval statistics from one (N, N*K) score matrix.

    Returns dict with:
      ``pos_ranks``  (N, K) LongTensor, 1-based rank of each own caption
      ``i2t_hit``    {k: bool (N,)} any-own-caption-in-top-k per image
      ``i2t_r``      {k: float} I2T Recall@K
      ``t2i_r``      {k: float} T2I Recall@K (per-caption queries)
      ``t2i_hit``    {k: bool (N*K,)} per-caption-query hit vector
      ``pair_count`` {k: float} mean own-captions in top-K
      ``worst_rank`` (N,) LongTensor max own-caption rank per image
    """
    N, M = sim.shape
    maxk = max(k_values)
    top = torch.topk(sim.float(), min(maxk, M), dim=1).indices          # (N, maxk)
    rows = torch.arange(N).unsqueeze(1)
    in_own_range = (top >= rows * K) & (top < rows * K + K)             # (N, maxk)

    i2t_hit, i2t_r = {}, {}
    for k in k_values:
        hit = in_own_range[:, :k].any(dim=1)
        i2t_hit[k] = hit
        i2t_r[k] = hit.float().mean().item()

    pos_ranks = _positive_ranks(sim, K)
    worst_rank = pos_ranks.max(dim=1).values
    pair_count = {k: (pos_ranks <= k).sum(dim=1).float().mean().item() for k in k_values}

    # T2I: caption query j ranks the N images = column ranking (topk along
    # dim=0 keeps memory bounded; no full transpose materialization).
    t2i_hit, t2i_r = {}, {}
    col_top = torch.topk(sim.float(), min(maxk, N), dim=0).indices      # (<=maxk, M)
    q_idx = torch.arange(M).unsqueeze(0)
    match = col_top == (q_idx // K)
    for k in k_values:
        hit = match[:k].any(dim=0)
        t2i_hit[k] = hit
        t2i_r[k] = hit.float().mean().item()

    return {
        "pos_ranks": pos_ranks, "worst_rank": worst_rank,
        "i2t_hit": i2t_hit, "i2t_r": i2t_r,
        "t2i_hit": t2i_hit, "t2i_r": t2i_r,
        "pair_count": pair_count,
    }


def mr_from(r: Dict[str, object], k_values=(1, 5, 10)) -> float:
    """mR = mean of the 6 I2T+T2I recalls (plan §6.1)."""
    vals = [r["i2t_r"][k] for k in k_values] + [r["t2i_r"][k] for k in k_values]
    return sum(vals) / len(vals)


def full_ranking(
    img_mu: torch.Tensor, img_logvar: torch.Tensor,
    text_mus: torch.Tensor, text_logvars: torch.Tensor,
    k_values=(1, 5, 10), tau: float = 0.07,
    sim_builder: Optional[Callable] = None,
) -> Dict[str, object]:
    """Convenience: mc-scored ranking over (N, K, D) per-caption features."""
    N, K, D = text_mus.shape
    cap_mu = text_mus.reshape(N * K, D)
    cap_lv = text_logvars.reshape(N * K, D)
    builder = sim_builder or mc_sim_rows
    sim = builder(img_mu, img_logvar, cap_mu, cap_lv, tau=tau)
    return ranks_from_sim(sim, K, k_values)
