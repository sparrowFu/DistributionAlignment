"""Shared image-text Recall@K computation. Supports both retrieval directions: I2T (query=image, gallery=text) and T2I (query=text, gallery=image)."""

from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def compute_recall_chunked(
    query: torch.Tensor,
    gallery: torch.Tensor,
    k_values: List[int],
    chunk_size: int = 1000,
    normalize: bool = True,
) -> Dict[int, float]:
    """
    Recall@K where query[i]'s positive match is gallery[i] (diagonal pairing).

    Direction-agnostic: pass images as ``query`` for I2T, texts as ``query``
    for T2I.

    Args:
        query: Query features (N, D)
        gallery: Gallery features (N, D); the i-th gallery item matches query i
        k_values: List of K values for Recall@K
        chunk_size: Chunk size to avoid OOM on large similarity matrices
        normalize: L2-normalize features before computing cosine similarity

    Returns:
        Dict mapping K -> recall@K (raw K ints, e.g. {1: 0.82, 5: 0.97, 10: 0.99})
    """
    n = query.shape[0]
    if normalize:
        query = F.normalize(query, dim=-1)
        gallery = F.normalize(gallery, dim=-1)

    hits = {k: 0 for k in k_values}

    for start in tqdm(range(0, n, chunk_size), desc="Recall chunks", leave=False):
        end = min(start + chunk_size, n)
        sim_chunk = torch.matmul(query[start:end], gallery.T)  # (chunk, N)
        ranked = torch.argsort(sim_chunk, dim=1, descending=True)
        gt = torch.arange(start, end, device=query.device).unsqueeze(1)
        for k in k_values:
            hits[k] += (ranked[:, :k] == gt).any(dim=1).sum().item()

    return {k: hits[k] / n for k in k_values}


@torch.no_grad()
def compute_recall_bidirectional(
    img_features: torch.Tensor,
    text_features: torch.Tensor,
    k_values: List[int],
    chunk_size: int = 1000,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Bidirectional (I2T + T2I) Recall@K between aligned image/text feature sets.

    Used by the CLIP baseline evaluations (single-feature models without
    distribution parameters).

    Returns keys: ``recall_i2t@{k}``, ``recall_t2i@{k}``, ``recall@{k}`` (mean).
    """
    i2t = compute_recall_chunked(img_features, text_features, k_values, chunk_size, normalize)
    t2i = compute_recall_chunked(text_features, img_features, k_values, chunk_size, normalize)

    out: Dict[str, float] = {}
    for k in k_values:
        out[f"recall_i2t@{k}"] = i2t[k]
        out[f"recall_t2i@{k}"] = t2i[k]
        out[f"recall@{k}"] = (i2t[k] + t2i[k]) / 2
    return out


@torch.no_grad()
def compute_multicaption_recall(
    img_mu: torch.Tensor,
    img_logvar: torch.Tensor,
    text_mus: torch.Tensor,
    text_logvars: torch.Tensor,
    k_values: List[int],
    tau: float = 0.07,
    chunk_size: int = 1000,
) -> Dict[str, float]:
    """Standard multi-caption bidirectional Recall@K (N images vs N*K captions).

    THE evaluation protocol for MCDisp_Align: checkpoint selection during
    training (``select_by="recall"`` / ``"mr"``) and test-time evaluation both
    use it. This is the canonical MS-COCO/Flickr30k retrieval protocol
    (evaluating on the merged text mean would collapse the one-to-many
    structure at eval time and is not comparable to published baselines):

      I2T: query = N image means, gallery = N*K per-caption means. Image i has K
           positives -- its own captions at flattened indices [i*K, i*K+K). A hit
           is ANY of them landing in the top-K (any-hit).
      T2I: query = N*K per-caption means, gallery = N image means. Each caption's
           ONLY positive is its own image (single-positive per query).

    Two score families are computed:

      cosine : the PLAIN cosine of the means -- the same score the L_ctr
               contrastive loss optimizes (paper §3.3: the similarity involves
               no variance). This is the primary/headline metric.
      CSD    : the ProLIP-style Contraction-Subspace distance used by this
               repo's ProLIP baselines (utils/retrieval_metrics.py):
               score = cos(mu_q, mu_g) - 0.5 * sum_d sigma_g,d^2, i.e. the
               GALLERY side is discounted by its uncertainty. NOTE the two
               families use different uncertainty semantics: ProLIP's sigma
               means "unreliability" (small, learned implicitly), whereas our
               image variance means "semantic range" (dispersion-supervised,
               ~caption-spread scale) -- see the CSD probe discussion before
               using it as a headline number.

    ``tau`` is accepted for API compatibility and unused. Means are
    L2-normalized internally. Uses torch.topk (not a full argsort) since the
    N*K gallery is large and only the top max(k_values) are needed.

    Returns keys (per k in k_values)::

        mc_recall_i2t@{k}, mc_recall_t2i@{k}, mc_recall@{k} (mean of the two)
        mc_csd_recall_i2t@{k}, mc_csd_recall_t2i@{k}, mc_csd_recall@{k}
    """
    N, K, _ = text_mus.shape
    cap_mu = text_mus.reshape(N * K, -1)              # (N*K, D)
    cap_lv = text_logvars.reshape(N * K, -1)          # (N*K, D)

    img_mu_n = F.normalize(img_mu, dim=-1)            # (N, D)
    cap_mu_n = F.normalize(cap_mu, dim=-1)            # (N*K, D)
    img_unc = torch.exp(img_logvar).sum(dim=-1)       # (N,) sum sigma_v^2
    cap_unc = torch.exp(cap_lv).sum(dim=-1)           # (N*K,) sum sigma_k^2

    maxk = max(k_values)

    def _i2t(use_csd: bool):
        """N image queries vs N*K caption gallery; any-of-K-own-caption hit."""
        hits = {k: 0 for k in k_values}
        for start in tqdm(range(0, N, chunk_size),
                          desc="MC I2T (csd)" if use_csd else "MC I2T", leave=False):
            end = min(start + chunk_size, N)
            sim = img_mu_n[start:end] @ cap_mu_n.T                    # (chunk, N*K)
            if use_csd:
                sim = sim - 0.5 * cap_unc.unsqueeze(0)                # gallery-side discount
            mk = min(maxk, sim.size(1))
            top = torch.topk(sim, mk, dim=1).indices                   # (chunk, mk)
            rows = torch.arange(start, end, device=sim.device).unsqueeze(1)
            in_range = (top >= rows * K) & (top < rows * K + K)        # (chunk, mk)
            for k in k_values:
                hits[k] += in_range[:, :k].any(dim=1).sum().item()     # :k clamps to mk
        return {k: hits[k] / N for k in k_values}

    def _t2i(use_csd: bool):
        """N*K caption queries vs N image gallery; single-positive per query."""
        Q = N * K
        hits = {k: 0 for k in k_values}
        for start in tqdm(range(0, Q, chunk_size),
                          desc="MC T2I (csd)" if use_csd else "MC T2I", leave=False):
            end = min(start + chunk_size, Q)
            sim = cap_mu_n[start:end] @ img_mu_n.T                    # (chunk_q, N)
            if use_csd:
                sim = sim - 0.5 * img_unc.unsqueeze(0)                # gallery-side discount
            mk = min(maxk, sim.size(1))
            top = torch.topk(sim, mk, dim=1).indices                   # (chunk_q, mk)
            q_idx = torch.arange(start, end, device=sim.device)
            gt_img = (q_idx // K).unsqueeze(1)                         # (chunk_q, 1)
            match = top == gt_img                                      # (chunk_q, mk)
            for k in k_values:
                hits[k] += match[:, :k].any(dim=1).sum().item()
        return {k: hits[k] / Q for k in k_values}

    i2t, t2i = _i2t(False), _t2i(False)
    csd_i2t, csd_t2i = _i2t(True), _t2i(True)

    out: Dict[str, float] = {}
    for k in k_values:
        out[f"mc_recall_i2t@{k}"] = i2t[k]
        out[f"mc_recall_t2i@{k}"] = t2i[k]
        out[f"mc_recall@{k}"] = (i2t[k] + t2i[k]) / 2
        out[f"mc_csd_recall_i2t@{k}"] = csd_i2t[k]
        out[f"mc_csd_recall_t2i@{k}"] = csd_t2i[k]
        out[f"mc_csd_recall@{k}"] = (csd_i2t[k] + csd_t2i[k]) / 2
    return out
