"""
GaussianImageDistribution - Shared image-text retrieval utilities.

Centralizes Recall@K computation so every evaluation script reports the same
metrics in the same way, including BOTH retrieval directions:

    - I2T (Image -> Text): query = image, gallery = text
    - T2I (Text -> Image): query = text,  gallery = image

Previously each eval script carried its own (I2T-only) copy of this logic.
"""

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
def compute_recall_msda_chunked(
    img_mu: torch.Tensor,
    img_logvar: torch.Tensor,
    text_mu: torch.Tensor,
    text_logvar: torch.Tensor,
    k_values: List[int],
    tau: float = 0.07,
    chunk_size: int = 1000,
) -> Dict[str, float]:
    """
    Bidirectional Recall@K under the MSDA uncertainty-discounted cosine score,
    i.e. the same score used by the L_set contrastive loss (train and eval agree):

        sim(x, y) = (mu_x . mu_y) / (tau * sqrt(1 + mean(sigma_x^2)) * sqrt(1 + mean(sigma_y^2)))

    Means are L2-normalized internally; mean(sigma^2) averages over D.

    Returns keys: ``msda_recall_i2t@{k}``, ``msda_recall_t2i@{k}``,
    ``msda_recall@{k}`` (mean of the two directions).
    """
    img_mu_n = F.normalize(img_mu, dim=-1)
    text_mu_n = F.normalize(text_mu, dim=-1)
    img_scale = torch.sqrt(1.0 + torch.exp(img_logvar).mean(dim=-1))    # (N,)
    text_scale = torch.sqrt(1.0 + torch.exp(text_logvar).mean(dim=-1))  # (N,)
    n = img_mu.shape[0]

    def _direction(query_mu, gallery_mu, query_scale, gallery_scale, desc):
        hits = {k: 0 for k in k_values}
        for start in tqdm(range(0, n, chunk_size), desc=desc, leave=False):
            end = min(start + chunk_size, n)
            sim = torch.matmul(query_mu[start:end], gallery_mu.T)
            scale = query_scale[start:end].unsqueeze(1) * gallery_scale.unsqueeze(0)
            sim = sim / (tau * scale)
            ranked = torch.argsort(sim, dim=1, descending=True)
            gt = torch.arange(start, end, device=query_mu.device).unsqueeze(1)
            for k in k_values:
                hits[k] += (ranked[:, :k] == gt).any(dim=1).sum().item()
        return {k: hits[k] / n for k in k_values}

    i2t = _direction(img_mu_n, text_mu_n, img_scale, text_scale, "MSDA I2T chunks")
    t2i = _direction(text_mu_n, img_mu_n, text_scale, img_scale, "MSDA T2I chunks")

    out: Dict[str, float] = {}
    for k in k_values:
        out[f"msda_recall_i2t@{k}"] = i2t[k]
        out[f"msda_recall_t2i@{k}"] = t2i[k]
        out[f"msda_recall@{k}"] = (i2t[k] + t2i[k]) / 2
    return out


@torch.no_grad()
def compute_i2t_caption_pair_counts(
    img_mu: torch.Tensor,
    img_logvar: torch.Tensor,
    text_mus: torch.Tensor,
    text_logvars: torch.Tensor,
    k_values: List[int],
    tau: float = 0.07,
    chunk_size: int = 1000,
) -> Dict[str, float]:
    """
    Image->text retrieval against a per-caption gallery: for each image, count
    how many of its K captions land in the top-K retrieved, averaged over images.

    Unlike :func:`compute_recall_msda_chunked` (1:1 against the single merged
    text mean), the gallery here is ALL N*K per-caption means, and image i has K
    positives -- its own captions at the flattened (image-major) indices
    ``[i*K, i*K+K)``. The reported value is the mean per-image hit count
    (range ``0..K``), under both the cosine and the MSDA uncertainty-discounted
    scores.

    Args:
        img_mu:        (N, D) image means
        img_logvar:    (N, D) image log-variances (MSDA scorer only)
        text_mus:      (N, K, D) per-caption means
        text_logvars:  (N, K, D) per-caption log-variances (MSDA scorer only)
        k_values:      top-K cutoffs for the hit count (e.g. [5, 10])
        tau:           MSDA score temperature
        chunk_size:    query (image) chunk size to bound the sim-matrix memory

    Returns:
        ``{cos_pair_count@{k}, msda_pair_count@{k}}`` for each k in ``k_values``.
    """
    N, K, _ = text_mus.shape
    gallery_mu = text_mus.reshape(N * K, -1)             # (N*K, D)
    gallery_logvar = text_logvars.reshape(N * K, -1)     # (N*K, D)

    img_mu_n = F.normalize(img_mu, dim=-1)               # (N, D)
    gallery_mu_n = F.normalize(gallery_mu, dim=-1)       # (N*K, D)
    img_scale = torch.sqrt(1.0 + torch.exp(img_logvar).mean(dim=-1))          # (N,)
    gallery_scale = torch.sqrt(1.0 + torch.exp(gallery_logvar).mean(dim=-1))  # (N*K,)

    def _count(scorer, desc):
        hits = {k: 0 for k in k_values}
        for start in tqdm(range(0, N, chunk_size), desc=desc, leave=False):
            end = min(start + chunk_size, N)
            sim = scorer(img_mu_n[start:end], gallery_mu_n,
                         img_scale[start:end], gallery_scale)   # (chunk, N*K)
            ranked = torch.argsort(sim, dim=1, descending=True)
            rows = torch.arange(start, end, device=sim.device).unsqueeze(1)   # (chunk, 1) global img idx
            pos_lo = rows * K
            pos_hi = pos_lo + K
            for k in k_values:
                in_range = (ranked[:, :k] >= pos_lo) & (ranked[:, :k] < pos_hi)
                hits[k] += in_range.sum().item()
        return {k: hits[k] / N for k in k_values}

    def _cos(q, g, _qs, _gs):
        return torch.matmul(q, g.T)

    def _msda(q, g, qs, gs):
        sim = torch.matmul(q, g.T)
        return sim / (tau * qs.unsqueeze(1) * gs.unsqueeze(0))

    cos = _count(_cos, "I2T pair-count (cos)")
    msda = _count(_msda, "I2T pair-count (msda)")

    out: Dict[str, float] = {}
    for k in k_values:
        out[f"cos_pair_count@{k}"] = cos[k]
        out[f"msda_pair_count@{k}"] = msda[k]
    return out
