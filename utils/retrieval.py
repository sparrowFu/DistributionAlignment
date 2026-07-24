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
