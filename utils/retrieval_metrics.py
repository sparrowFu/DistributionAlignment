"""
Retrieval Metrics

Image-text retrieval Recall@K supporting both retrieval directions (I2T and T2I)
and two similarity notions used by probabilistic models:

  - cosine : mean . mean (means are L2-normalized)
  - CSD    : Contraction-SubSpace distance.
            score(image_i, text_j) = mu_i . mu_j - 0.5 * sigma^2_j
            I2T discounts by the *text* uncertainty, T2I by the *image* uncertainty
            (i.e. the uncertainty of the gallery side).

The similarity matrix is processed in chunks over the query axis to avoid OOM on
large galleries. Ground truth is the diagonal (image i matches text i).
"""

from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _direction_recall(
    query_mean: torch.Tensor,
    gallery_mean: torch.Tensor,
    gallery_logvar: torch.Tensor,
    k_values: List[int],
    use_csd: bool,
    chunk_size: int,
    desc: str,
) -> Dict[str, float]:
    """Rank the gallery for each query and compute Recall@K.

    Args:
        query_mean: (N, D) query means (re-normalized here)
        gallery_mean: (G, D) gallery means (re-normalized here)
        gallery_logvar: (G, D) gallery log variances (for CSD discount)
        k_values: list of K
        use_csd: if True, subtract 0.5 * sum(sigma^2) of each gallery item
        chunk_size: query-axis chunk size to bound peak memory
        desc: tqdm description

    Returns:
        {"recall@k": float} for each k. Ground truth is the diagonal
        (query i matches gallery i).
    """
    query_mean = F.normalize(query_mean, dim=-1)
    gallery_mean = F.normalize(gallery_mean, dim=-1)
    n = query_mean.shape[0]
    gallery_unc = torch.exp(gallery_logvar).sum(dim=-1)     # (G,) sum sigma^2

    hits = {k: 0 for k in k_values}
    for start in tqdm(range(0, n, chunk_size), desc=desc):
        end = min(start + chunk_size, n)
        sim = query_mean[start:end] @ gallery_mean.T        # (chunk, G)
        if use_csd:
            sim = sim - 0.5 * gallery_unc.unsqueeze(0)      # discount by gallery uncertainty
        ranked = torch.argsort(sim, dim=1, descending=True)
        gt = torch.arange(start, end, device=sim.device).unsqueeze(1)
        for k in k_values:
            top_k = ranked[:, :k]
            hits[k] += (top_k == gt).any(dim=1).sum().item()

    return {f"recall@{k}": hits[k] / n for k in k_values}


def compute_retrieval_metrics(
    img_mean: torch.Tensor,
    img_logvar: torch.Tensor,
    text_mean: torch.Tensor,
    text_logvar: torch.Tensor,
    k_values: List[int],
    chunk_size: int = 1000,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute I2T + T2I Recall@K under cosine and CSD.

    Args:
        img_mean, img_logvar: (N, D); img_logvar = log sigma^2
        text_mean, text_logvar: (N, D); text_logvar = log sigma^2
        k_values: list of K
        chunk_size: query-axis chunk size

    Returns:
        {"i2t": {"cosine": {recall@k}, "csd": {recall@k}},
         "t2i": {"cosine": {...}, "csd": {...}}}
    """
    return {
        "i2t": {
            "cosine": _direction_recall(img_mean, text_mean, text_logvar,
                                        k_values, False, chunk_size, "I2T cosine"),
            "csd": _direction_recall(img_mean, text_mean, text_logvar,
                                     k_values, True, chunk_size, "I2T CSD"),
        },
        "t2i": {
            "cosine": _direction_recall(text_mean, img_mean, img_logvar,
                                        k_values, False, chunk_size, "T2I cosine"),
            "csd": _direction_recall(text_mean, img_mean, img_logvar,
                                     k_values, True, chunk_size, "T2I CSD"),
        },
    }
