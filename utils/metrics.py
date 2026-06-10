"""
GaussianImageDistribution - Evaluation Metrics

This module provides evaluation metrics for image-text retrieval,
particularly Recall@K metrics.
"""

from typing import Dict, List

import torch


def compute_recall_at_k(
    similarities: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[int, float]:
    """
    Compute Recall@K for retrieval task.

    For each query, check if the correct result is in the top-K predictions.

    Args:
        similarities: Similarity matrix of shape (N_queries, N_targets)
        k_values: List of K values for Recall@K computation

    Returns:
        Dictionary mapping K to recall@K value
    """
    n_queries = similarities.shape[0]
    recalls = {}

    ranked_indices = torch.argsort(similarities, dim=1, descending=True)

    for k in k_values:
        top_k_indices = ranked_indices[:, :k]
        ground_truth_indices = torch.arange(n_queries, device=similarities.device)
        ground_truth_indices = ground_truth_indices.unsqueeze(1)
        is_in_top_k = (top_k_indices == ground_truth_indices).any(dim=1)
        recall_k = is_in_top_k.float().mean().item()
        recalls[k] = recall_k

    return recalls
