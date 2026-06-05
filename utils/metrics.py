"""
GaussianImageDistribution - Evaluation Metrics

This module provides evaluation metrics for image-text retrieval,
particularly Recall@K metrics.
"""

from typing import Dict, List, Tuple

import numpy as np
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
                     For image-to-text: similarities[i, j] = similarity(image_i, text_j)
                     Assumes diagonal elements (i, i) are the ground truth pairs
        k_values: List of K values for Recall@K computation

    Returns:
        Dictionary mapping K to recall@K value
    """
    n_queries = similarities.shape[0]
    recalls = {}

    # Get ranking indices (sorted by similarity, descending)
    # shape: (N_queries, N_targets)
    ranked_indices = torch.argsort(similarities, dim=1, descending=True)

    for k in k_values:
        # Get top-K indices for each query
        # shape: (N_queries, K)
        top_k_indices = ranked_indices[:, :k]

        # Check if ground truth (diagonal) is in top-K
        # Ground truth index for query i is i
        ground_truth_indices = torch.arange(n_queries, device=similarities.device)
        ground_truth_indices = ground_truth_indices.unsqueeze(1)  # (N_queries, 1)

        # Check if ground truth is in top-K
        # shape: (N_queries, K)
        is_in_top_k = (top_k_indices == ground_truth_indices).any(dim=1)

        # Compute recall
        recall_k = is_in_top_k.float().mean().item()
        recalls[k] = recall_k

    return recalls


def compute_image_to_text_recall(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[int, float]:
    """
    Compute Image-to-Text Recall@K.

    For each image, retrieve the most similar texts and check if the correct
    text is in the top-K results.

    Args:
        image_features: Image features of shape (N_images, D)
        text_features: Text features of shape (N_texts, D)
        k_values: List of K values

    Returns:
        Dictionary mapping K to recall@K value
    """
    # Compute similarity matrix: (N_images, N_texts)
    similarities = image_features @ text_features.T

    return compute_recall_at_k(similarities, k_values)


def compute_text_to_image_recall(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[int, float]:
    """
    Compute Text-to-Image Recall@K.

    For each text, retrieve the most similar images and check if the correct
    image is in the top-K results.

    Args:
        image_features: Image features of shape (N_images, D)
        text_features: Text features of shape (N_texts, D)
        k_values: List of K values

    Returns:
        Dictionary mapping K to recall@K value
    """
    # Compute similarity matrix: (N_texts, N_images)
    similarities = text_features @ image_features.T

    return compute_recall_at_k(similarities, k_values)


def compute_bidirectional_recall(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, Dict[int, float]]:
    """
    Compute both Image-to-Text and Text-to-Image Recall@K.

    Args:
        image_features: Image features of shape (N_images, D)
        text_features: Text features of shape (N_texts, D)
        k_values: List of K values

    Returns:
        Dictionary with keys "image_to_text" and "text_to_image",
        each mapping to a dict of K -> recall@K
    """
    i2t_recall = compute_image_to_text_recall(image_features, text_features, k_values)
    t2i_recall = compute_text_to_image_recall(image_features, text_features, k_values)

    return {
        "image_to_text": i2t_recall,
        "text_to_image": t2i_recall
    }


def format_recall_results(recall_dict: Dict[str, Dict[int, float]]) -> str:
    """
    Format recall results as a readable string.

    Args:
        recall_dict: Output from compute_bidirectional_recall

    Returns:
        Formatted string
    """
    lines = ["Recall@K Results:", "=" * 50]

    for direction, recalls in recall_dict.items():
        direction_name = "Image-to-Text" if direction == "image_to_text" else "Text-to-Image"
        lines.append(f"\n{direction_name}:")
        for k in sorted(recalls.keys()):
            lines.append(f"  Recall@{k}: {recalls[k]:.4f}")

    return "\n".join(lines)


def compute_mean_recall(recall_dict: Dict[str, Dict[int, float]]) -> Dict[str, float]:
    """
    Compute mean recall across all K values for each direction.

    Args:
        recall_dict: Output from compute_bidirectional_recall

    Returns:
        Dictionary with mean recall for each direction
    """
    mean_recalls = {}

    for direction, recalls in recall_dict.items():
        mean_recalls[direction] = np.mean(list(recalls.values()))

    return mean_recalls


if __name__ == "__main__":
    # Test metrics computation
    import torch

    # Create dummy features (10 samples, 128 dimensions)
    n_samples = 10
    dim = 128

    # Create features with some correlation (diagonal dominance)
    torch.manual_seed(42)
    image_features = torch.randn(n_samples, dim)
    text_features = image_features + 0.1 * torch.randn(n_samples, dim)

    # Normalize features
    image_features = torch.nn.functional.normalize(image_features, dim=1)
    text_features = torch.nn.functional.normalize(text_features, dim=1)

    # Compute recalls
    k_values = [1, 5, 10]
    results = compute_bidirectional_recall(image_features, text_features, k_values)

    # Print results
    print(format_recall_results(results))

    # Compute mean recall
    mean_recalls = compute_mean_recall(results)
    print(f"\nMean Recall: {mean_recalls}")
