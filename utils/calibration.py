"""
GaussianImageDistribution - Uncertainty Calibration Metrics (Exp3)

Implements calibration metrics for evaluating whether σ² truly encodes
semantic uncertainty. These metrics are used in Exp3.

Metrics implemented:
    - ECE (Expected Calibration Error)
    - MCE (Maximum Calibration Error)
    - NLL (Negative Log-Likelihood)
    - Brier Score
    - AUROC (using 1-confidence as OOD score)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import config
from utils.logger import get_logger


logger = get_logger("calibration")


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE).

    ECE = Σ_b (n_b / N) * |acc_b - conf_b|

    Args:
        confidences: Predicted confidence (max softmax prob), shape (N,)
        accuracies: Binary correctness indicators, shape (N,)
        num_bins: Number of equal-width bins

    Returns:
        (ece, bin_accuracies, bin_confidences, bin_counts)
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    for i in range(num_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_accuracies[i] = accuracies[mask].mean()
            bin_confidences[i] = confidences[mask].mean()
            bin_counts[i] = mask.sum()

    ece = np.sum(bin_counts / bin_counts.sum() * np.abs(bin_accuracies - bin_confidences))
    return ece, bin_accuracies, bin_confidences, bin_counts


def compute_mce(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 15,
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE = max_b |acc_b - conf_b|
    """
    _, bin_acc, bin_conf, bin_counts = compute_ece(confidences, accuracies, num_bins)
    # Only consider bins with samples
    valid = bin_counts > 0
    if not valid.any():
        return 0.0
    return np.max(np.abs(bin_acc[valid] - bin_conf[valid]))


def compute_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute Negative Log-Likelihood.

    NLL = -log(p(y_true))

    Args:
        logits: Model logits (N, C)
        labels: True labels (N,)

    Returns:
        Mean NLL
    """
    log_probs = F.log_softmax(logits, dim=-1)
    nll = F.nll_loss(log_probs, labels, reduction="mean")
    return nll.item()


def compute_brier_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute Brier Score.

    Brier = (1/C) Σ_c (p(c) - 1(y=c))²

    Args:
        logits: Model logits (N, C)
        labels: True labels (N,)

    Returns:
        Mean Brier score
    """
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(labels, num_classes=probs.shape[-1]).float()
    brier = ((probs - one_hot) ** 2).sum(dim=-1).mean()
    return brier.item()


def compute_auroc(
    in_scores: np.ndarray,
    out_scores: np.ndarray,
) -> float:
    """
    Compute AUROC for OOD detection.

    Uses 1-confidence as the anomaly score (lower confidence → more likely OOD).

    Args:
        in_scores: Confidence scores for in-distribution samples (N,)
        out_scores: Confidence scores for OOD samples (M,)

    Returns:
        AUROC value
    """
    # Higher anomaly score → more likely OOD
    anomaly_in = 1.0 - in_scores
    anomaly_out = 1.0 - out_scores

    labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    scores = np.concatenate([anomaly_in, anomaly_out])

    # Sort by score descending
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]

    # Compute ROC curve
    tpr_list, fpr_list = [0.0], [0.0]
    tp, fp = 0, 0
    total_pos = sorted_labels.sum()
    total_neg = len(sorted_labels) - total_pos

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / total_pos if total_pos > 0 else 0)
        fpr_list.append(fp / total_neg if total_neg > 0 else 0)

    # AUROC via trapezoidal rule
    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    auroc = np.trapz(tpr_arr, fpr_arr)

    return auroc


def compute_fpr_at_tpr(
    in_scores: np.ndarray,
    out_scores: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """
    Compute FPR at a given TPR (e.g., FPR@95TPR).

    Args:
        in_scores: Confidence scores for in-distribution samples
        out_scores: Confidence scores for OOD samples
        target_tpr: Target true positive rate

    Returns:
        FPR at target TPR
    """
    anomaly_in = 1.0 - in_scores
    anomaly_out = 1.0 - out_scores

    labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    scores = np.concatenate([anomaly_in, anomaly_out])

    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]

    total_pos = sorted_labels.sum()
    total_neg = len(sorted_labels) - total_pos
    tp, fp = 0, 0

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos if total_pos > 0 else 0
        if tpr >= target_tpr:
            return fp / total_neg if total_neg > 0 else 1.0

    return 1.0
