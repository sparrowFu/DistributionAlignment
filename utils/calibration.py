"""
GaussianImageDistribution - Uncertainty Calibration Metrics (Exp3)

Implements OOD-detection scoring metrics used by Exp4 (eval_ood.py):
    - AUROC (using 1-confidence as OOD score)
    - FPR@TPR
"""

import numpy as np

from utils.logger import get_logger


logger = get_logger("calibration")


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
