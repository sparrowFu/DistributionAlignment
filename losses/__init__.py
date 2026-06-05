"""
GaussianImageDistribution - Loss Functions

This package provides loss functions for CLIP training and
distribution-based alignment.
"""

from .clip_losses import clip_contrastive_loss, compute_similarity_matrix
from .dist_align_losses import (
    DistributionAlignmentLoss,
    VarianceRegularizationLoss,
    CombinedDistributionLoss
)

__all__ = [
    "clip_contrastive_loss",
    "compute_similarity_matrix",
    "DistributionAlignmentLoss",
    "VarianceRegularizationLoss",
    "CombinedDistributionLoss"
]
