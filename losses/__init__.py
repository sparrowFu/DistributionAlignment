"""
GaussianImageDistribution - Loss Functions

This package provides loss functions for CLIP training and
distribution-based alignment.
"""

from .clip_losses import clip_contrastive_loss
from .dist_align_losses import (
    DistributionAlignmentLoss,
    CombinedDistributionLoss,
    DistributionalContrastiveLoss,
    UncertaintyCalibratedContrastiveLoss
)

__all__ = [
    "clip_contrastive_loss",
    "DistributionAlignmentLoss",
    "CombinedDistributionLoss",
    "DistributionalContrastiveLoss",
    "UncertaintyCalibratedContrastiveLoss"
]
