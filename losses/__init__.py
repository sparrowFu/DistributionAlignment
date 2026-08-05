"""
GaussianImageDistribution - Loss Functions

This package provides loss functions for CLIP training and
distribution-based alignment.
"""

from .clip_losses import clip_contrastive_loss
from .mcdisp_align_losses import (
    MCDispAlignLoss,
)

__all__ = [
    "clip_contrastive_loss",
    "MCDispAlignLoss",
]
