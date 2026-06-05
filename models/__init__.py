"""
GaussianImageDistribution - Model Modules

This package provides model definitions for CLIP fine-tuning and
distribution-based image-text alignment.
"""

from .clip_baseline import CLIPFineTuneBaseline
from .dist_align_model import DistributionAlignmentModel

__all__ = ["CLIPFineTuneBaseline", "DistributionAlignmentModel"]
