"""
GaussianImageDistribution - Model Modules

This package provides model definitions for distribution-based image-text
alignment (MSDA) and comparison baselines.

Experiment baselines:
    B2 CLIP Fine-Tune  -- clip_baseline.py
    B3 ProLIP          -- prolip_model.py
    Ours (MSDA)        -- dist_align_model.py

(The B1 CLIP Zero-Shot baseline is eval-time only -- see
 scripts/evaluate_clip_zero_shot.py; it has no trainable model class here.)
"""

from .clip_baseline import CLIPFineTuneBaseline
from .dist_align_model import DistributionAlignmentModel
from .prolip_model import ProLIPModel

__all__ = [
    "CLIPFineTuneBaseline",
    "DistributionAlignmentModel",
    "ProLIPModel",
]
