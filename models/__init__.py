"""
GaussianImageDistribution - Model Modules

This package provides model definitions for distribution-based image-text
alignment (MSDA) and comparison baselines.

Experiment baselines:
    B1 CLIP Zero-Shot  -- clip_zero_shot.py
    B2 CLIP Fine-Tune  -- clip_baseline.py
    B3 ProLIP          -- prolip_model.py
    B4 GroVE           -- grove_model.py
    Ours (MSDA)        -- dist_align_model.py
"""

from .clip_baseline import CLIPFineTuneBaseline
from .dist_align_model import DistributionAlignmentModel
from .clip_zero_shot import CLIPZeroShotVQA
from .vqa_model import VQAModel
from .prolip_model import ProLIPModel
from .grove_model import GroVEModel

__all__ = [
    "CLIPFineTuneBaseline",
    "DistributionAlignmentModel",
    "CLIPZeroShotVQA",
    "VQAModel",
    "ProLIPModel",
    "GroVEModel",
]
