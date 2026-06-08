"""
GaussianImageDistribution - Model Modules

This package provides model definitions for CLIP fine-tuning,
distribution-based image-text alignment, and comparison baselines.
"""

from .clip_baseline import CLIPFineTuneBaseline
from .dist_align_model import DistributionAlignmentModel
from .freeze_align_model import FreezeAlignModel
from .fate_model import FATEModel
from .clip_ast_model import CLIPASTModel
from .clip_zero_shot import CLIPZeroShotVQA
from .vqa_model import VQAModel

__all__ = [
    "CLIPFineTuneBaseline",
    "DistributionAlignmentModel",
    "FreezeAlignModel",
    "FATEModel",
    "CLIPASTModel",
    "CLIPZeroShotVQA",
    "VQAModel",
]
