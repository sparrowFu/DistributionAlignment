"""
GaussianImageDistribution - Model Modules

This package provides model definitions for distribution-based image-text
alignment (UC-CL) and comparison baselines.

Experiment baselines:
    B1 CLIP Zero-Shot  -- clip_zero_shot.py
    B2 CLIP Fine-Tune  -- clip_baseline.py
    B3 ProLIP          -- prolip_model.py
    B4 GroVE           -- grove_model.py
    B5 ICPE            -- icpe_model.py
    B6 D2P             -- d2p_model.py
    Ours (UC-CL)       -- dist_align_model.py
"""

from .clip_baseline import CLIPFineTuneBaseline
from .dist_align_model import DistributionAlignmentModel
from .clip_zero_shot import CLIPZeroShotVQA
from .vqa_model import VQAModel
from .prolip_model import ProLIPModel
from .grove_model import GroVEModel
from .icpe_model import ICPEModel
from .d2p_model import D2PModel

__all__ = [
    "CLIPFineTuneBaseline",
    "DistributionAlignmentModel",
    "CLIPZeroShotVQA",
    "VQAModel",
    "ProLIPModel",
    "GroVEModel",
    "ICPEModel",
    "D2PModel",
]
