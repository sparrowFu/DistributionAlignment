"""
Model Modules

This package provides model definitions for distribution-based image-text
alignment (MCDisp_Align) and comparison baselines.

Experiment baselines:
    B2 CLIP Fine-Tune  -- clip_baseline.py
    B3 ProLIP          -- prolip_model.py
    B4 ProbVLM         -- probvlm_model.py
    B5 PCME++          -- pcmepp_model.py (wraps the vendored pcmepp/ package)
    Ours (MCDisp_Align)        -- mcdisp_align_model.py

(The B1 CLIP Zero-Shot baseline is eval-time only; it has no trainable model class here.)
"""

from .clip_baseline import CLIPFineTuneBaseline
from .mcdisp_align_model import MCDispAlignModel
from .pcmepp_model import PCMEPPModel
from .probvlm_model import ProbVLMModel
from .prolip_model import ProLIPModel

__all__ = [
    "CLIPFineTuneBaseline",
    "MCDispAlignModel",
    "PCMEPPModel",
    "ProbVLMModel",
    "ProLIPModel",
]
