"""
GaussianImageDistribution - Data Modules

This package provides dataset classes for image-text pair loading
and VQA (Visual Question Answering) data loading.
"""

from .caption_dataset import ImageCaptionDataset
from .vqa_dataset import VQADataset

__all__ = ["ImageCaptionDataset", "VQADataset"]
