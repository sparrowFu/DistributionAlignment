"""
GaussianImageDistribution - Data Modules

This package provides dataset classes for image-text pair loading,
Flickr30K cross-dataset evaluation, and VQA-as-retrieval expansion data.
"""

from .caption_dataset import ImageCaptionDataset
from .flickr30k_dataset import Flickr30KDataset

__all__ = ["ImageCaptionDataset", "Flickr30KDataset"]
