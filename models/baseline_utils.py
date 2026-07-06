"""
GaussianImageDistribution - Common utilities for baseline models.

Shared functions used by ProLIP, GroVE baseline models.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

import config
from utils.logger import get_logger


logger = get_logger("baseline_utils")


def merge_distributions_moment_matching(
    mus: torch.Tensor,
    logvars: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge K Gaussian distributions via moment matching.

    Args:
        mus: Distribution means (B, K, D)
        logvars: Distribution log variances (B, K, D)

    Returns:
        (combined_mu, combined_logvar) each (B, D)
    """
    combined_mu = mus.mean(dim=1)
    vars = torch.exp(logvars)
    combined_var = (vars + mus ** 2).mean(dim=1) - combined_mu ** 2
    combined_logvar = torch.log(combined_var + 1e-6)
    return combined_mu, combined_logvar


def encode_clip_features(
    clip_model: CLIPModel,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Encode image and K text captions via CLIP.

    Args:
        clip_model: HuggingFace CLIPModel instance
        pixel_values: Image tensor (B, C, H, W)
        input_ids: Text token IDs (B, K, max_len)
        attention_mask: Text attention mask (B, K, max_len)

    Returns:
        (img_features, text_features, B, K)
        - img_features: (B, D)
        - text_features: (B, K, D)
    """
    img_features = clip_model.get_image_features(pixel_values)
    img_features = img_features.pooler_output

    B, K, max_len = input_ids.shape
    input_ids_flat = input_ids.view(B * K, max_len)
    attention_mask_flat = attention_mask.view(B * K, max_len)

    text_features = clip_model.get_text_features(
        input_ids=input_ids_flat, attention_mask=attention_mask_flat
    )
    text_features = text_features.pooler_output
    text_features = text_features.view(B, K, -1)

    return img_features, text_features, B, K


def init_heads_xavier(heads):
    """Initialize MLP heads with Xavier initialization."""
    for head in heads:
        for layer in head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
