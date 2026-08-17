"""
CLIP Loss Functions

This module implements the standard CLIP contrastive loss for
image-text representation learning.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_similarity_matrix(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute similarity matrix between image and text features.

    Args:
        image_features: Image features of shape (B, D), assumed L2-normalized
        text_features: Text features of shape (B, D), assumed L2-normalized
        temperature: Temperature scaling parameter

    Returns:
        Similarity matrix of shape (B, B)
    """
    logits = image_features @ text_features.T
    logits = logits / temperature
    return logits


def clip_contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.07
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the standard CLIP bidirectional contrastive loss.

    Args:
        image_features: Image features of shape (B, D), assumed L2-normalized
        text_features: Text features of shape (B, D), assumed L2-normalized
        temperature: Temperature parameter for softmax

    Returns:
        Tuple of (loss_value, loss_info_dict)
    """
    batch_size = image_features.shape[0]

    i2t_logits = compute_similarity_matrix(image_features, text_features, temperature)
    t2i_logits = compute_similarity_matrix(text_features, image_features, temperature)

    labels = torch.arange(batch_size, device=image_features.device)

    loss_i2t = F.cross_entropy(i2t_logits, labels)
    loss_t2i = F.cross_entropy(t2i_logits, labels)

    loss = (loss_i2t + loss_t2i) / 2

    with torch.no_grad():
        i2t_pred = i2t_logits.argmax(dim=1)
        acc_i2t = (i2t_pred == labels).float().mean()

        t2i_pred = t2i_logits.argmax(dim=1)
        acc_t2i = (t2i_pred == labels).float().mean()

        acc = (acc_i2t + acc_t2i) / 2

    loss_info = {
        "loss": loss.item(),
        "loss_i2t": loss_i2t.item(),
        "loss_t2i": loss_t2i.item(),
        "acc": acc.item(),
        "acc_i2t": acc_i2t.item(),
        "acc_t2i": acc_t2i.item(),
    }

    return loss, loss_info
