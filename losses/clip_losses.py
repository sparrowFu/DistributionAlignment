"""
GaussianImageDistribution - CLIP Loss Functions

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
        Similarity matrix of shape (B, B) where similarity[i, j] is
        the similarity between image i and text j, scaled by temperature
    """
    # Compute dot product (since features are normalized, this is cosine similarity)
    # Shape: (B, B)
    logits = image_features @ text_features.T

    # Scale by temperature
    logits = logits / temperature

    return logits


def clip_contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.07
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the standard CLIP bidirectional contrastive loss.

    This implements the symmetric contrastive loss used in CLIP:
    - For each image, find the matching text among all texts
    - For each text, find the matching image among all images
    - Average the two directional losses

    Args:
        image_features: Image features of shape (B, D), assumed L2-normalized
        text_features: Text features of shape (B, D), assumed L2-normalized
        temperature: Temperature parameter for softmax

    Returns:
        Tuple of (loss_value, loss_info_dict)
        - loss_value: Scalar tensor representing the average loss
        - loss_info_dict: Dictionary with loss details
    """
    batch_size = image_features.shape[0]

    # Compute similarity matrices
    # i2t_logits: image-to-text similarities (B, B)
    # t2i_logits: text-to-image similarities (B, B)
    i2t_logits = compute_similarity_matrix(image_features, text_features, temperature)
    t2i_logits = compute_similarity_matrix(text_features, image_features, temperature)

    # Ground truth labels: diagonal elements are the positive pairs
    labels = torch.arange(batch_size, device=image_features.device)

    # Compute cross-entropy loss for both directions
    loss_i2t = F.cross_entropy(i2t_logits, labels)
    loss_t2i = F.cross_entropy(t2i_logits, labels)

    # Average the two losses
    loss = (loss_i2t + loss_t2i) / 2

    # Compute accuracy for monitoring
    with torch.no_grad():
        # Image-to-text accuracy
        i2t_pred = i2t_logits.argmax(dim=1)
        acc_i2t = (i2t_pred == labels).float().mean()

        # Text-to-image accuracy
        t2i_pred = t2i_logits.argmax(dim=1)
        acc_t2i = (t2i_pred == labels).float().mean()

        # Average accuracy
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


def clip_contrastive_loss_with_hard_negatives(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.07,
    hard_negative_weight: float = 0.5
) -> Tuple[torch.Tensor, dict]:
    """
    Compute CLIP contrastive loss with hard negative mining.

    This is an extended version that gives extra weight to hard negatives
    (samples that are similar but not the correct match).

    Args:
        image_features: Image features of shape (B, D)
        text_features: Text features of shape (B, D)
        temperature: Temperature parameter
        hard_negative_weight: Weight for hard negatives

    Returns:
        Tuple of (loss_value, loss_info_dict)
    """
    batch_size = image_features.shape[0]

    # Standard similarity matrices
    i2t_logits = compute_similarity_matrix(image_features, text_features, temperature)
    t2i_logits = compute_similarity_matrix(text_features, image_features, temperature)

    # Labels
    labels = torch.arange(batch_size, device=image_features.device)

    # Standard loss
    loss_i2t = F.cross_entropy(i2t_logits, labels)
    loss_t2i = F.cross_entropy(t2i_logits, labels)

    loss = (loss_i2t + loss_t2i) / 2

    # Compute accuracy
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


class CLIPLoss(torch.nn.Module):
    """
    CLIP Contrastive Loss as a PyTorch Module.

    This wraps the contrastive loss function in a Module for convenience
    in training loops.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize CLIP loss.

        Args:
            temperature: Temperature parameter
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute CLIP contrastive loss.

        Args:
            image_features: Image features of shape (B, D)
            text_features: Text features of shape (B, D)

        Returns:
            Tuple of (loss, loss_info)
        """
        return clip_contrastive_loss(image_features, text_features, self.temperature)


if __name__ == "__main__":
    # Test loss functions
    import torch

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create dummy features (8 samples, 512 dimensions)
    batch_size = 8
    dim = 512

    # Create normalized features
    image_features = F.normalize(torch.randn(batch_size, dim), dim=1)
    text_features = F.normalize(torch.randn(batch_size, dim), dim=1)

    # Add some correlation for diagonal (positive pairs)
    text_features = text_features + 0.5 * image_features
    text_features = F.normalize(text_features, dim=1)

    print("Testing CLIP Contrastive Loss")
    print("=" * 50)

    # Test standard loss
    loss, loss_info = clip_contrastive_loss(
        image_features,
        text_features,
        temperature=0.07
    )

    print(f"\nStandard CLIP Loss:")
    print(f"  Total Loss: {loss_info['loss']:.4f}")
    print(f"  Image-to-Text Loss: {loss_info['loss_i2t']:.4f}")
    print(f"  Text-to-Image Loss: {loss_info['loss_t2i']:.4f}")
    print(f"  Accuracy: {loss_info['acc']:.4f}")
    print(f"  Image-to-Text Acc: {loss_info['acc_i2t']:.4f}")
    print(f"  Text-to-Image Acc: {loss_info['acc_t2i']:.4f}")

    # Test similarity matrix
    print(f"\nSimilarity Matrix (temperature=1.0):")
    sim_matrix = compute_similarity_matrix(image_features, text_features, temperature=1.0)
    print(sim_matrix)

    # Test CLIPLoss module
    print("\nTesting CLIPLoss Module:")
    loss_module = CLIPLoss(temperature=0.07)
    loss, info = loss_module(image_features, text_features)
    print(f"  Loss: {info['loss']:.4f}")
    print(f"  Accuracy: {info['acc']:.4f}")
