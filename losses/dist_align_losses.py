"""
GaussianImageDistribution - Distribution Alignment Loss Functions

This module implements loss functions for distribution-based alignment,
including CLIP contrastive loss, KL divergence loss, and variance regularization.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from utils.logger import get_logger


logger = get_logger("dist_align_losses")


class DistributionAlignmentLoss(nn.Module):
    """
    Combined loss for distribution alignment.

    Total loss = λ_contrastive * L_contrastive + λ_kl * L_kl

    This combines:
    - CLIP-style contrastive loss for global feature alignment
    - KL divergence loss for distribution shape alignment
    """

    def __init__(
        self,
        lambda_contrastive: float = 1.0,
        lambda_kl: float = 0.5,
        temperature: float = 0.07,
        kl_type: str = "symmetric"
    ):
        """
        Initialize distribution alignment loss.

        Args:
            lambda_contrastive: Weight for contrastive loss
            lambda_kl: Weight for KL divergence loss
            temperature: Temperature parameter for contrastive loss
            kl_type: Type of KL divergence ("symmetric", "forward", "reverse", "wasserstein")
        """
        super().__init__()

        self.lambda_contrastive = lambda_contrastive
        self.lambda_kl = lambda_kl
        self.temperature = temperature
        self.kl_type = kl_type

    def clip_contrastive_loss(
        self,
        img_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CLIP-style contrastive loss (InfoNCE).

        Args:
            img_features: Image features of shape (B, D)
            text_features: Text features of shape (B, D)

        Returns:
            Contrastive loss scalar
        """
        B = img_features.shape[0]

        # L2 normalize features
        img_features = img_features / img_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Compute similarity matrix
        logits = torch.matmul(img_features, text_features.T) / self.temperature  # (B, B)

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(B, device=img_features.device)

        # Bidirectional contrastive loss
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)

        loss = (loss_i + loss_t) / 2
        return loss

    def kl_divergence(
        self,
        mu1: torch.Tensor,
        logvar1: torch.Tensor,
        mu2: torch.Tensor,
        logvar2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence KL(N1 || N2).

        Args:
            mu1, logvar1: Parameters of first distribution, shape (B, D)
            mu2, logvar2: Parameters of second distribution, shape (B, D)

        Returns:
            KL divergence per sample, shape (B,)
        """
        sigma1_sq = torch.exp(logvar1) + 1e-6
        sigma2_sq = torch.exp(logvar2) + 1e-6

        kl = (torch.log(sigma2_sq / sigma1_sq) +
              (sigma1_sq + (mu1 - mu2) ** 2) / (2 * sigma2_sq) -
              0.5)

        return kl.mean(dim=-1)  # (B,) average over dimensions

    def symmetric_kl(
        self,
        mu1: torch.Tensor,
        logvar1: torch.Tensor,
        mu2: torch.Tensor,
        logvar2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute symmetric KL divergence: KL(P||Q) + KL(Q||P).

        Args:
            mu1, logvar1: Parameters of first distribution, shape (B, D)
            mu2, logvar2: Parameters of second distribution, shape (B, D)

        Returns:
            Symmetric KL divergence per sample, shape (B,)
        """
        kl_pq = self.kl_divergence(mu1, logvar1, mu2, logvar2)
        kl_qp = self.kl_divergence(mu2, logvar2, mu1, logvar1)
        return kl_pq + kl_qp

    def wasserstein_distance(
        self,
        mu1: torch.Tensor,
        logvar1: torch.Tensor,
        mu2: torch.Tensor,
        logvar2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Wasserstein-2 distance between two Gaussian distributions.

        Args:
            mu1, logvar1: Parameters of first distribution, shape (B, D)
            mu2, logvar2: Parameters of second distribution, shape (B, D)

        Returns:
            Wasserstein distance per sample, shape (B,)
        """
        sigma1 = torch.exp(0.5 * logvar1)
        sigma2 = torch.exp(0.5 * logvar2)

        # Wasserstein-2 distance formula
        diff_mu = (mu1 - mu2) ** 2
        diff_sigma = (sigma1 - sigma2) ** 2

        distance = torch.sqrt((diff_mu + diff_sigma).mean(dim=-1) + 1e-6)
        return distance

    def forward(
        self,
        img_features: torch.Tensor,
        text_features: torch.Tensor,
        img_mu: torch.Tensor,
        img_logvar: torch.Tensor,
        text_mu: torch.Tensor,
        text_logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            img_features: CLIP image features, shape (B, D)
            text_features: CLIP text features, shape (B, D)
            img_mu, img_logvar: Image distribution parameters, shape (B, D)
            text_mu, text_logvar: Text distribution parameters, shape (B, D)

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
        """
        # CLIP contrastive loss
        contrastive_loss = self.clip_contrastive_loss(img_features, text_features)

        # KL divergence loss
        if self.kl_type == "symmetric":
            kl_loss = self.symmetric_kl(img_mu, img_logvar, text_mu, text_logvar)
        elif self.kl_type == "forward":
            kl_loss = self.kl_divergence(img_mu, img_logvar, text_mu, text_logvar)
        elif self.kl_type == "reverse":
            kl_loss = self.kl_divergence(text_mu, text_logvar, img_mu, img_logvar)
        elif self.kl_type == "wasserstein":
            kl_loss = self.wasserstein_distance(img_mu, img_logvar, text_mu, text_logvar)
        else:
            raise ValueError(f"Unknown KL type: {self.kl_type}")

        kl_loss = kl_loss.mean()

        # Total loss
        total_loss = (
            self.lambda_contrastive * contrastive_loss +
            self.lambda_kl * kl_loss
        )

        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'contrastive': contrastive_loss.item(),
            'kl': kl_loss.item(),
        }

        return total_loss, loss_dict


class VarianceRegularizationLoss(nn.Module):
    """
    Variance regularization loss to prevent distributions from becoming too narrow or wide.
    """

    def __init__(self, target_variance: float = 0.5, lambda_var: float = 0.1):
        """
        Initialize variance regularization loss.

        Args:
            target_variance: Target variance value
            lambda_var: Weight for variance loss
        """
        super().__init__()

        self.target_variance = target_variance
        self.lambda_var = lambda_var

    def forward(self, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute variance regularization loss.

        Args:
            logvar: Log variance of shape (B, D)

        Returns:
            Variance loss scalar
        """
        variance = torch.exp(logvar)
        var_loss = F.mse_loss(variance, torch.ones_like(variance) * self.target_variance)
        return self.lambda_var * var_loss


class CombinedDistributionLoss(nn.Module):
    """
    Complete distribution alignment loss with variance regularization.

    Total loss = λ_contrastive * L_contrastive + λ_kl * L_kl + λ_var * L_var
    """

    def __init__(
        self,
        lambda_contrastive: float = 1.0,
        lambda_kl: float = 0.5,
        lambda_var: float = 0.1,
        temperature: float = 0.07,
        kl_type: str = "symmetric",
        target_variance: float = 0.5
    ):
        """
        Initialize combined distribution loss.

        Args:
            lambda_contrastive: Weight for contrastive loss
            lambda_kl: Weight for KL divergence loss
            lambda_var: Weight for variance regularization loss
            temperature: Temperature parameter for contrastive loss
            kl_type: Type of KL divergence
            target_variance: Target variance for regularization
        """
        super().__init__()

        self.dist_loss = DistributionAlignmentLoss(
            lambda_contrastive=lambda_contrastive,
            lambda_kl=lambda_kl,
            temperature=temperature,
            kl_type=kl_type
        )

        self.var_loss = VarianceRegularizationLoss(
            target_variance=target_variance,
            lambda_var=lambda_var
        )

    def forward(
        self,
        img_features: torch.Tensor,
        text_features: torch.Tensor,
        img_mu: torch.Tensor,
        img_logvar: torch.Tensor,
        text_mu: torch.Tensor,
        text_logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            img_features: CLIP image features, shape (B, D)
            text_features: CLIP text features, shape (B, D)
            img_mu, img_logvar: Image distribution parameters, shape (B, D)
            text_mu, text_logvar: Text distribution parameters, shape (B, D)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Distribution alignment loss
        main_loss, loss_dict = self.dist_loss(
            img_features, text_features,
            img_mu, img_logvar,
            text_mu, text_logvar
        )

        # Variance regularization
        img_var_loss = self.var_loss(img_logvar)
        text_var_loss = self.var_loss(text_logvar)
        var_loss = img_var_loss + text_var_loss

        # Total loss
        total_loss = main_loss + var_loss

        # Update loss dictionary
        loss_dict['total'] = total_loss.item()
        loss_dict['variance'] = var_loss.item()

        return total_loss, loss_dict


if __name__ == "__main__":
    # Test loss functions
    print("Testing distribution alignment loss functions...")

    B, D = 4, 768

    # Create dummy outputs
    img_features = torch.randn(B, D)
    text_features = torch.randn(B, D)
    img_mu = torch.randn(B, D)
    img_logvar = torch.randn(B, D)
    text_mu = torch.randn(B, D)
    text_logvar = torch.randn(B, D)

    # Test DistributionAlignmentLoss
    print("\n1. Testing DistributionAlignmentLoss:")
    criterion = DistributionAlignmentLoss(
        lambda_contrastive=1.0,
        lambda_kl=0.5,
        kl_type="symmetric"
    )
    loss, loss_dict = criterion(
        img_features, text_features,
        img_mu, img_logvar, text_mu, text_logvar
    )
    print(f"   Total loss: {loss_dict['total']:.4f}")
    print(f"   Contrastive loss: {loss_dict['contrastive']:.4f}")
    print(f"   KL loss: {loss_dict['kl']:.4f}")

    # Test VarianceRegularizationLoss
    print("\n2. Testing VarianceRegularizationLoss:")
    var_criterion = VarianceRegularizationLoss(target_variance=0.5, lambda_var=0.1)
    var_loss = var_criterion(img_logvar)
    print(f"   Variance loss: {var_loss.item():.4f}")

    # Test CombinedDistributionLoss
    print("\n3. Testing CombinedDistributionLoss:")
    combined_criterion = CombinedDistributionLoss(
        lambda_contrastive=1.0,
        lambda_kl=0.5,
        lambda_var=0.1,
        kl_type="symmetric"
    )
    combined_loss, combined_dict = combined_criterion(
        img_features, text_features,
        img_mu, img_logvar, text_mu, text_logvar
    )
    print(f"   Total loss: {combined_dict['total']:.4f}")
    print(f"   Contrastive loss: {combined_dict['contrastive']:.4f}")
    print(f"   KL loss: {combined_dict['kl']:.4f}")
    print(f"   Variance loss: {combined_dict.get('variance', 0):.4f}")

    print("\nAll loss functions tested successfully!")
