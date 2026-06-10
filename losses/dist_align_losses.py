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


class DistributionalContrastiveLoss(nn.Module):
    """
    Distributional Contrastive Learning via Optimal Transport.

    Replaces point-level contrastive similarity with distribution-level
    similarity based on Wasserstein-2 distance between Gaussian distributions.

    Key idea:
        Standard CLIP:  sim(x, y) = x · y^T / τ
        This method:    sim(N_x, N_y) = exp(-W_2(N_x, N_y) / τ)

    For diagonal Gaussians, W_2 has a closed-form solution:
        W_2^2 = ||μ_1 - μ_2||^2 + ||σ_1 - σ_2||^2

    The distributional similarity naturally encodes:
    - Mean alignment (via μ distance)
    - Uncertainty matching (via σ distance)
    - One-to-many relationships (via σ magnitude)
    """

    def __init__(
        self,
        lambda_ot: float = 1.0,
        temperature: float = 0.1,
        min_sigma: float = 1e-3,
        target_variance: float = 0.5,
        lambda_var: float = 0.1,
    ):
        """
        Initialize distributional contrastive loss.

        Args:
            lambda_ot: Weight for distributional contrastive loss
            temperature: Temperature for similarity (τ), controls sharpness
            min_sigma: Minimum sigma to prevent numerical collapse
            target_variance: Target variance for regularization
            lambda_var: Weight for variance regularization
        """
        super().__init__()

        self.lambda_ot = lambda_ot
        self.temperature = temperature
        self.min_sigma = min_sigma
        self.target_variance = target_variance
        self.lambda_var = lambda_var

    def compute_w2_squared_matrix(
        self,
        mu1: torch.Tensor,
        logvar1: torch.Tensor,
        mu2: torch.Tensor,
        logvar2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise W_2^2 distance matrix between two sets of distributions.

        For diagonal Gaussians:
            W_2^2(N(μ₁,Σ₁), N(μ₂,Σ₂)) = ||μ₁ - μ₂||² + ||σ₁ - σ₂||²

        Args:
            mu1: Mean of first distributions, shape (B1, D)
            logvar1: Log variance of first distributions, shape (B1, D)
            mu2: Mean of second distributions, shape (B2, D)
            logvar2: Log variance of second distributions, shape (B2, D)

        Returns:
            W_2^2 distance matrix of shape (B1, B2)
        """
        sigma1 = torch.exp(0.5 * logvar1).clamp(min=self.min_sigma)  # (B1, D)
        sigma2 = torch.exp(0.5 * logvar2).clamp(min=self.min_sigma)  # (B2, D)

        # ||μ₁ - μ₂||²: (B1, B2) via broadcasting
        # ||μ₁ - μ₂||² = ||μ₁||² + ||μ₂||² - 2 μ₁·μ₂ᵀ
        mu1_sq = (mu1 ** 2).sum(dim=-1, keepdim=True)  # (B1, 1)
        mu2_sq = (mu2 ** 2).sum(dim=-1, keepdim=True)  # (B2, 1)
        mu_cross = torch.matmul(mu1, mu2.T)  # (B1, B2)
        dist_mu_sq = mu1_sq + mu2_sq.T - 2 * mu_cross  # (B1, B2)
        dist_mu_sq = dist_mu_sq.clamp(min=0.0)

        # ||σ₁ - σ₂||²: (B1, B2) via broadcasting
        # ||σ₁ - σ₂||² = ||σ₁||² + ||σ₂||² - 2 σ₁·σ₂ᵀ
        s1_sq = (sigma1 ** 2).sum(dim=-1, keepdim=True)  # (B1, 1)
        s2_sq = (sigma2 ** 2).sum(dim=-1, keepdim=True)  # (B2, 1)
        s_cross = torch.matmul(sigma1, sigma2.T)  # (B1, B2)
        dist_sigma_sq = s1_sq + s2_sq.T - 2 * s_cross  # (B1, B2)
        dist_sigma_sq = dist_sigma_sq.clamp(min=0.0)

        # Total W_2^2
        w2_sq = dist_mu_sq + dist_sigma_sq  # (B1, B2)

        return w2_sq

    def distributional_infonce(
        self,
        img_mu: torch.Tensor,
        img_logvar: torch.Tensor,
        text_mu: torch.Tensor,
        text_logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distributional InfoNCE loss based on W_2 distance.

        sim(N_i, N_j) = exp(-W_2^2(N_i, N_j) / τ)

        L = -½ [ log exp(-W_2^2(f_x(i), f_y(i)) / τ) / Σ_j exp(-W_2^2(f_x(i), f_y(j)) / τ) ]
            -½ [ log exp(-W_2^2(f_y(i), f_x(i)) / τ) / Σ_j exp(-W_2^2(f_y(i), f_x(j)) / τ) ]

        Args:
            img_mu, img_logvar: Image distribution params, shape (B, D)
            text_mu, text_logvar: Text distribution params, shape (B, D)

        Returns:
            Tuple of (loss, info_dict)
        """
        B = img_mu.shape[0]

        # Compute W_2^2 distance matrix: (B, B)
        w2_sq = self.compute_w2_squared_matrix(
            img_mu, img_logvar, text_mu, text_logvar
        )

        # Distributional similarity: sim = exp(-W_2^2 / τ)
        # Use negative W_2^2 as logits for cross entropy
        logits = -w2_sq / self.temperature  # (B, B)

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(B, device=img_mu.device)

        # Bidirectional contrastive loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2

        # Compute average W_2 distance for positive pairs (diagonal)
        with torch.no_grad():
            pos_w2 = w2_sq[torch.arange(B), torch.arange(B)]
            avg_w2_pos = pos_w2.mean().sqrt().item()
            avg_w2_all = w2_sq.mean().sqrt().item()

        info = {
            'loss_i2t': loss_i2t.item(),
            'loss_t2i': loss_t2i.item(),
            'avg_w2_pos': avg_w2_pos,
            'avg_w2_all': avg_w2_all,
        }

        return loss, info

    def variance_regularization(
        self,
        img_logvar: torch.Tensor,
        text_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Variance regularization to prevent distribution collapse.

        Prevents σ from collapsing to 0 (trivial mapping) or
        exploding to infinity.

        Args:
            img_logvar: Image log variance, shape (B, D)
            text_logvar: Text log variance, shape (B, D)

        Returns:
            Variance regularization loss scalar
        """
        img_var = torch.exp(img_logvar)
        text_var = torch.exp(text_logvar)

        img_var_loss = F.mse_loss(img_var, torch.ones_like(img_var) * self.target_variance)
        text_var_loss = F.mse_loss(text_var, torch.ones_like(text_var) * self.target_variance)

        return self.lambda_var * (img_var_loss + text_var_loss)

    def forward(
        self,
        img_features: torch.Tensor,
        text_features: torch.Tensor,
        img_mu: torch.Tensor,
        img_logvar: torch.Tensor,
        text_mu: torch.Tensor,
        text_logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total distributional contrastive loss.

        L_total = λ_ot · L_dist_contrastive + λ_var · L_variance

        Args:
            img_features: CLIP image features, shape (B, D) [unused, kept for API compat]
            text_features: CLIP text features, shape (B, D) [unused, kept for API compat]
            img_mu, img_logvar: Image distribution parameters, shape (B, D)
            text_mu, text_logvar: Text distribution parameters, shape (B, D)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Distributional contrastive loss
        ot_loss, ot_info = self.distributional_infonce(
            img_mu, img_logvar, text_mu, text_logvar
        )

        # Variance regularization
        var_loss = self.variance_regularization(img_logvar, text_logvar)

        # Total loss
        total_loss = self.lambda_ot * ot_loss + var_loss

        loss_dict = {
            'total': total_loss.item(),
            'contrastive': ot_loss.item(),
            'kl': var_loss.item(),  # reuse key name for compatibility with training loop
            'avg_w2_pos': ot_info['avg_w2_pos'],
            'avg_w2_all': ot_info['avg_w2_all'],
            'variance': var_loss.item(),
        }

        return total_loss, loss_dict


class UncertaintyCalibratedContrastiveLoss(nn.Module):
    """
    Uncertainty-Calibrated Distributional Contrastive Learning.

    Three components:
        1. Uncertainty-Calibrated Similarity: σ modulates similarity sharpness
           sim(x,y) = μ_x · μ_y / (τ · √(1 + ‖σ_x‖²) · √(1 + ‖σ_y‖²))

        2. Distributional Consistency: σ²_img must match Var(μ_text_1,...,μ_text_K)
           L_consist = MSE(σ²_img, Var_{k=1..K}(μ_text_k))

        3. Variance regularization to prevent collapse

    Total loss:
        L_total = λ_cl · L_calibrated_cl + λ_consist · L_consist + λ_var · L_var
    """

    def __init__(
        self,
        lambda_cl: float = 1.0,
        lambda_consist: float = 1.0,
        lambda_var: float = 0.1,
        temperature: float = 0.07,
        target_variance: float = 0.5,
    ):
        """
        Initialize Uncertainty-Calibrated Contrastive Loss.

        Args:
            lambda_cl: Weight for uncertainty-calibrated contrastive loss (λ_cl)
            lambda_consist: Weight for distributional consistency loss (λ_consist)
            lambda_var: Weight for variance regularization (λ_var)
            temperature: Temperature parameter τ for similarity scaling
            target_variance: Target variance for regularization
        """
        super().__init__()

        self.lambda_cl = lambda_cl
        self.lambda_consist = lambda_consist
        self.lambda_var = lambda_var
        self.temperature = temperature
        self.target_variance = target_variance

    def uncertainty_calibrated_similarity(
        self,
        img_mu: torch.Tensor,
        img_logvar: torch.Tensor,
        text_mu: torch.Tensor,
        text_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute uncertainty-calibrated similarity matrix.

        sim(x,y) = μ_x · μ_y / (τ · √(1 + ‖σ_x‖²) · √(1 + ‖σ_y‖²))

        Intuition:
            - High uncertainty (large σ) → similarity is flattened → softer gradients
            - Low uncertainty (small σ) → similarity is sharper → more confident gradients
            - Does NOT force σ_img ≈ σ_text, instead lets σ modulate sharpness

        Args:
            img_mu: Image distribution mean, shape (B, D)
            img_logvar: Image distribution log variance, shape (B, D)
            text_mu: Text distribution mean, shape (B, D)
            text_logvar: Text distribution log variance, shape (B, D)

        Returns:
            Similarity matrix of shape (B, B)
        """
        # Compute average variance per sample (mean over dimensions, not sum)
        # Using mean keeps the scale factor dimension-independent:
        #   with logvar~0: avg_var=1, scale≈1.41  (reasonable)
        #   with sum:      total_var=768, scale≈27.7 (kills gradients)
        img_var_avg = torch.exp(img_logvar).mean(dim=-1)  # (B,)
        text_var_avg = torch.exp(text_logvar).mean(dim=-1)  # (B,)

        # Uncertainty scaling factors: √(1 + avg_σ²)
        img_scale = torch.sqrt(1.0 + img_var_avg)  # (B,)
        text_scale = torch.sqrt(1.0 + text_var_avg)  # (B,)

        # L2 normalize means for cosine-like similarity
        img_mu_norm = F.normalize(img_mu, dim=-1)  # (B, D)
        text_mu_norm = F.normalize(text_mu, dim=-1)  # (B, D)

        # Compute dot product of means: (B, B)
        mean_similarity = torch.matmul(img_mu_norm, text_mu_norm.T)

        # Apply uncertainty calibration: divide by (τ · scale_img · scale_text)
        # Outer product of scales: (B, B)
        scale_matrix = img_scale.unsqueeze(1) * text_scale.unsqueeze(0)  # (B, B)
        calibrated_sim = mean_similarity / (self.temperature * scale_matrix)  # (B, B)

        return calibrated_sim

    def calibrated_infonce(
        self,
        img_mu: torch.Tensor,
        img_logvar: torch.Tensor,
        text_mu: torch.Tensor,
        text_logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute InfoNCE loss with uncertainty-calibrated similarity.

        Args:
            img_mu, img_logvar: Image distribution params, shape (B, D)
            text_mu, text_logvar: Text distribution params, shape (B, D)

        Returns:
            Tuple of (loss, info_dict)
        """
        B = img_mu.shape[0]

        # Uncertainty-calibrated similarity matrix
        logits = self.uncertainty_calibrated_similarity(
            img_mu, img_logvar, text_mu, text_logvar
        )

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(B, device=img_mu.device)

        # Bidirectional contrastive loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2

        # Logging info
        with torch.no_grad():
            pos_sim = logits[torch.arange(B), torch.arange(B)].mean().item()
            img_var_avg = torch.exp(img_logvar).mean(dim=-1).mean().item()
            text_var_avg = torch.exp(text_logvar).mean(dim=-1).mean().item()

        info = {
            'loss_i2t': loss_i2t.item(),
            'loss_t2i': loss_t2i.item(),
            'avg_pos_sim': pos_sim,
            'img_var_avg': img_var_avg,
            'text_var_avg': text_var_avg,
        }

        return loss, info

    def distributional_consistency_loss(
        self,
        img_logvar: torch.Tensor,
        text_mus: torch.Tensor,
    ) -> torch.Tensor:
        """
        Distributional Consistency Loss (Core Innovation).

        L_consist = MSE(σ²_img, Var_{k=1..K}(μ_text_k))

        This constrains the image distribution's variance to equal the variance
        of its K caption means, giving σ² a clear, verifiable semantic meaning:
        "the image's semantic uncertainty equals the diversity of its descriptions."

        Args:
            img_logvar: Image distribution log variance, shape (B, D)
            text_mus: Individual caption distribution means, shape (B, K, D)
                      where K is the number of captions per image

        Returns:
            Consistency loss scalar
        """
        # σ²_img: (B, D)
        img_var = torch.exp(img_logvar)

        # Var_{k=1..K}(μ_text_k): variance across K captions, shape (B, D)
        # Var = E[μ²] - E[μ]²
        caption_var = text_mus.var(dim=1)  # (B, D)

        # MSE loss between image variance and caption mean variance
        consist_loss = F.mse_loss(img_var, caption_var)

        return consist_loss

    def variance_regularization(
        self,
        img_logvar: torch.Tensor,
        text_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Variance regularization to prevent distribution collapse.

        Args:
            img_logvar: Image log variance, shape (B, D)
            text_logvar: Text log variance, shape (B, D)

        Returns:
            Variance regularization loss scalar
        """
        img_var = torch.exp(img_logvar)
        text_var = torch.exp(text_logvar)

        img_var_loss = F.mse_loss(img_var, torch.ones_like(img_var) * self.target_variance)
        text_var_loss = F.mse_loss(text_var, torch.ones_like(text_var) * self.target_variance)

        return img_var_loss + text_var_loss

    def forward(
        self,
        img_features: torch.Tensor,
        text_features: torch.Tensor,
        img_mu: torch.Tensor,
        img_logvar: torch.Tensor,
        text_mu: torch.Tensor,
        text_logvar: torch.Tensor,
        text_mus: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total Uncertainty-Calibrated Distributional Contrastive loss.

        L_total = λ_cl · L_calibrated_cl + λ_consist · L_consist + λ_var · L_var

        Args:
            img_features: CLIP image features, shape (B, D) [unused, kept for API compat]
            text_features: CLIP text features, shape (B, D) [unused, kept for API compat]
            img_mu, img_logvar: Image distribution parameters, shape (B, D)
            text_mu, text_logvar: Merged text distribution parameters, shape (B, D)
            text_mus: Individual caption distribution means, shape (B, K, D)
                      Required for distributional consistency loss.

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Component 1: Uncertainty-calibrated contrastive loss
        cl_loss, cl_info = self.calibrated_infonce(
            img_mu, img_logvar, text_mu, text_logvar
        )

        # Component 2: Distributional consistency loss (core innovation)
        if text_mus is not None:
            consist_loss = self.distributional_consistency_loss(
                img_logvar, text_mus
            )
        else:
            # Use zeros to keep the value on the same device and maintain gradient graph
            consist_loss = torch.zeros(1, device=img_mu.device, requires_grad=False).squeeze()

        # Component 3: Variance regularization
        var_loss = self.variance_regularization(img_logvar, text_logvar)

        # Total loss
        total_loss = (
            self.lambda_cl * cl_loss +
            self.lambda_consist * consist_loss +
            self.lambda_var * var_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'contrastive': cl_loss.item(),
            'kl': consist_loss.item(),  # reuse key for training loop compatibility
            'consistency': consist_loss.item(),
            'variance': var_loss.item(),
            'avg_pos_sim': cl_info['avg_pos_sim'],
            'img_var_avg': cl_info['img_var_avg'],
            'text_var_avg': cl_info['text_var_avg'],
        }

        return total_loss, loss_dict


if __name__ == "__main__":
    # Test loss functions
    print("Testing distribution alignment loss functions...")

    B, D, K = 4, 768, 5

    # Create dummy outputs
    img_features = torch.randn(B, D)
    text_features = torch.randn(B, D)
    img_mu = torch.randn(B, D)
    img_logvar = torch.randn(B, D)
    text_mu = torch.randn(B, D)
    text_logvar = torch.randn(B, D)
    text_mus = torch.randn(B, K, D)

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

    # Test DistributionalContrastiveLoss
    print("\n4. Testing DistributionalContrastiveLoss:")
    ot_criterion = DistributionalContrastiveLoss(
        lambda_ot=1.0,
        temperature=0.1,
        min_sigma=1e-3,
        target_variance=0.5,
        lambda_var=0.1,
    )
    ot_loss, ot_dict = ot_criterion(
        img_features, text_features,
        img_mu, img_logvar, text_mu, text_logvar
    )
    print(f"   Total loss: {ot_dict['total']:.4f}")
    print(f"   OT contrastive: {ot_dict['contrastive']:.4f}")
    print(f"   Variance reg: {ot_dict['variance']:.4f}")
    print(f"   Avg W2 (positive): {ot_dict['avg_w2_pos']:.4f}")
    print(f"   Avg W2 (all): {ot_dict['avg_w2_all']:.4f}")

    # Test UncertaintyCalibratedContrastiveLoss
    print("\n5. Testing UncertaintyCalibratedContrastiveLoss:")
    uc_criterion = UncertaintyCalibratedContrastiveLoss(
        lambda_cl=1.0,
        lambda_consist=1.0,
        lambda_var=0.1,
        temperature=0.07,
        target_variance=0.5,
    )
    uc_loss, uc_dict = uc_criterion(
        img_features, text_features,
        img_mu, img_logvar, text_mu, text_logvar,
        text_mus=text_mus,
    )
    print(f"   Total loss: {uc_dict['total']:.4f}")
    print(f"   Calibrated CL loss: {uc_dict['contrastive']:.4f}")
    print(f"   Consistency loss: {uc_dict['consistency']:.4f}")
    print(f"   Variance reg: {uc_dict['variance']:.4f}")
    print(f"   Avg positive similarity: {uc_dict['avg_pos_sim']:.4f}")
    print(f"   Avg img var: {uc_dict['img_var_avg']:.4f}")
    print(f"   Avg text var: {uc_dict['text_var_avg']:.4f}")

    # Test gradient flow
    print("\n6. Testing gradient flow:")
    img_mu_g = torch.randn(B, D, requires_grad=True)
    img_lv_g = torch.randn(B, D, requires_grad=True)
    txt_mu_g = torch.randn(B, D, requires_grad=True)
    txt_lv_g = torch.randn(B, D, requires_grad=True)
    txt_mus_g = torch.randn(B, K, D, requires_grad=True)
    loss_g, _ = uc_criterion(
        torch.randn(B, D), torch.randn(B, D),
        img_mu_g, img_lv_g, txt_mu_g, txt_lv_g,
        text_mus=txt_mus_g,
    )
    loss_g.backward()
    print(f"   grad img_mu: {img_mu_g.grad.norm():.4f}")
    print(f"   grad img_logvar: {img_lv_g.grad.norm():.4f}")
    print(f"   grad text_mu: {txt_mu_g.grad.norm():.4f}")
    print(f"   grad text_logvar: {txt_lv_g.grad.norm():.4f}")
    print(f"   grad text_mus: {txt_mus_g.grad.norm():.4f}")

    print("\nAll loss functions tested successfully!")
