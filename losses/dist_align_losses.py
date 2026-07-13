"""
GaussianImageDistribution - Distribution Alignment Loss Functions

This module implements loss functions for distribution-based alignment.
The primary loss is MSDALoss (Multi-caption Semantic Distribution Alignment).
Generic contrastive/KL losses (DistributionAlignmentLoss, CombinedDistributionLoss)
are retained for the ProLIP baseline.
"""

import math
from typing import Dict, Optional, Tuple

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


class MSDALoss(nn.Module):
    """
    Multi-caption Semantic Distribution Alignment (MSDA) loss.

    Total loss:
        L = lambda_ctr * L_set-NCE
          + lambda_mu  * L_mu
          + lambda_var * L_var      (stop-gradient on caption spread; core)
          + lambda_cover * L_cover
          + lambda_cov  * L_cov     (active only when cov_rank > 0)
          + lambda_reg  * L_reg

    Both image and text are general Gaussians N(mu, Sigma) with
    Sigma = diag(sigma^2) + U U^T. The image variance is supervised to match
    the multi-caption semantic spread, and the image covariance directions are
    supervised to match caption deviation directions.
    """

    def __init__(
        self,
        lambda_ctr: float = 1.0,
        lambda_mu: float = 0.5,
        lambda_var: float = 1.0,
        lambda_cover: float = 0.5,
        lambda_cov: float = 0.1,
        lambda_reg: float = 0.01,
        tau: float = 0.07,
        m_pos: float = 1.0,
        target_var: float = 0.5,
        eps: float = 1e-6,
        use_neg_cover: bool = False,
        m_neg: float = 2.0,
        use_uncertainty_sim: bool = True,
        var_loss_mode: str = "rescaled",
    ):
        """
        Initialize MSDA loss.

        Args:
            lambda_ctr: Weight for set-level contrastive loss L_set-NCE
            lambda_mu: Weight for mean-center alignment loss L_mu
            lambda_var: Weight for variance semantic consistency L_var (core)
            lambda_cover: Weight for multi-caption coverage loss L_cover
            lambda_cov: Weight for covariance direction alignment L_cov
            lambda_reg: Weight for variance regularization L_reg
            tau: Temperature for uncertainty-discounted similarity
            m_pos: Per-dim-normalized positive coverage radius
            target_var: Target variance sigma_0^2 for L_reg
            eps: Numerical stabilizer for Mahalanobis / log
            use_neg_cover: Whether to add the negative coverage repulsion term
            m_neg: Negative coverage margin
            use_uncertainty_sim: Use uncertainty-discounted similarity (True) or
                                standard cosine (False) in L_set-NCE
            var_loss_mode: How L_var supervises σ² against caption_spread:
                "raw"      - target = caption_spread (original). The large mean
                             offset between σ² (~0.1) and caption_spread (~0.003)
                             drowns the per-image signal and collapses the variance
                             head to a near-constant.
                "rescaled" - target = caption_spread rescaled so its batch mean
                             matches σ²'s batch mean. Removes the offset, preserves
                             per-image variation (CV), so the head learns to track
                             caption diversity. (Default; the fix.)
        """
        super().__init__()
        self.lambda_ctr = lambda_ctr
        self.lambda_mu = lambda_mu
        self.lambda_var = lambda_var
        self.lambda_cover = lambda_cover
        self.lambda_cov = lambda_cov
        self.lambda_reg = lambda_reg
        self.tau = tau
        self.m_pos = m_pos
        self.target_var = target_var
        self.eps = eps
        self.use_neg_cover = use_neg_cover
        self.m_neg = m_neg
        self.use_uncertainty_sim = use_uncertainty_sim
        self.var_loss_mode = var_loss_mode

    def _variance_target(self, img_var: torch.Tensor, caption_spread: torch.Tensor,
                         mode: str) -> torch.Tensor:
        """Return the (detached) L_var target for img_var given caption_spread.

        Args:
            img_var: (B, D) predicted σ².
            caption_spread: (B, D) per-dim variance of caption means (the raw
                semantic-spread signal).
            mode: "raw" (identity) or "rescaled" (mean-match to img_var).
        """
        if mode == "rescaled":
            scale = (img_var.mean().detach() + self.eps) / (
                caption_spread.mean().detach() + self.eps)
            return (caption_spread * scale).detach()
        return caption_spread.detach()

    @staticmethod
    def _mahalanobis(
        diff: torch.Tensor,
        var: torch.Tensor,
        U: Optional[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        """
        Squared Mahalanobis distance d^T (Sigma + eps I)^{-1} d, with
        Sigma = diag(var) + U U^T. Uses the Woodbury identity so the only solve
        is the r x r matrix (I + U^T D^{-1} U).

        Args:
            diff: (..., D)
            var:  (..., D) diagonal variances
            U:    (..., D, r) or None (diagonal-only when None)
            eps:  numerical stabilizer

        Returns:
            (...,) per-leading-shape squared Mahalanobis distance.
        """
        inv = 1.0 / (var + eps)                  # (..., D) = D^{-1}
        a = inv * diff                           # (..., D) = D^{-1} d
        if U is None:
            return (diff * a).sum(dim=-1)        # d^T D^{-1} d
        W = inv.unsqueeze(-1) * U                # (..., D, r) = D^{-1} U
        UtA = U.transpose(-1, -2) @ a.unsqueeze(-1)          # (..., r, 1)
        S = torch.eye(U.shape[-1], device=U.device) + U.transpose(-1, -2) @ W  # (..., r, r)
        z = torch.linalg.solve(S, UtA)           # (..., r, 1)
        correction = (W @ z).squeeze(-1)         # (..., D)
        sigma_inv_diff = a - correction          # Sigma^{-1} d
        return (diff * sigma_inv_diff).sum(dim=-1)

    def forward(
        self,
        img_mu: torch.Tensor,
        img_logvar: torch.Tensor,
        img_U: Optional[torch.Tensor],
        text_mu_bar: torch.Tensor,
        text_logvar_bar: torch.Tensor,
        text_mus: torch.Tensor,
        text_logvars: torch.Tensor,
        text_Us: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total MSDA loss.

        Args:
            img_mu, img_logvar: (B, D) image distribution parameters.
            img_U: (B, D, r) or None.
            text_mu_bar, text_logvar_bar: (B, D) caption-set distribution params.
            text_mus: (B, K, D) per-caption means.
            text_logvars: (B, K, D) per-caption log variances.
            text_Us: (B, K, D, r) or None.

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        B, D = img_mu.shape
        K = text_mus.shape[1]
        img_var = torch.exp(img_logvar)                       # (B, D)

        # ---- L_set-NCE: uncertainty-discounted bidirectional InfoNCE ----
        img_mu_n = F.normalize(img_mu, dim=-1)
        text_mu_n = F.normalize(text_mu_bar, dim=-1)
        sim = img_mu_n @ text_mu_n.T                          # (B, B)
        if self.use_uncertainty_sim:
            img_scale = torch.sqrt(1.0 + img_var.mean(dim=-1))              # (B,)
            text_scale = torch.sqrt(1.0 + torch.exp(text_logvar_bar).mean(dim=-1))  # (B,)
            scale = img_scale.unsqueeze(1) * text_scale.unsqueeze(0)        # (B, B)
            logits = sim / (self.tau * scale)
        else:
            logits = sim / self.tau
        labels = torch.arange(B, device=img_mu.device)
        set_nce = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )

        # ---- L_mu: mean-center alignment ----
        mu_loss = (1.0 - F.cosine_similarity(img_mu, text_mu_bar, dim=-1)).mean()

        # ---- L_var: variance semantic consistency (stop-grad on caption spread)
        caption_spread = ((text_mus - text_mu_bar.unsqueeze(1)) ** 2).mean(dim=1)  # (B, D)
        var_target = self._variance_target(img_var, caption_spread, self.var_loss_mode)
        var_loss = F.mse_loss(img_var, var_target)

        # ---- L_cover: multi-caption coverage via Mahalanobis ----
        diff = text_mus - img_mu.unsqueeze(1)                 # (B, K, D)
        var_exp = img_var.unsqueeze(1).expand_as(diff)        # (B, K, D)
        U_exp = img_U.unsqueeze(1).expand(-1, K, -1, -1) if img_U is not None else None
        dM = self._mahalanobis(diff, var_exp, U_exp, self.eps)    # (B, K)
        dM_mean = dM / D                                      # per-dim normalize
        cover_loss = F.relu(dM_mean - self.m_pos).mean()

        neg_cover_loss = img_mu.new_zeros(())
        if self.use_neg_cover and B > 1:
            # Repel other images' caption centers from this image distribution.
            bar_exp = text_mu_bar.unsqueeze(1).expand(B, B, D)        # (B, B, D) j
            img_mu_exp = img_mu.unsqueeze(0).expand(B, B, D)          # (B, B, D) i
            var_e2 = img_var.unsqueeze(1).expand(B, B, D)
            U_e2 = img_U.unsqueeze(1).expand(B, B, -1, -1) if img_U is not None else None
            d_other = self._mahalanobis(
                bar_exp - img_mu_exp, var_e2, U_e2, self.eps
            ) / D
            mask = ~torch.eye(B, dtype=torch.bool, device=img_mu.device)
            neg_cover_loss = F.relu(self.m_neg - d_other[mask]).mean()

        # ---- L_cov: covariance direction alignment (subspace Frobenius) ----
        # Compares the image covariance subspace Qv = orth(U) against the caption
        # deviation subspace Qt. ||Pv - Pt||_F^2 = 2r - 2 ||Qv^T Qt||_F^2.
        cov_loss = img_mu.new_zeros(())
        if img_U is not None and self.lambda_cov > 0:
            r_eff = min(img_U.shape[-1], K)                  # cannot exceed #captions
            if r_eff > 0:
                # Image subspace basis Qv (orthonormalize U columns via QR).
                Qv, _ = torch.linalg.qr(img_U)               # (B, D, r)
                Qv = Qv[:, :, :r_eff]
                # Caption deviation subspace basis Qt. The deviation matrix is
                # K x D with K << D and often ill-conditioned, so derive Qt from
                # an eigendecomposition of the small K x K Gram matrix instead
                # of a batched SVD on K x D.
                # NOTE: detach() -- the caption deviation directions are a *target*
                # (exactly like L_var's caption spread). Without stop-gradient,
                # L_cov would also push the caption/text means to match the image
                # covariance, corrupting the retrieval means and crashing Recall@1
                # the moment L_cov activates.
                dev = (text_mus - text_mu_bar.unsqueeze(1)).detach()    # (B, K, D)
                G = dev @ dev.transpose(-1, -2) + self.eps * torch.eye(
                    K, device=dev.device)                    # (B, K, K)
                eigvals, eigvecs = torch.linalg.eigh(G)      # ascending
                top_vals = eigvals[:, -r_eff:].clamp(min=self.eps)   # (B, r_eff)
                top_vecs = eigvecs[:, :, -r_eff:]                    # (B, K, r_eff)
                Qt = torch.matmul(
                    dev.transpose(-1, -2),
                    top_vecs / torch.sqrt(top_vals).unsqueeze(1),
                )                                            # (B, D, r_eff)
                Qt = F.normalize(Qt, dim=-2)
                C = Qv.transpose(-1, -2) @ Qt                # (B, r_eff, r_eff)
                cov_loss = (2 * r_eff - 2 * (C ** 2).sum(dim=(-1, -2))).mean()
                # QR / eigh / solve backward can emit Inf/NaN at near-degenerate
                # points (linearly-dependent U columns, near-duplicate captions).
                # Detach-and-zero any non-finite cov_loss so a single bad batch
                # cannot corrupt the cov head -- and through L_cover (which shares
                # img_U and img_mu) the retrieval means. Forward value dropped.
                if not torch.isfinite(cov_loss).all():
                    logger.warning(
                        "L_cov produced a non-finite value; zeroing (detached) "
                        "for this batch to protect the cov head / retrieval means."
                    )
                    # Fresh constant zero -- NOT `nan * 0` (which stays nan under
                    # IEEE-754). Contributes 0 to the loss and 0 gradient, i.e.
                    # skip L_cov for this batch only.
                    cov_loss = torch.zeros_like(cov_loss)

        # ---- L_reg: variance regularization (image + text, symmetric) ----
        log_t = math.log(self.target_var)
        img_reg = F.mse_loss(img_logvar, torch.full_like(img_logvar, log_t))
        text_reg = F.mse_loss(text_logvars, torch.full_like(text_logvars, log_t))
        reg_loss = img_reg + text_reg

        total = (
            self.lambda_ctr * set_nce
            + self.lambda_mu * mu_loss
            + self.lambda_var * var_loss
            + self.lambda_cover * cover_loss
            + self.lambda_cov * cov_loss
            + self.lambda_reg * reg_loss
        )
        # NOTE: L_neg is added at weight 1.0 (no lambda). It is experimental and
        # disabled by default (MSDA_USE_NEG_COVER=False); enable only for trials.
        if self.use_neg_cover:
            total = total + neg_cover_loss

        with torch.no_grad():
            avg_pos_sim = logits[torch.arange(B), torch.arange(B)].mean()
            img_var_avg = img_var.mean()
            text_var_avg = torch.exp(text_logvar_bar).mean()

        loss_dict = {
            'total': total.item(),
            'set_nce': set_nce.item(),
            'mu': mu_loss.item(),
            'var': var_loss.item(),
            'cover': cover_loss.item(),
            'cov': cov_loss.item(),
            'reg': reg_loss.item(),
            'contrastive': set_nce.item(),   # alias for training-loop compat
            'avg_pos_sim': avg_pos_sim.item(),
            'img_var_avg': img_var_avg.item(),
            'text_var_avg': text_var_avg.item(),
        }

        return total, loss_dict


if __name__ == "__main__":
    # Test loss functions
    print("Testing MSDA loss functions...")

    B, D, K, r = 4, 768, 5, 4
    img_mu = torch.randn(B, D)
    img_logvar = torch.randn(B, D)
    img_U = torch.randn(B, D, r)
    text_mu_bar = torch.randn(B, D)
    text_logvar_bar = torch.randn(B, D)
    text_mus = torch.randn(B, K, D)
    text_logvars = torch.randn(B, K, D)
    text_Us = torch.randn(B, K, D, r)

    print("\n1. Testing MSDALoss (full, with covariance):")
    crit = MSDALoss()
    loss, d = crit(img_mu, img_logvar, img_U, text_mu_bar, text_logvar_bar,
                   text_mus, text_logvars, text_Us)
    for k in ('total', 'set_nce', 'mu', 'var', 'cover', 'cov', 'reg'):
        print(f"   {k}: {d[k]:.4f}")

    print("\n2. Testing MSDALoss (diagonal only, img_U=None):")
    crit_d = MSDALoss()
    loss_d, d_d = crit_d(img_mu, img_logvar, None, text_mu_bar, text_logvar_bar,
                         text_mus, text_logvars, None)
    print(f"   total: {d_d['total']:.4f}, cov: {d_d['cov']:.4f} (expect 0)")

    print("\n3. Testing gradient flow (mu / logvar / U):")
    im = img_mu.clone().requires_grad_(True)
    il = img_logvar.clone().requires_grad_(True)
    iU = img_U.clone().requires_grad_(True)
    tm = text_mus.clone().requires_grad_(True)
    tl = text_logvars.clone().requires_grad_(True)
    loss_g, _ = crit(im, il, iU, text_mu_bar, text_logvar_bar, tm, tl, text_Us)
    loss_g.backward()
    print(f"   grad img_mu: {im.grad.norm():.4f}")
    print(f"   grad img_logvar: {il.grad.norm():.4f}")
    print(f"   grad img_U: {iU.grad.norm():.4f}")
    print(f"   grad text_mus: {tm.grad.norm():.4f}")
    print(f"   grad text_logvars: {tl.grad.norm():.4f}")
    assert im.grad.norm() > 0 and il.grad.norm() > 0 and iU.grad.norm() > 0
    print("\nAll MSDA loss tests passed.")
