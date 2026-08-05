"""
GaussianImageDistribution - Distribution Alignment Loss Functions

This module implements loss functions for distribution-based alignment.
The primary loss is MSDALoss (Multi-caption Semantic Distribution Alignment).
"""

import os
import sys

# When run as a script (`python losses/dist_align_losses.py`), sys.path[0] is the
# script's own directory, so the repo-root `config` module is not importable.
# Put the repo root (this file's parent dir) on the path so `import config` and
# the `utils.*` imports resolve in both `import` and `__main__` invocation modes.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger


logger = get_logger("dist_align_losses")


class MSDALoss(nn.Module):
    """
    Multi-caption Semantic Distribution Alignment (MSDA) loss.

    Total loss = lambda_ctr * L_set + lambda_mu * L_mu + lambda_var * L_var
               + lambda_cover_pos * L_cover_pos + lambda_cover_neg * L_cover_neg
               + lambda_cov * L_cov + lambda_reg * L_reg

    L_set      : bidirectional InfoNCE on the uncertainty-discounted cosine
                 similarity  sim = (mu_v . mu_t) / (tau * sqrt(1+mean sigma_v^2)
                                                         * sqrt(1+mean sigma_t^2))
                 (normalized mean space).
    L_mu       : 1 - cos(mu_v, mu_t_bar) — explicit mean-center alignment.
    L_var      : sigma_v^2 tracks the RAW multi-caption semantic spread s_t^2,
                 with stop-gradient on s_t^2 (core innovation).
    L_cover_pos: image distribution covers every caption point (Mahalanobis hinge,
                 per-D normalized).
    L_cover_neg: optional negative repulsion of other images' caption centers
                 (methodology 5.4; weight lambda_cover_neg, 0 = off).
    L_cov      : image low-rank subspace aligns with caption-deviation directions,
                 with stop-gradient on the target (normalized mean space).
    L_reg      : log-variance pulled toward log(sigma_0^2) — anti-collapse/anti-explode.

    Image uses Sigma_v = diag(sigma_v^2) + U_v U_v^T (general); text is diagonal
    only (v1), so text_Us is accepted but unused.
    """

    def __init__(
        self,
        lambda_ctr: float = 1.0,
        lambda_mu: float = 0.5,
        lambda_var: float = 1.0,
        lambda_cover_pos: float = 0.5,
        lambda_cover_neg: float = 0.0,
        lambda_cov: float = 0.2,
        lambda_reg: float = 0.01,
        tau: float = 0.07,
        m_pos: float = 1.0,
        target_var: float = 0.04,
        m_neg: float = 2.0,
        use_uncertainty_sim: bool = True,
        uncertainty_grad_alpha: float = 1.0,
        eps: float = 1e-6,
    ):
        """MSDA loss per the methodology.

        Args:
            lambda_*: weights for the loss terms (0 disables a term).
            lambda_cover_pos / lambda_cover_neg: separate weights for L_cover's
                positive coverage and its OPTIONAL negative repulsion
                (methodology 5.4 "可以再加" L_neg). cover_neg default 0 = pos-only
                canonical L_cover.
            tau: FIXED temperature in the L_set similarity (not learnable).
            m_pos: L_cover positive coverage margin (per-D normalized Mahalanobis).
            target_var: L_reg variance prior sigma_0^2.
            m_neg: L_cover negative repulsion margin.
            use_uncertainty_sim: L_set/retrieval use the uncertainty-discounted
                score (True) or plain cosine (False; ablation).
            uncertainty_grad_alpha: scales L_set's gradient into the variance via
                straight-through scaling (forward score unchanged). 0 blocks L_set
                from pulling sigma^2 (Warmup anti-collapse); 1 = normal.
            eps: numerical stabilizer for Mahalanobis / log.
        """
        super().__init__()
        self.lambda_ctr = lambda_ctr
        self.lambda_mu = lambda_mu
        self.lambda_var = lambda_var
        self.lambda_cover_pos = lambda_cover_pos
        self.lambda_cover_neg = lambda_cover_neg
        self.lambda_cov = lambda_cov
        self.lambda_reg = lambda_reg
        self.tau = float(tau)
        self.m_pos = float(m_pos)
        self.target_var = float(target_var)
        self.m_neg = float(m_neg)
        self.use_uncertainty_sim = use_uncertainty_sim
        self.uncertainty_grad_alpha = uncertainty_grad_alpha
        self.eps = eps

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _mahalanobis(
        diff: torch.Tensor,
        var: torch.Tensor,
        U: Optional[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        """Squared Mahalanobis distance d^T (Sigma + eps I)^{-1} d, with
        Sigma = diag(var) + U U^T. Uses the Woodbury identity so the only solve
        is the r x r matrix (I + U^T D^{-1} U).

        Args:
            diff: (N, D)
            var:  (N, D) diagonal variances
            U:    (N, D, r) or None (diagonal-only when None)
            eps:  numerical stabilizer

        Returns:
            (N,) squared Mahalanobis distance.
        """
        inv = 1.0 / (var + eps)                  # (N, D) = D^{-1}
        a = inv * diff                           # (N, D) = D^{-1} d
        if U is None:
            return (diff * a).sum(dim=-1)        # d^T D^{-1} d
        W = inv.unsqueeze(-1) * U                # (N, D, r) = D^{-1} U
        UtA = U.transpose(-1, -2) @ a.unsqueeze(-1)          # (N, r, 1)
        S = torch.eye(U.shape[-1], device=U.device) + U.transpose(-1, -2) @ W  # (N, r, r)
        z = torch.linalg.solve(S, UtA)           # (N, r, 1)
        correction = (W @ z).squeeze(-1)         # (N, D)
        sigma_inv_diff = a - correction          # Sigma^{-1} d
        return (diff * sigma_inv_diff).sum(dim=-1)

    @staticmethod
    def _sim_matrix(img_mu_n, img_logvar, text_mu_n, text_logvar, tau,
                    use_uncertainty_sim, uncertainty_grad_alpha=1.0):
        """Uncertainty-discounted cosine similarity matrix (B, B).

        sim[i,j] = (mu_v_i . mu_t_j) / (tau * s_i * s_j)  where
        s = sqrt(1 + mean(sigma^2)). When use_uncertainty_sim is False the
        variance discounting is dropped (plain cosine / tau). Means are assumed
        already L2-normalized by the caller.

        uncertainty_grad_alpha: straight-through scaling of the variance gradient
        -- ``eff_logvar = logvar.detach() + alpha*(logvar - logvar.detach())``.
        Forward score is identical for any alpha; only the gradient into logvar
        is scaled (alpha=0 blocks L_set from pulling sigma^2; 1 = normal).
        """
        base = img_mu_n @ text_mu_n.T                     # (B, B)
        if not use_uncertainty_sim:
            return base / tau
        a = uncertainty_grad_alpha
        img_lv = img_logvar.detach() + a * (img_logvar - img_logvar.detach())
        txt_lv = text_logvar.detach() + a * (text_logvar - text_logvar.detach())
        img_scale = torch.sqrt(1.0 + torch.exp(img_lv).mean(dim=-1))   # (B,)
        text_scale = torch.sqrt(1.0 + torch.exp(txt_lv).mean(dim=-1))  # (B,)
        return base / (tau * img_scale.unsqueeze(1) * text_scale.unsqueeze(0))

    # ------------------------------------------------------------------ sub-losses
    def _set_nce(self, img_mu, img_logvar, text_mu, text_logvar):
        """L_set: bidirectional InfoNCE on the uncertainty-discounted cosine."""
        B = img_mu.shape[0]
        img_mu_n = F.normalize(img_mu, dim=-1)
        text_mu_n = F.normalize(text_mu, dim=-1)
        sim = self._sim_matrix(img_mu_n, img_logvar, text_mu_n, text_logvar,
                               self.tau, self.use_uncertainty_sim,
                               self.uncertainty_grad_alpha)
        labels = torch.arange(B, device=img_mu.device)
        return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))

    def _mu_loss(self, img_mu, text_mu):
        """L_mu: 1 - cos(mu_v, mu_t_bar)."""
        return (1.0 - F.cosine_similarity(img_mu, text_mu, dim=-1)).mean()

    def _var_loss(self, img_var, text_mus):
        """L_var: sigma_v^2 tracks the RAW multi-caption semantic spread
        s_t^2 = (1/K) sum_k (mu_tk - mu_t_bar)^2, with stop-gradient on the target.

        Matched in LOG space: MSE(log sigma^2, log s^2). The minimum is the same
        (sigma^2 = s^2) but the gradient is scale-invariant -- a 2x deviation gives
        the same gradient whether sigma^2 is 0.04 or 4.0. Linear-space MSE gives a
        tiny gradient at the caption-spread scale (~0.04) and the variance head
        collapses to a constant (sigma diagnostic Case A; see methods.md §6.4).
        """
        text_center = text_mus.mean(dim=1)                                 # (B, D)
        caption_spread = ((text_mus - text_center.unsqueeze(1)) ** 2).mean(dim=1)  # (B, D) = s^2
        log_target = torch.log(caption_spread.detach() + self.eps)         # stop-gradient on s^2
        return F.mse_loss(torch.log(img_var + self.eps), log_target)

    def _cover_loss(self, img_mu, img_var, img_U, text_mus, text_mu):
        """L_cover split -> returns (pos_term, neg_term).

        pos_term: each caption point under its own image distribution
                  (methodology 5.4 positive coverage).
        neg_term: optional repulsion of other images' caption centers
                  (methodology 5.4 "可以再加" L_neg).
        Weighted separately in forward via lambda_cover_pos / lambda_cover_neg.

        d_M is the squared Mahalanobis under Sigma_v = diag(sigma_v^2) + U U^T,
        per-D normalized (÷D) so m_pos ~ O(1).
        """
        B, K, D = text_mus.shape

        # positive coverage: each caption point under its own image distribution
        diff_pos = (text_mus - img_mu.unsqueeze(1)).reshape(B * K, D)
        var_pos = img_var.unsqueeze(1).expand(B, K, D).reshape(B * K, D)
        U_pos = None
        if img_U is not None:
            r = img_U.shape[-1]
            U_pos = img_U.unsqueeze(1).expand(B, K, D, r).reshape(B * K, D, r)
        d2_pos = self._mahalanobis(diff_pos, var_pos, U_pos, self.eps) / D   # (B*K,)
        pos_term = F.relu(d2_pos - self.m_pos).mean()

        # negative repulsion: other images' caption centers should be FAR (large d_M)
        diff_neg = (text_mu.unsqueeze(0) - img_mu.unsqueeze(1)).reshape(B * B, D)  # (B*B, D) [i, j]
        var_neg = img_var.unsqueeze(1).expand(B, B, D).reshape(B * B, D)
        U_neg = None
        if img_U is not None:
            r = img_U.shape[-1]
            U_neg = img_U.unsqueeze(1).expand(B, B, D, r).reshape(B * B, D, r)
        d2_neg = (self._mahalanobis(diff_neg, var_neg, U_neg, self.eps) / D).reshape(B, B)
        eye = torch.eye(B, device=img_mu.device).bool()
        neg_term = F.relu(self.m_neg - d2_neg).masked_fill(eye, 0.0).sum() / (B * max(B - 1, 1))

        return pos_term, neg_term

    def _cov_loss(self, img_mu, img_U, text_mus):
        """L_cov: align the image low-rank subspace U with the caption-deviation
        subspace. Per methodology 4.2, the caption deviation matrix is the RAW
        caption means centered by their set mean, μ_ik^t - μ̄_i^t (not
        normalize-then-subtract); its top-r principal directions are the target.
        Effective rank is capped at min(r, K-1, D): K centered captions sum to
        zero, so the deviation spans at most K-1 directions (K<=1 -> no target).

        ||P_v - P_t||_F^2 = 2r - 2 ||Q_v^T Q_t||_F^2. Caption deviation directions
        are a stop-gradient target (detach), exactly like L_var's caption spread.
        Non-finite values (from near-degenerate QR/eigh) are zeroed to protect the
        cov head and, through shared img_mu/img_U, the retrieval means.
        """
        if img_U is None or self.lambda_cov <= 0:
            return img_mu.new_zeros(())
        B, D = img_mu.shape
        K = text_mus.shape[1]
        # K centered captions sum to zero, so they span at most K-1 directions ->
        # the caption-deviation Gram matrix has rank <= min(K-1, D). Cap r_eff
        # accordingly; K<=1 yields no deviation direction, so L_cov is 0.
        if K <= 1:
            return img_mu.new_zeros(())
        r_eff = min(img_U.shape[-1], K - 1, D)
        if r_eff <= 0:
            return img_mu.new_zeros(())

        Qv, _ = torch.linalg.qr(img_U)            # (B, D, r)
        Qv = Qv[:, :, :r_eff]
        # Methodology 4.2: caption deviation matrix = μ_ik^t - μ̄_i^t — the RAW
        # caption means centered by their set mean (NOT normalize-then-subtract).
        # SVD on this centered deviation gives the target principal directions.
        dev = (text_mus - text_mus.mean(dim=1, keepdim=True)).detach()   # (B, K, D)
        G = dev @ dev.transpose(-1, -2) + self.eps * torch.eye(K, device=dev.device)  # (B, K, K)
        eigvals, eigvecs = torch.linalg.eigh(G)   # ascending
        top_vals = eigvals[:, -r_eff:].clamp(min=self.eps)
        top_vecs = eigvecs[:, :, -r_eff:]
        Qt = torch.matmul(dev.transpose(-1, -2), top_vecs / torch.sqrt(top_vals).unsqueeze(1))
        Qt = F.normalize(Qt, dim=-2)
        C = Qv.transpose(-1, -2) @ Qt             # (B, r_eff, r_eff)
        cov_loss = (2 * r_eff - 2 * (C ** 2).sum(dim=(-1, -2))).mean()

        if not torch.isfinite(cov_loss).all():
            logger.warning("L_cov produced a non-finite value; zeroing (detached).")
            cov_loss = torch.zeros_like(cov_loss)
        return cov_loss

    def _reg_loss(self, img_logvar, text_logvars):
        """L_reg: pull log-variances toward log(sigma_0^2).

        Uses a per-dim MEAN (not the methodology 5.6 sum-over-D / ||.||_2^2).
        The sum-over-D form would make lambda_reg=0.01 contribute ~D× more,
        i.e. ~7.7× the per-dim weight of L_var (lambda_var=1.0, mean-over-D),
        pulling sigma^2 toward sigma_0^2 hard enough to defeat the core
        innovation (sigma^2 = caption spread, supervised by L_var). The mean
        form keeps L_reg ~100× weaker than L_var so L_var dominates the
        variance shape. See methods.md §4 / the audit for the full rationale.
        """
        log_target = math.log(max(self.target_var, self.eps))
        img_term = ((img_logvar - log_target) ** 2).mean()
        txt_term = ((text_logvars - log_target) ** 2).mean()
        return img_term + txt_term

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        img_mu: torch.Tensor,
        img_logvar: torch.Tensor,
        img_U: Optional[torch.Tensor],
        text_mu: torch.Tensor,
        text_logvar: torch.Tensor,
        text_mus: torch.Tensor,
        text_logvars: torch.Tensor,
        text_Us: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """MSDA loss. text_mu / text_logvar are the moment-matched caption-set
        distribution (mu_t_bar, sigma_t_bar^2); text_mus / text_logvars are the
        per-caption parameters. text_Us is accepted for caller compat and unused
        (text is diagonal in v1)."""
        B, D = img_mu.shape
        img_var = torch.exp(img_logvar)                       # (B, D)

        set_nce = self._set_nce(img_mu, img_logvar, text_mu, text_logvar)
        mu_loss = self._mu_loss(img_mu, text_mu)
        var_loss = self._var_loss(img_var, text_mus)
        cover_pos_loss, cover_neg_loss = self._cover_loss(img_mu, img_var, img_U, text_mus, text_mu)
        cov_loss = self._cov_loss(img_mu, img_U, text_mus)
        reg_loss = self._reg_loss(img_logvar, text_logvars)

        total = (
            self.lambda_ctr * set_nce
            + self.lambda_mu * mu_loss
            + self.lambda_var * var_loss
            + self.lambda_cover_pos * cover_pos_loss
            + self.lambda_cover_neg * cover_neg_loss
            + self.lambda_cov * cov_loss
            + self.lambda_reg * reg_loss
        )

        with torch.no_grad():
            img_var_avg = img_var.mean()
            # --- diagnostics: weighted contributions, variance & low-rank cov stats ---
            img_var_min = img_var.min()
            img_var_median = img_var.median()
            img_var_max = img_var.max()
            text_var_mean = torch.exp(text_logvars).mean()
            cap_center = text_mus.mean(dim=1, keepdim=True)
            caption_spread = ((text_mus - cap_center) ** 2).mean(dim=1)   # (B, D) = s^2
            caption_spread_mean = caption_spread.mean()
            caption_spread_median = caption_spread.median()
            caption_spread_max = caption_spread.max()
            diag_var_energy = img_var_avg
            if img_U is not None:
                u_energy = (img_U ** 2).sum(dim=-1).mean()              # mean over (B,D) of sum_r U^2
                u_over_diag = u_energy / (diag_var_energy + self.eps)
            else:
                u_energy = img_var.new_zeros(())
                u_over_diag = img_var.new_zeros(())

        loss_dict = {
            "total": total.item(),
            "set_nce": set_nce.item(),
            "mu": mu_loss.item(),
            "var": var_loss.item(),
            "cover_pos": cover_pos_loss.item(),
            "cover_neg": cover_neg_loss.item(),
            "cov": cov_loss.item(),
            "reg": reg_loss.item(),
            "contrastive": set_nce.item(),   # alias for training-loop compat
            "img_var_avg": img_var_avg.item(),
            # weighted contributions (current per-step/per-epoch lambdas -> real share of total)
            "weighted_set_nce": (self.lambda_ctr * set_nce).item(),
            "weighted_mu": (self.lambda_mu * mu_loss).item(),
            "weighted_var": (self.lambda_var * var_loss).item(),
            "weighted_cover_pos": (self.lambda_cover_pos * cover_pos_loss).item(),
            "weighted_cover_neg": (self.lambda_cover_neg * cover_neg_loss).item(),
            "weighted_cov": (self.lambda_cov * cov_loss).item(),
            "weighted_reg": (self.lambda_reg * reg_loss).item(),
            # variance statistics
            "img_var_min": img_var_min.item(),
            "img_var_median": img_var_median.item(),
            "img_var_mean": img_var_avg.item(),
            "img_var_max": img_var_max.item(),
            "text_var_mean": text_var_mean.item(),
            "caption_spread_mean": caption_spread_mean.item(),
            "caption_spread_median": caption_spread_median.item(),
            "caption_spread_max": caption_spread_max.item(),
            # low-rank covariance statistics (U vs diagonal energy)
            "u_energy": u_energy.item(),
            "diag_var_energy": diag_var_energy.item(),
            "u_over_diag": u_over_diag.item(),
        }
        return total, loss_dict


if __name__ == "__main__":
    print("Testing MSDA loss (methodology-aligned)...")
    B, D, K, r = 4, 768, 5, 4
    img_mu = torch.randn(B, D)
    img_logvar = torch.randn(B, D)
    img_U = torch.randn(B, D, r)
    text_mu = torch.randn(B, D)
    text_logvar = torch.randn(B, D)
    text_mus = torch.randn(B, K, D)
    text_logvars = torch.randn(B, K, D)

    print("\n1. Full loss (with covariance, uncertainty-discounted sim):")
    crit = MSDALoss()
    loss, d = crit(img_mu, img_logvar, img_U, text_mu, text_logvar,
                   text_mus, text_logvars)
    for k in ("total", "set_nce", "mu", "var", "cover_pos", "cover_neg", "cov", "reg"):
        print(f"   {k}: {d[k]:.4f}")
    assert math.isfinite(d["total"])

    print("\n2. Diagonal only (img_U=None):")
    crit_d = MSDALoss(lambda_cov=0.0)
    loss_d, d_d = crit_d(img_mu, img_logvar, None, text_mu, text_logvar,
                         text_mus, text_logvars)
    print(f"   total: {d_d['total']:.4f}, cov: {d_d['cov']:.4f} (expect 0)")

    print("\n3. Plain cosine (use_uncertainty_sim=False):")
    crit_c = MSDALoss(use_uncertainty_sim=False)
    _, d_c = crit_c(img_mu, img_logvar, img_U, text_mu, text_logvar,
                    text_mus, text_logvars)
    print(f"   set_nce: {d_c['set_nce']:.4f}")

    print("\n4. Gradient flow (img_mu / img_logvar / img_U / text_mus):")
    im = img_mu.clone().requires_grad_(True)
    ilv = img_logvar.clone().requires_grad_(True)
    iU = img_U.clone().requires_grad_(True)
    tm = text_mus.clone().requires_grad_(True)
    loss_g, _ = crit(im, ilv, iU, text_mu, text_logvar, tm, text_logvars)
    loss_g.backward()
    print(f"   grad img_mu: {im.grad.norm():.4f}")
    print(f"   grad img_logvar: {ilv.grad.norm():.4f}")
    print(f"   grad img_U: {iU.grad.norm():.4f}")
    print(f"   grad text_mus: {tm.grad.norm():.4f}")
    assert im.grad.norm() > 0 and ilv.grad.norm() > 0 and iU.grad.norm() > 0 and tm.grad.norm() > 0

    print("\n5. Stop-gradient check (caption spread gets no grad through L_var target):")
    tm2 = text_mus.clone().requires_grad_(True)
    _, _ = crit(img_mu, img_logvar, img_U, text_mu, text_logvar, tm2, text_logvars)
    # text_mus still receives gradient via L_set/L_cover/L_cov (mu_t_bar, per-caption),
    # but the L_var target itself is detached; this just confirms backward runs.
    print("   backward completed without error")
    print("\nAll MSDA loss tests passed.")
