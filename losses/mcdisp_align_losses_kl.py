"""
MCDisp_Align KL-variant loss.

Variant of :class:`losses.mcdisp_align_losses.MCDispAlignLoss` that replaces
the two separate mean / variance alignment terms (L_mu + L_var) with a single
KL-divergence alignment term -- the same construction as
``losses/dist_align_losses_kl.py`` (MSDAKLLoss), ported onto the CURRENT
MCDispAlignLoss terms:

    total = lambda_ctr * L_set + lambda_kl * KL(p_v || p_t)
          + lambda_cover_pos * L_cover_pos + lambda_cover_neg * L_cover_neg
          + lambda_cov * L_cov + lambda_reg * L_reg

Design notes (inherited from the MSDA KL variant, adapted here):
- KL(p_v || p_t) is computed between the DIAGONAL parts of the two
  distributions: the image log-variance head output vs. the moment-matched
  caption-set variance diag(Sigma_bar_t) = s_t^2 + (1/K) sum_k sigma_k^2.
  The closed-form Gaussian KL then decomposes per dimension into
      0.5 * [ (mu_v - mu_t)^2 / sigma_t^2                          (mean part)
            + sigma_v^2 / sigma_t^2 - 1 + log(sigma_t^2/sigma_v^2) ] (var part)
  which unifies the old L_mu (mean-center, cosine) and L_var (log-MSE on the
  caption spread s_t^2) under one likelihood-ratio objective. Unlike the old
  L_var -- whose target was the RAW spread s_t^2 only -- the KL target is the
  FULL moment-matched text variance, so the averaged per-caption variances
  also supervise the image range (they are the "irreducible noise floor" of
  the caption cloud).
- Direction: KL(p_v || p_t) treats the caption-set distribution as the
  reference and stop-gradients it (kl_detach_target), matching the current
  MCDispAlignLoss design where every caption-derived target (s_t^2, deviation
  subspace, cover anchors) is detached. ``kl_symmetric=True`` averages both
  directions instead (each direction detaches its own target); the reverse
  direction then DOES train the caption side through the KL.
- The low-rank image factor U does NOT enter the diagonal KL: with a diagonal
  target, the trace term only sees diag(Sigma_v) and the log-det term only
  constrains the total "volume" of U, never its directions. L_cov (subspace
  direction alignment) is therefore kept unchanged (r capped at min(r, K-1, D)
  as in the current loss).
- Numerics: the 1/sigma_t^2 weight is clamped (kl_var_clamp, added ONLY to
  the weight, not to the ratio / log-det terms) so a collapsed text-variance
  dimension cannot explode the loss; the log-det term is computed as a
  logvar difference (no exp/log round-trip); optional per-dimension
  normalization (kl_normalize_by_dim) keeps the scale comparable to the
  per-D-normalized L_cover and does not drift with D.
- Interaction: the KL var part already constrains log sigma_v^2 and
  partially overlaps L_reg (which pulls BOTH image and text log-variances
  toward the prior sigma_0^2). When integrating, consider reducing
  lambda_reg relative to the MCDispAlignLoss setting to avoid double-pulling
  the variances.
- Key contract: loss_dict keeps the MCDispAlignLoss keys ("set_nce", "mu",
  "var", "cover_pos", ..., "contrastive", "img_var_avg", weighted_*,
  variance statistics) so a trainer accumulating MCDispAlignLoss can swap
  this class in without touching call sites; "mu" / "var" report the KL
  mean / var decomposition instead of the separate losses.

Wired into training via the loss-selection interface:
utils/mcdisp_align_trainer.py (MCDispAlignTrainConfig.loss_name="kl") and
scripts/train_mcdisp_align.py --loss kl [--lambda-kl W]. The KL term rides
the same per-step var_ramp anti-collapse schedule the old L_var followed
(off in Warmup, ramped in Var-Bootstrap, full afterwards); lambda_kl
replaces the (lambda_mu, lambda_var) pair. Default ("standard") keeps the
original MCDispAlignLoss construction byte-identical.
"""

import os
import sys

# When run as a script (`python losses/mcdisp_align_losses_kl.py`), sys.path[0]
# is the script's own directory, so the repo-root `config` module is not
# importable. Put the repo root (this file's parent dir) on the path so
# `import config` and the `utils.*` imports resolve in both `import` and
# `__main__` invocation modes.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger


logger = get_logger("mcdisp_align_losses_kl")


class MCDispAlignKLLoss(nn.Module):
    """MCDisp_Align loss with the mean/variance alignment terms folded into
    one KL divergence.

    total = lambda_ctr * L_set + lambda_kl * KL(p_v || p_t)
          + lambda_cover_pos * L_cover_pos + lambda_cover_neg * L_cover_neg
          + lambda_cov * L_cov + lambda_reg * L_reg

    L_set       : bidirectional InfoNCE on the uncertainty-discounted cosine
                  (identical to MCDispAlignLoss, incl. the straight-through
                  uncertainty_grad_alpha scaling).
    KL          : closed-form diagonal Gaussian KL between the image
                  distribution and the moment-matched caption-set
                  distribution; replaces L_mu (cosine mean-center) + L_var
                  (log-MSE on the caption spread) with a single objective
                  whose per-dim mean part is the sigma_t^2-weighted squared
                  mean gap and whose var part is
                  sigma_v^2/sigma_t^2 - 1 + log(sigma_t^2/sigma_v^2).
    L_cover_pos : per-caption Mahalanobis hinge coverage (caption means are
                  detached anchors, as in MCDispAlignLoss).
    L_cover_neg : optional negative repulsion of other images' caption
                  centers (weight 0 = off, as in MCDispAlignLoss).
    L_cov       : low-rank subspace direction alignment (identical to
                  MCDispAlignLoss: raw-centered caption deviations, r capped
                  at min(r, K-1, D), stop-gradient target; the diagonal KL
                  cannot see U's directions).
    L_reg       : log-variance prior stabilizer on image + caption
                  log-variances (identical to MCDispAlignLoss).

    Image uses Sigma_v = diag(sigma_v^2) + U_v U_v^T (general); text is
    diagonal only, so text_Us is accepted but unused.
    """

    def __init__(
        self,
        lambda_ctr: float = 1.0,
        lambda_kl: float = 1.0,
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
        kl_var_clamp: float = 0.01,
        kl_symmetric: bool = False,
        kl_detach_target: bool = True,
        kl_normalize_by_dim: bool = True,
    ):
        """MCDisp_Align KL-variant loss.

        Args:
            lambda_ctr / lambda_kl / lambda_cover_pos / lambda_cover_neg /
            lambda_cov / lambda_reg: loss-term weights (0 disables a term).
                lambda_kl REPLACES the (lambda_mu, lambda_var) pair of
                MCDispAlignLoss.
            tau: FIXED temperature in the L_set similarity (not learnable).
            m_pos: L_cover positive coverage margin (per-D normalized
                Mahalanobis).
            target_var: L_reg variance prior sigma_0^2.
            m_neg: L_cover negative repulsion margin.
            use_uncertainty_sim: L_set uses the uncertainty-discounted score
                (True) or plain cosine (False; ablation).
            uncertainty_grad_alpha: straight-through scaling of L_set's
                gradient into the variance (1 = normal, 0 = blocked).
            eps: numerical stabilizer for Mahalanobis / log.
            kl_var_clamp: lower bound added to sigma_t^2 in the 1/sigma_t^2
                WEIGHT only, so a collapsed text-variance dimension cannot
                blow up the KL.
            kl_symmetric: use 0.5*(KL(v||t) + KL(t||v)) instead of the
                one-directional KL(v||t).
            kl_detach_target: stop-gradient on the KL target distribution
                (the caption set is a fixed reference, matching the
                detach-everything-caption philosophy of MCDispAlignLoss).
            kl_normalize_by_dim: divide the KL by D so its scale matches the
                per-D-normalized L_cover and does not drift with D.
        """
        super().__init__()
        self.lambda_ctr = lambda_ctr
        self.lambda_kl = lambda_kl
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
        self.kl_var_clamp = float(kl_var_clamp)
        self.kl_symmetric = kl_symmetric
        self.kl_detach_target = kl_detach_target
        self.kl_normalize_by_dim = kl_normalize_by_dim

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _mahalanobis(
        diff: torch.Tensor,
        var: torch.Tensor,
        U: Optional[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        """Squared Mahalanobis distance d^T (Sigma + eps I)^{-1} d, with
        Sigma = diag(var) + U U^T. Uses the Woodbury identity so the only
        solve is the r x r matrix (I + U^T D^{-1} U).

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

        uncertainty_grad_alpha: straight-through scaling of the variance
        gradient -- ``eff_logvar = logvar.detach() + alpha*(logvar - logvar.detach())``.
        Forward score is identical for any alpha; only the gradient into
        logvar is scaled (alpha=0 blocks L_set from pulling sigma^2; 1 = normal).
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

    def _kl_parts(
        self,
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
        mu_p: torch.Tensor,
        logvar_p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-dimension closed-form diagonal Gaussian KL(q || p) split into
        its mean and variance contributions.

        KL(q||p) = 0.5 * sum_d [ (mu_q - mu_p)^2 / sigma_p^2
                                 + sigma_q^2 / sigma_p^2 - 1
                                 + (logvar_p - logvar_q) ]

        Numerics (differs from the MSDA KL variant, which adds the clamp to
        the 1/sigma_p^2 weight ONLY -- that makes the "KL" slightly NEGATIVE
        at q == p, i.e. not a divergence). Here the target variance is floored
        in LOG space (``logvar_p.clamp_min(log kl_var_clamp)``) and the
        clamped value is used consistently in the weight, the ratio, and the
        log-det, so the loss is the EXACT KL(q || p~) against the floored
        target p~: nonnegative everywhere, exactly zero when q == p~, and the
        floor only engages for collapsed target dimensions. The log-det term
        stays a logvar difference (no exp/log round-trip).

        Args:
            mu_q / logvar_q: (B, D) query distribution parameters (image in
                the forward direction).
            mu_p / logvar_p: (B, D) target distribution parameters (caption
                set); detached when ``kl_detach_target``.

        Returns:
            (kl_per_dim, mean_part_per_dim, var_part_per_dim), each (B, D).
        """
        if self.kl_detach_target:
            mu_p = mu_p.detach()
            logvar_p = logvar_p.detach()
        logvar_p = logvar_p.clamp_min(math.log(self.kl_var_clamp))  # floor
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        inv_p = 1.0 / var_p                                  # (B, D)
        mean_part = 0.5 * (mu_q - mu_p) ** 2 * inv_p
        var_part = 0.5 * (var_q * inv_p + (logvar_p - logvar_q) - 1.0)
        return mean_part + var_part, mean_part, var_part

    def _kl_loss(
        self,
        img_mu: torch.Tensor,
        img_logvar: torch.Tensor,
        text_mu: torch.Tensor,
        text_logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """KL alignment between the image and caption-set distributions.

        Returns:
            (kl, kl_mean_term, kl_var_term) scalars: the total KL and its
            mean/var decomposition (from the primary direction when
            symmetric), averaged over the batch.
        """
        kl_per_dim, mean_per_dim, var_per_dim = self._kl_parts(
            img_mu, img_logvar, text_mu, text_logvar)
        if self.kl_symmetric:
            kl_rev, _, _ = self._kl_parts(
                text_mu, text_logvar, img_mu, img_logvar)
            kl_per_dim = 0.5 * (kl_per_dim + kl_rev)
        if self.kl_normalize_by_dim:
            D = img_mu.shape[-1]
            kl_per_dim = kl_per_dim / D
            mean_per_dim = mean_per_dim / D
            var_per_dim = var_per_dim / D
        return kl_per_dim.sum(dim=-1).mean(), \
            mean_per_dim.sum(dim=-1).mean(), \
            var_per_dim.sum(dim=-1).mean()

    def _cover_loss(self, img_mu, img_var, img_U, text_mus, text_mu):
        """L_cover split -> returns (pos_term, neg_term), identical to
        MCDispAlignLoss (caption means are detached anchors).

        pos_term: each caption point under its own image distribution
                  (positive coverage).
        neg_term: optional repulsion of other images' caption centers
                  (L_neg).

        d_M is the squared Mahalanobis under Sigma_v = diag(sigma_v^2) + U U^T,
        per-D normalized (/D) so m_pos ~ O(1).
        """
        # Stop-gradient the caption targets -- they are the anchors being
        # covered, not free points L_cover may drag around (same rationale as
        # MCDispAlignLoss: without the detach, cover_pos pulls every mu_tk
        # toward mu_v through the 1/sigma_v^2-amplified Mahalanobis metric and
        # collapses the between-caption spread the KL target is built from).
        text_mus = text_mus.detach()
        text_mu = text_mu.detach()

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
        subspace (identical to MCDispAlignLoss).

        The caption deviation matrix is the RAW caption means centered by
        their set mean; its top-r principal directions are the target.
        Effective rank is capped at min(r, K-1, D): K centered captions sum to
        zero, so the deviation spans at most K-1 directions (K<=1 -> no target).

        ||P_v - P_t||_F^2 = 2r - 2 ||Q_v^T Q_t||_F^2. Caption deviation
        directions are a stop-gradient target (detach), exactly like the KL
        target. Non-finite values (from near-degenerate QR/eigh) are zeroed.
        """
        if img_U is None or self.lambda_cov <= 0:
            return img_mu.new_zeros(())
        B, D = img_mu.shape
        K = text_mus.shape[1]
        if K <= 1:
            return img_mu.new_zeros(())
        r_eff = min(img_U.shape[-1], K - 1, D)
        if r_eff <= 0:
            return img_mu.new_zeros(())

        Qv, _ = torch.linalg.qr(img_U)            # (B, D, r)
        Qv = Qv[:, :, :r_eff]
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
        """L_reg: pull log-variances toward log(sigma_0^2) (per-dim MEAN, as
        in MCDispAlignLoss). NOTE the KL var part also constrains
        log sigma_v^2; consider a smaller lambda_reg than the
        MCDispAlignLoss setting to avoid double-pulling."""
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
        """MCDisp_Align KL-variant loss. text_mu / text_logvar are the
        moment-matched caption-set distribution (mu_t_bar, diag(Sigma_bar_t));
        text_mus / text_logvars are the per-caption parameters. text_Us is
        accepted for caller compat and unused (text is diagonal).

        Signature matches MCDispAlignLoss.forward so the trainer can swap the
        two without touching the call sites.
        """
        B, D = img_mu.shape
        img_var = torch.exp(img_logvar)                       # (B, D)

        set_nce = self._set_nce(img_mu, img_logvar, text_mu, text_logvar)
        kl_loss, kl_mean_term, kl_var_term = self._kl_loss(
            img_mu, img_logvar, text_mu, text_logvar)
        cover_pos_loss, cover_neg_loss = self._cover_loss(img_mu, img_var, img_U, text_mus, text_mu)
        cov_loss = self._cov_loss(img_mu, img_U, text_mus)
        reg_loss = self._reg_loss(img_logvar, text_logvars)

        total = (
            self.lambda_ctr * set_nce
            + self.lambda_kl * kl_loss
            + self.lambda_cover_pos * cover_pos_loss
            + self.lambda_cover_neg * cover_neg_loss
            + self.lambda_cov * cov_loss
            + self.lambda_reg * reg_loss
        )

        with torch.no_grad():
            img_var_avg = img_var.mean()
            img_var_min = img_var.min()
            img_var_median = img_var.median()
            img_var_max = img_var.max()
            text_var_mean = torch.exp(text_logvar).mean()
            cap_center = text_mus.mean(dim=1, keepdim=True)
            caption_spread = ((text_mus - cap_center) ** 2).mean(dim=1)   # (B, D) = s^2
            caption_spread_mean = caption_spread.mean()
            caption_spread_median = caption_spread.median()
            caption_spread_max = caption_spread.max()
            diag_var_energy = img_var_avg
            if img_U is not None:
                u_energy = (img_U ** 2).sum(dim=-1).mean()
                u_over_diag = u_energy / (diag_var_energy + self.eps)
            else:
                u_energy = img_var.new_zeros(())
                u_over_diag = img_var.new_zeros(())

        loss_dict = {
            "total": total.item(),
            "set_nce": set_nce.item(),
            "kl": kl_loss.item(),
            "kl_mean": kl_mean_term.item(),
            "kl_var": kl_var_term.item(),
            # "mu"/"var" keep the MCDispAlignLoss key contract so the
            # trainer's accumulation/tqdm/epoch-log code works unchanged;
            # they report the KL decomposition (weighted mean part / var part).
            "mu": kl_mean_term.item(),
            "var": kl_var_term.item(),
            "cover_pos": cover_pos_loss.item(),
            "cover_neg": cover_neg_loss.item(),
            "cov": cov_loss.item(),
            "reg": reg_loss.item(),
            "contrastive": set_nce.item(),   # alias for training-loop compat
            "img_var_avg": img_var_avg.item(),
            # weighted contributions (lambda_kl weights the KL decomposition)
            "weighted_set_nce": (self.lambda_ctr * set_nce).item(),
            "weighted_kl": (self.lambda_kl * kl_loss).item(),
            "weighted_mu": (self.lambda_kl * kl_mean_term).item(),
            "weighted_var": (self.lambda_kl * kl_var_term).item(),
            "weighted_cover_pos": (self.lambda_cover_pos * cover_pos_loss).item(),
            "weighted_cover_neg": (self.lambda_cover_neg * cover_neg_loss).item(),
            "weighted_cov": (self.lambda_cov * cov_loss).item(),
            "weighted_reg": (self.lambda_reg * reg_loss).item(),
            # variance statistics (same readouts as MCDispAlignLoss)
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


# ---------------------------------------------------------------------------
# Self-tests (run: python losses/mcdisp_align_losses_kl.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    print("Testing MCDisp_Align KL-variant loss...")

    B, D, K, r = 4, 768, 5, 4
    img_mu = torch.randn(B, D)
    img_logvar = torch.randn(B, D) * 0.5
    img_U = torch.randn(B, D, r)
    text_mu = torch.randn(B, D)
    text_logvar = torch.randn(B, D) * 0.5
    text_mus = torch.randn(B, K, D)
    text_logvars = torch.randn(B, K, D) * 0.5

    print("\n1. Full loss (with covariance, uncertainty-discounted sim):")
    crit = MCDispAlignKLLoss()
    loss, d = crit(img_mu, img_logvar, img_U, text_mu, text_logvar,
                   text_mus, text_logvars)
    for k in ("total", "set_nce", "kl", "kl_mean", "kl_var",
              "cover_pos", "cover_neg", "cov", "reg"):
        print(f"   {k}: {d[k]:.4f}")
    assert math.isfinite(d["total"])

    print("\n2. Diagonal only (img_U=None, lambda_cov=0):")
    crit_d = MCDispAlignKLLoss(lambda_cov=0.0)
    loss_d, d_d = crit_d(img_mu, img_logvar, None, text_mu, text_logvar,
                         text_mus, text_logvars)
    print(f"   total: {d_d['total']:.4f}, cov: {d_d['cov']:.4f} (expect 0)")
    assert d_d["cov"] == 0.0

    print("\n3. Plain cosine (use_uncertainty_sim=False):")
    crit_c = MCDispAlignKLLoss(use_uncertainty_sim=False)
    _, d_c = crit_c(img_mu, img_logvar, img_U, text_mu, text_logvar,
                    text_mus, text_logvars)
    print(f"   set_nce: {d_c['set_nce']:.4f}")

    print("\n4. Closed-form KL vs Monte-Carlo estimate of the KL definition"
          "\n   E_{x~q}[log q(x) - log p(x)]:")
    mu_q, lv_q = img_mu[0], img_logvar[0]          # (D,)
    mu_p, lv_p = text_mu[0], text_logvar[0]        # (D,)
    closed = crit._kl_parts(mu_q.unsqueeze(0), lv_q.unsqueeze(0),
                            mu_p.unsqueeze(0), lv_p.unsqueeze(0))[0].sum().item()
    q = torch.distributions.Normal(mu_q, torch.exp(0.5 * lv_q))
    p = torch.distributions.Normal(mu_p, torch.exp(0.5 * lv_p))
    n = 400_000
    x = q.sample((n,))                              # (n, D)
    mc = (q.log_prob(x) - p.log_prob(x)).sum(dim=-1).mean().item()
    print(f"   closed-form: {closed:.4f}")
    print(f"   Monte-Carlo ({n} samples): {mc:.4f}")
    assert abs(closed - mc) < 0.05 * max(1.0, abs(closed)), \
        f"closed-form {closed:.4f} vs MC {mc:.4f} mismatch"

    print("\n5. KL == 0 exactly when the distributions are aligned "
          "(q == p), > 0 otherwise:")
    crit_zero = MCDispAlignKLLoss(lambda_ctr=0.0, lambda_cover_pos=0.0,
                                  lambda_cover_neg=0.0, lambda_cov=0.0,
                                  lambda_reg=0.0)
    _, d_zero = crit_zero(img_mu, img_logvar, None, img_mu, img_logvar,
                          text_mus, text_logvars)
    print(f"   kl at q == p: {d_zero['kl']:.3e} (expect 0 up to float32 rounding)")
    # mathematically exact zero; float32 var*(1/var) leaves ~1e-9/dim residue
    assert abs(d_zero["kl"]) < 1e-5
    assert d["kl"] > 0.0

    print("\n6. Gradient flow (img_mu / img_logvar / img_U / text_mus):")
    im = img_mu.clone().requires_grad_(True)
    ilv = img_logvar.clone().requires_grad_(True)
    iU = img_U.clone().requires_grad_(True)
    tm = text_mus.clone().requires_grad_(True)
    # In real training text_mu/text_logvar are the model's moment-matching
    # merge of text_mus/text_logvars, so L_set backprops into the captions
    # THROUGH the merged target. Mimic that here so the gradient-flow check
    # reflects reality (the KL itself detaches the target).
    text_mu_from_tm = tm.mean(dim=1)
    loss_g, _ = crit(im, ilv, iU, text_mu_from_tm, text_logvar, tm, text_logvars)
    loss_g.backward()
    print(f"   grad img_mu: {im.grad.norm():.4f}")
    print(f"   grad img_logvar: {ilv.grad.norm():.4f}")
    print(f"   grad img_U: {iU.grad.norm():.4f}")
    print(f"   grad text_mus: {tm.grad.norm():.4f}")
    assert im.grad.norm() > 0 and ilv.grad.norm() > 0 \
        and iU.grad.norm() > 0 and tm.grad.norm() > 0

    print("\n7. Stop-gradient on the KL target (caption set is a fixed reference):")
    t_mu = text_mu.clone().requires_grad_(True)
    t_lv = text_logvar.clone().requires_grad_(True)
    loss_kl, _ = crit_zero(img_mu, img_logvar, None, t_mu, t_lv,
                           text_mus, text_logvars)
    loss_kl.backward()
    g_mu = 0.0 if t_mu.grad is None else t_mu.grad.norm().item()
    g_lv = 0.0 if t_lv.grad is None else t_lv.grad.norm().item()
    print(f"   grad text_mu via KL: {g_mu} (expect 0)")
    print(f"   grad text_logvar via KL: {g_lv} (expect 0)")
    assert g_mu == 0.0 and g_lv == 0.0

    print("\n8. Symmetric KL mode (each direction detaches its own target):")
    crit_s = MCDispAlignKLLoss(kl_symmetric=True, lambda_ctr=0.0,
                               lambda_cover_pos=0.0, lambda_cover_neg=0.0,
                               lambda_cov=0.0, lambda_reg=0.0)
    t_mu2 = text_mu.clone().requires_grad_(True)
    loss_s, d_s = crit_s(img_mu, img_logvar, None, t_mu2, text_logvar,
                         text_mus, text_logvars)
    loss_s.backward()
    print(f"   kl(symmetric): {d_s['kl']:.4f}")
    print(f"   grad text_mu via reverse direction: {t_mu2.grad.norm():.4f} (expect > 0)")
    assert t_mu2.grad.norm() > 0

    print("\n9. Trainer key contract (MCDispAlignLoss keys present):")
    required = ("total", "set_nce", "mu", "var", "cover_pos", "cover_neg",
                "cov", "reg", "contrastive", "img_var_avg",
                "weighted_set_nce", "weighted_mu", "weighted_var",
                "img_var_min", "caption_spread_mean", "u_over_diag")
    missing = [k for k in required if k not in d]
    print(f"   missing keys: {missing if missing else 'none'}")
    assert not missing

    print("\nAll MCDisp_Align KL-variant loss tests passed.")
