"""
MCDisp_Align Loss Functions

Implements the four-group objective of the paper's §3.3 "Text--Image
Distribution Alignment" (docs/MCDisp_Align/iclr2027_conference.tex):

    L = lambda_match * L_match + lambda_mu * L_mu
      + (lambda_var * L_var + lambda_reg * R_prior) + lambda_dir * L_dir

L_match : distribution-to-set bidirectional contrastive between the image
          Gaussian and the B*K caption Gaussians. i2t: each image vs ALL
          captions with its K own captions as positives (logsumexp any-hit
          form, the differentiable analogue of the any-hit I2T metric); t2i:
          every caption vs the B images (per-caption direction). The score is
          switchable via match_score:
            "gaussian" (default): pairwise Gaussian overlap
              log integral p_v(z) p_t(z) dz (losses/gaussian_overlap.py,
              Woodbury form, per-dim normalized), divided by tau_match. The
              score involves the means AND the variances (d_v, U_v,
              sigma_k^2), so this branch DOES send gradient to every
              variance -- that is its purpose (A16): the matching objective
              itself supervises the dispersion; text_logvars is deliberately
              NOT detached.
            "cosine" (the cosine_match ablation baseline): plain cosine of
              the means with the fixed tau. NO gradient reaches any variance
              or covariance parameter in this branch -- the image variance is
              then supervised only by the dispersion statistics.
L_mu    : explicit center alignment in RAW coordinates (A01):
          MSE(mu_v, sg[mu_t]). The center is no longer supervised only
          implicitly through the contrastive cosine, which discarded scale.
L_var   : log-space alignment of the FULL image marginal variance
          d_v + sum_r U_r^2 (A02) to the stop-gradient text variance
          diag(Sigma_bar_t) = s_t^2 + (1/K) sum_k sigma_k^2, which encodes the
          empirical caption dispersion. Aligning the diagonal alone cannot
          see low-rank energy growth. Log space keeps the gradient
          well-scaled when the caption dispersion is small.
R_prior : caption variances calibrated toward the prior sigma_0^2 (renamed
          from L_cal); besides the gaussian L_match the only supervisor of
          the caption-level variances, which enter the text variance through
          the moment-matched merge.
L_dir   : subspace alignment between the image covariance factor U_v and the
          top-r eigenvectors of the between-caption covariance S_t
          (stop-gradient target; r capped at min(r, K-1, D) because K centered
          captions span at most K-1 directions), guarded by a spectral rank
          check (A05): samples whose caption deviations are numerically
          rank-deficient (actual rank < r_eff, e.g. collapsed captions) are
          excluded from the mean (dir_valid / dir_total report how many
          survived) instead of being charged the constant 2r.

Gradient scope per parameter (which term reaches what):
  gaussian L_match: mu_v, d_v, U_v, mu_{i,k}^t, sigma_k^2 -- all of them;
  cosine  L_match : the means only;
  L_mu            : mu_v only (the text target is detached);
  L_var           : d_v and U_v (the text target is detached);
  R_prior         : sigma_k^2 only;
  L_dir           : U_v only (the caption target is detached).
"""

import os
import sys

# When run as a script (`python losses/mcdisp_align_losses.py`), sys.path[0] is the
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

from losses.gaussian_overlap import gaussian_overlap_scores
from utils.logger import get_logger


logger = get_logger("mcdisp_align_losses")


class MCDispAlignLoss(nn.Module):
    """
    Multi-Caption Semantic Dispersion Guided Distribution Alignment (MCDisp_Align)
    loss -- the parameter-level alignment of the image distribution with the
    moment-matched text distribution (paper §3.3).

    Total loss = lambda_match * L_match + lambda_mu * L_mu
               + (lambda_var * L_var + lambda_reg * R_prior)
               + lambda_dir * L_dir

    L_match : distribution-to-set bidirectional contrastive, image vs the
              B*K caption Gaussians. Gaussian overlap score (default):
              gradient reaches mu_v, d_v, U_v AND the caption variances
              sigma_k^2 (A16 -- the match itself supervises dispersion).
              Cosine score (match_score="cosine", ablation baseline):
              means only, no gradient to any variance.
    L_mu    : MSE(mu_v, sg[mu_t]) in raw coordinates (A01).
    L_var   : MSE(log(d_v + sum_r U_r^2), sg[log diag(Sigma_bar_t)]) -- the
              FULL image marginal variance, log space (A02).
    R_prior : MSE(log sigma_k^2, log sigma_0^2) over the caption variances
              (renamed from L_cal).
    L_dir   : 2r - 2 ||Q_v^T sg(Q_t)||_F^2 projection distance between the
              image low-rank subspace and the top-r eigenspace of the caption
              between-covariance, computed only on samples whose caption
              deviation spectrum has actual rank >= r_eff (A05 guard).

    Image uses Sigma_v = diag(sigma_v^2) + U_v U_v^T; each caption distribution
    is diagonal (Sigma_k^t = diag(sigma_k^2)), so text_Us is accepted for
    caller compatibility and unused.
    """

    def __init__(
        self,
        lambda_match: float = 1.0,
        lambda_mu: float = 0.5,
        lambda_var: float = 1.0,
        lambda_reg: float = 0.01,
        lambda_dir: float = 0.5,
        tau: float = 0.07,
        tau_match: float = 1.0,
        sigma0_sq: float = 0.04,
        match_score: str = "gaussian",
        eps: float = 1e-6,
        dir_eig_rel_tol: float = 1e-3,
        lambda_ctr: Optional[float] = None,
        lambda_cal: Optional[float] = None,
    ):
        """MCDisp_Align four-group loss (paper §3.3).

        Args:
            lambda_match / lambda_mu / lambda_var / lambda_reg / lambda_dir:
                weights of the five atomics (0 disables a term; the ablations
                zero the corresponding one). lambda_var and lambda_reg form
                the dispersion group ("disp" in the loss dict).
            tau: FIXED temperature of the cosine match score (ablation
                baseline only; not learnable).
            tau_match: temperature of the gaussian overlap match logits. The
                overlap score is per-dimension normalized, so O(1) scores
                take a O(1) temperature.
            sigma0_sq: caption-variance prior sigma_0^2 of R_prior (measured
                on held-out data; MSCOCO caption spread ~0.04).
            match_score: "gaussian" (default; overlap score, gradient also to
                the variances) or "cosine" (means-only ablation baseline).
            eps: numerical stabilizer for log / eigh.
            dir_eig_rel_tol: relative tolerance of the L_dir spectral rank
                guard: an eigenvalue counts toward the actual rank only if it
                exceeds max_eig * dir_eig_rel_tol (floored at eps).
            lambda_ctr / lambda_cal: DEPRECATED aliases of lambda_match /
                lambda_reg (kept so the trainer keeps running across this
                refactor; they override the new kwargs when given).
        """
        super().__init__()
        if match_score not in ("gaussian", "cosine"):
            raise ValueError(
                f"match_score must be 'gaussian' or 'cosine', got {match_score!r}")
        if lambda_ctr is not None:
            logger.warning("deprecated: lambda_ctr -> lambda_match")
            lambda_match = lambda_ctr
        if lambda_cal is not None:
            logger.warning("deprecated: lambda_cal -> lambda_reg")
            lambda_reg = lambda_cal
        self.lambda_match = lambda_match
        self.lambda_mu = lambda_mu
        self.lambda_var = lambda_var
        self.lambda_reg = lambda_reg
        self.lambda_dir = lambda_dir
        self.tau = float(tau)
        self.tau_match = float(tau_match)
        self.sigma0_sq = float(sigma0_sq)
        self.match_score = match_score
        self.eps = eps
        self.dir_eig_rel_tol = float(dir_eig_rel_tol)

    # ------------------------------------------------------------------ sub-losses
    def _match_loss(
        self,
        img_mu: torch.Tensor,
        img_diag: torch.Tensor,
        img_U: Optional[torch.Tensor],
        text_mus: torch.Tensor,
        text_logvars: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """L_match: distribution-to-set bidirectional contrastive (paper §3.3).

        i2t: image i is contrasted against ALL B*K captions of the batch;
             its K own captions are positives. The logsumexp-over-positives
             form maximizes P(top-1 caption ∈ own set) -- the differentiable
             analogue of the any-hit I2T retrieval metric.
        t2i: EVERY caption must retrieve its own image among the B images
             (single positive per query), mirroring the per-caption T2I
             metric.

        Score (match_score):
          "gaussian": pairwise Gaussian overlap of the image Gaussian
             (mu_v, d_v, U_v) with each caption Gaussian (mu_k, d_k) divided
             by tau_match. The score involves the variances, so gradient
             flows to img_logvar, img_U AND text_logvars (NOT detached) --
             this branch is itself a variance supervisor (A16).
          "cosine": plain cosine of the means with the fixed tau; no
             gradient reaches any variance or covariance parameter.

        Returns (L_match, L_i2t, L_t2i) -- the two parts are logged separately.
        """
        B, K, D = text_mus.shape
        if self.match_score == "gaussian":
            cap_mu = text_mus.reshape(B * K, D)
            cap_d = torch.exp(text_logvars.reshape(B * K, D))   # NOT detached (A16)
            logits = gaussian_overlap_scores(
                img_mu, img_diag, img_U, cap_mu, cap_d) / self.tau_match
        else:  # "cosine"
            img_n = F.normalize(img_mu, dim=-1)                     # (B, D)
            cap_n = F.normalize(text_mus.reshape(B * K, D), dim=-1)  # (B*K, D)
            logits = (img_n @ cap_n.T) / self.tau                   # (B, B*K)

        # i2t: multi-positive, softmax-ratio form: maximize
        # P(top-1 caption ∈ own set) = -log[ sum_own / sum_all ]
        #   = logsumexp(all B*K) - logsumexp(own K)
        own = logits.reshape(B, B, K)                               # [i, j, k]
        pos = own[torch.arange(B, device=logits.device),
                  torch.arange(B, device=logits.device)]            # (B, K)
        loss_i2t = (torch.logsumexp(logits, dim=-1)
                    - torch.logsumexp(pos, dim=-1)).mean()

        # t2i: per-caption cross entropy (label = owning image)
        labels = torch.arange(B * K, device=logits.device) // K
        loss_t2i = F.cross_entropy(logits.T, labels)

        return 0.5 * (loss_i2t + loss_t2i), loss_i2t, loss_t2i

    def _dir_loss(
        self, img_U: Optional[torch.Tensor], text_mus: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int]:
        """L_dir: align the image low-rank subspace with the caption variation
        directions (top-r eigenvectors of the between-caption covariance S_t),
        with the A05 spectral rank guard.

        ||P_v - P_t||_F^2 = 2r - 2 ||Q_v^T Q_t||_F^2. Caption directions are a
        stop-gradient target. Effective rank is capped at min(r, K-1, D): K
        centered captions sum to zero, so the deviations span at most K-1
        directions (K<=1 -> no target). Rank guard (A05): per sample, the
        actual rank of the caption deviation spectrum is estimated from the
        eigenvalues of G = dev dev^T + eps*I (an eigenvalue counts if it
        exceeds max_eig * dir_eig_rel_tol, floored at eps); samples with
        actual rank < r_eff -- e.g. collapsed captions, where every eigenvalue
        is the eps ridge and none exceeds it -- are EXCLUDED from the mean
        instead of being charged the constant 2r. Non-finite values (from
        near-degenerate eigh) are zeroed to protect the cov head and, through
        shared img_mu/img_U, the retrieval means.

        Returns (L_dir, dir_valid, dir_total): the loss over the valid
        sub-batch, how many samples passed the guard, and the batch size.
        """
        B = text_mus.shape[0]
        if img_U is None or self.lambda_dir <= 0:
            return text_mus.new_zeros(()), 0, B
        D = img_U.shape[-2]
        K = text_mus.shape[1]
        if K <= 1:
            return text_mus.new_zeros(()), 0, B
        r_eff = min(img_U.shape[-1], K - 1, D)
        if r_eff <= 0:
            return text_mus.new_zeros(()), 0, B

        Qv, _ = torch.linalg.qr(img_U)            # (B, D, r)
        Qv = Qv[:, :, :r_eff]
        # Between-caption deviations mu_ik - mu_bar_i (raw centering), detached:
        # the caption covariance S_t is the supervision, not a trainable target.
        dev = (text_mus - text_mus.mean(dim=1, keepdim=True)).detach()   # (B, K, D)
        G = dev @ dev.transpose(-1, -2) + self.eps * torch.eye(K, device=dev.device)  # (B, K, K)
        eigvals, eigvecs = torch.linalg.eigh(G)   # ascending
        # A05 rank guard: with zero true spread every eigenvalue equals the
        # eps ridge, so none exceeds the (eps-floored) threshold -> rank 0.
        max_eig = eigvals.max(dim=-1).values                       # (B,)
        thresh = torch.clamp(max_eig * self.dir_eig_rel_tol, min=self.eps)
        actual_rank = (eigvals > thresh.unsqueeze(-1)).sum(-1)     # (B,)
        valid = actual_rank >= r_eff
        dir_valid = int(valid.sum().item())
        if dir_valid == 0:
            return text_mus.new_zeros(()), 0, B

        top_vals = eigvals[:, -r_eff:].clamp(min=self.eps)
        top_vecs = eigvecs[:, :, -r_eff:]
        Qt = torch.matmul(dev.transpose(-1, -2), top_vecs / torch.sqrt(top_vals).unsqueeze(1))
        Qt = F.normalize(Qt, dim=-2)
        C = Qv.transpose(-1, -2) @ Qt             # (B, r_eff, r_eff)
        per_sample = 2 * r_eff - 2 * (C ** 2).sum(dim=(-1, -2))    # (B,)
        dir_loss = per_sample[valid].mean()

        if not torch.isfinite(dir_loss).all():
            logger.warning("L_dir produced a non-finite value; zeroing (detached).")
            dir_loss = torch.zeros_like(dir_loss)
        return dir_loss, dir_valid, B

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
        """MCDisp_Align four-group loss. text_mu / text_logvar are the
        moment-matched text distribution (mean and diag of Sigma_bar_t);
        text_mus / text_logvars are the per-caption parameters. text_Us is
        accepted for caller compatibility and unused (caption distributions
        are diagonal). All terms are averaged over the batch, as in
        eq:overall_objective.

        In gaussian match mode text_logvars is NOT detached: L_match itself
        supervises the caption variances (A16). Only the cosine branch is
        variance-free.
        """
        B, K, D = text_mus.shape
        img_diag = torch.exp(img_logvar)                     # d_v
        img_marginal = img_diag if img_U is None else \
            img_diag + (img_U ** 2).sum(-1)                  # A02: d_v + sum_r U_r^2

        # --- L_match: distribution-to-set bidirectional contrastive ---
        match_loss, match_i2t, match_t2i = self._match_loss(
            img_mu, img_diag, img_U, text_mus, text_logvars)
        # --- L_mu: raw-coordinate center alignment (A01) ---
        mu_loss = F.mse_loss(img_mu, text_mu.detach())
        # --- dispersion group: full-marginal variance alignment + weak prior ---
        var_loss = F.mse_loss(torch.log(img_marginal + self.eps),
                              text_logvar.detach())          # A02: full marginal
        log_target = math.log(max(self.sigma0_sq, self.eps))
        reg_loss = ((torch.log(torch.exp(text_logvars) + self.eps)
                     - log_target) ** 2).mean()              # R_prior (ex-L_cal)
        # --- L_dir: between-caption variation subspace (A05 rank guard) ---
        dir_loss, dir_valid, dir_total = self._dir_loss(img_U, text_mus)

        total = (self.lambda_match * match_loss + self.lambda_mu * mu_loss
                 + self.lambda_var * var_loss + self.lambda_reg * reg_loss
                 + self.lambda_dir * dir_loss)

        with torch.no_grad():
            img_diag_var_mean = img_diag.mean()
            img_diag_var_median = img_diag.median()
            img_diag_var_min = img_diag.min()
            img_diag_var_max = img_diag.max()
            # low-rank vs diagonal energy on the image side
            if img_U is not None:
                img_lowrank_var_mean = (img_U ** 2).sum(dim=-1).mean()
                u_over_diag = img_lowrank_var_mean / (img_diag_var_mean + self.eps)
            else:
                img_lowrank_var_mean = img_diag.new_zeros(())
                u_over_diag = img_diag.new_zeros(())
            img_marginal_var_mean = img_marginal.mean()
            img_marginal_var_median = img_marginal.median()
            img_marginal_var_min = img_marginal.min()
            img_marginal_var_max = img_marginal.max()
            # text variance diag(Sigma_bar_t) and its two components
            text_var_mean = torch.exp(text_logvar).mean()
            cap_center = text_mus.mean(dim=1, keepdim=True)
            caption_spread = ((text_mus - cap_center) ** 2).mean(dim=1)   # (B, D) = s_t^2
            caption_spread_mean = caption_spread.mean()
            caption_spread_median = caption_spread.median()
            caption_spread_max = caption_spread.max()
            cap_var_mean = torch.exp(text_logvars).mean()
            # core-innovation readout: image variance vs the dispersion component
            var_over_spread = img_marginal_var_mean / (caption_spread_mean + self.eps)

        loss_dict = {
            "total": total.item(),
            "match": match_loss.item(),
            "match_i2t": match_i2t.item(),   # distribution-to-set: any-hit direction
            "match_t2i": match_t2i.item(),   # distribution-to-set: per-caption direction
            "mu": mu_loss.item(),
            "var": var_loss.item(),
            "reg": reg_loss.item(),
            "dir": dir_loss.item(),
            "dir_valid": dir_valid,
            "dir_total": dir_total,
            "weighted_match": (self.lambda_match * match_loss).item(),
            "weighted_mu": (self.lambda_mu * mu_loss).item(),
            "weighted_var": (self.lambda_var * var_loss).item(),
            "weighted_reg": (self.lambda_reg * reg_loss).item(),
            "weighted_dir": (self.lambda_dir * dir_loss).item(),
            # dispersion group contribution
            "disp": (self.lambda_var * var_loss + self.lambda_reg * reg_loss).item(),
            # ---- legacy aliases (temporary, until the trainer is updated) ----
            "ctr": match_loss.item(),
            "ctr_i2t": match_i2t.item(),
            "ctr_t2i": match_t2i.item(),
            "cal": reg_loss.item(),
            "weighted_ctr": (self.lambda_match * match_loss).item(),
            "weighted_cal": (self.lambda_reg * reg_loss).item(),
            # variance statistics
            "img_diag_var_min": img_diag_var_min.item(),
            "img_diag_var_median": img_diag_var_median.item(),
            "img_diag_var_mean": img_diag_var_mean.item(),
            "img_diag_var_max": img_diag_var_max.item(),
            "img_var_min": img_diag_var_min.item(),      # legacy name used by the trainer
            "img_var_median": img_diag_var_median.item(),
            "img_var_mean": img_diag_var_mean.item(),
            "img_var_max": img_diag_var_max.item(),
            "img_var_avg": img_diag_var_mean.item(),     # legacy alias used by the trainer
            # full-marginal statistics
            "img_lowrank_var_mean": img_lowrank_var_mean.item(),
            "img_marginal_var_min": img_marginal_var_min.item(),
            "img_marginal_var_median": img_marginal_var_median.item(),
            "img_marginal_var_mean": img_marginal_var_mean.item(),
            "img_marginal_var_max": img_marginal_var_max.item(),
            "marginal_log_mse": var_loss.item(),
            "mu_mse_raw": mu_loss.item(),
            "text_var_mean": text_var_mean.item(),
            "cap_var_mean": cap_var_mean.item(),
            "caption_spread_mean": caption_spread_mean.item(),
            "caption_spread_median": caption_spread_median.item(),
            "caption_spread_max": caption_spread_max.item(),
            "var_over_spread": var_over_spread.item(),
            # low-rank covariance statistics (U vs diagonal energy; legacy names)
            "u_energy": img_lowrank_var_mean.item(),
            "diag_var_energy": img_diag_var_mean.item(),
            "u_over_diag": u_over_diag.item(),
        }
        return total, loss_dict


if __name__ == "__main__":
    print("Testing MCDisp_Align loss (four-group objective)...")
    torch.manual_seed(0)
    B, D, K, r = 4, 64, 5, 4
    img_mu = torch.randn(B, D)
    img_logvar = -3 + 0.5 * torch.randn(B, D)
    img_U = 0.1 * torch.randn(B, D, r)
    text_mu = img_mu + 0.1 * torch.randn(B, D)
    text_logvar = -3 + 0.5 * torch.randn(B, D)
    text_mus = text_mu.unsqueeze(1) + 0.2 * torch.randn(B, K, D)
    text_logvars = -3 + 0.5 * torch.randn(B, K, D)

    print("\n1. Full four-group loss (gaussian match, default weights):")
    crit = MCDispAlignLoss()
    loss, d = crit(img_mu, img_logvar, img_U, text_mu, text_logvar,
                   text_mus, text_logvars)
    for k in ("total", "match", "match_i2t", "match_t2i", "mu", "var", "reg",
              "dir", "dir_valid", "dir_total", "disp"):
        print(f"   {k}: {d[k]:.4f}" if isinstance(d[k], float) else f"   {k}: {d[k]}")
    print(f"   loss_dict keys ({len(d)}): {sorted(d)}")
    assert all(math.isfinite(v) for v in d.values()), "gaussian mode must be finite everywhere"
    weighted_sum = (d["weighted_match"] + d["weighted_mu"] + d["weighted_var"]
                    + d["weighted_reg"] + d["weighted_dir"])
    assert abs(d["total"] - weighted_sum) < 1e-6, (d["total"], weighted_sum)
    assert abs(d["disp"] - (d["weighted_var"] + d["weighted_reg"])) < 1e-6

    print("\n2. Cosine match (cosine_match ablation baseline):")
    crit_cos = MCDispAlignLoss(match_score="cosine")
    _, d_cos = crit_cos(img_mu, img_logvar, img_U, text_mu, text_logvar,
                        text_mus, text_logvars)
    for k in ("total", "match", "match_i2t", "match_t2i"):
        print(f"   {k}: {d_cos[k]:.4f}")
    assert math.isfinite(d_cos["total"])
    assert d_cos["dir_valid"] == d["dir_valid"]

    print("\n3. Gaussian match-only gradient flow (A16: L_match supervises variances):")
    crit_m = MCDispAlignLoss(lambda_mu=0.0, lambda_var=0.0, lambda_reg=0.0,
                             lambda_dir=0.0)
    leaves = {
        "img_mu": img_mu.clone().requires_grad_(True),
        "img_logvar": img_logvar.clone().requires_grad_(True),
        "img_U": img_U.clone().requires_grad_(True),
        "text_mus": text_mus.clone().requires_grad_(True),
        "text_logvars": text_logvars.clone().requires_grad_(True),
    }
    loss_m, _ = crit_m(leaves["img_mu"], leaves["img_logvar"], leaves["img_U"],
                       text_mu, text_logvar, leaves["text_mus"],
                       leaves["text_logvars"])
    loss_m.backward()
    for name, t in leaves.items():
        g = t.grad
        assert g is not None and torch.isfinite(g).all() and g.norm() > 0, name
        print(f"   grad {name}: {g.norm().item():.4e}")
    print("   text_logvars grad > 0: the gaussian L_match supervises sigma_k^2 (A16)")

    print("\n4. Cosine match sends NO gradient to any variance:")
    ilv = img_logvar.clone().requires_grad_(True)
    tlv = text_logvars.clone().requires_grad_(True)
    crit_c = MCDispAlignLoss(match_score="cosine", lambda_mu=0.0, lambda_var=0.0,
                             lambda_reg=0.0, lambda_dir=0.0)
    total_c, _ = crit_c(img_mu, ilv, None, text_mu, text_logvar,
                        text_mus, tlv)
    total_c.backward()
    assert ilv.grad is None or ilv.grad.norm() == 0, "cosine L_match must not touch sigma_v^2"
    assert tlv.grad is None or tlv.grad.norm() == 0, "cosine L_match must not touch sigma_t^2"
    print("   img/text logvar grads from cosine L_match: 0 (verified)")

    print("\n5. L_dir rank guard (collapsed captions are skipped, not charged 2r):")
    collapsed = text_mus[:, :1].repeat(1, K, 1)
    _, d_coll = crit(img_mu, img_logvar, img_U, text_mu, text_logvar,
                     collapsed, text_logvars)
    print(f"   dir: {d_coll['dir']}, dir_valid: {d_coll['dir_valid']}/{d_coll['dir_total']}")
    assert d_coll["dir"] == 0.0 and d_coll["dir_valid"] == 0

    print("\n6. Deprecated aliases still map:")
    crit_alias = MCDispAlignLoss(lambda_ctr=0.7, lambda_cal=0.05)
    assert crit_alias.lambda_match == 0.7 and crit_alias.lambda_reg == 0.05
    print("   lambda_ctr=0.7 -> lambda_match, lambda_cal=0.05 -> lambda_reg")

    print("\nAll MCDisp_Align loss self-tests passed.")
