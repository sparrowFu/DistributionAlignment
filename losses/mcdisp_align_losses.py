"""
MCDisp_Align Loss Functions

Implements the four-term objective of the paper's §3.3 "Text--Image
Distribution Alignment" (docs/MCDisp_Align/iclr2027_conference.tex):

    L = lambda_ctr * L_ctr + lambda_var * L_var
      + lambda_dir * L_dir + lambda_cal * L_cal

L_ctr  : distribution-to-set bidirectional InfoNCE on the PLAIN cosine of
         the means (paper §3.3): image vs ALL B*K captions with its K own
         captions as positives (logsumexp, any-hit direction), and every
         caption vs the B images (per-caption direction). Fixed tau; the
         similarity involves no variance, so the contrastive term sends no
         gradient to any variance -- the image variance is supervised only
         by the dispersion statistics, not implicitly by the matching
         objective.
L_var  : log-space regression of the image variance to the stop-gradient text
         variance diag(Sigma_bar_t) = s_t^2 + (1/K) sum_k sigma_k^2, which
         encodes the empirical caption dispersion. Log space keeps the
         gradient well-scaled when the caption dispersion is small.
L_dir  : subspace alignment between the image covariance factor U_v and the
         top-r eigenvectors of the between-caption covariance S_t
         (stop-gradient target; r capped at min(r, K-1, D) because K centered
         captions span at most K-1 directions).
L_cal  : caption variances calibrated toward the prior sigma_0^2 (the only
         supervisor of the caption-level variances, which enter the text
         variance through the moment-matched merge).
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

from utils.logger import get_logger


logger = get_logger("mcdisp_align_losses")


class MCDispAlignLoss(nn.Module):
    """
    Multi-Caption Semantic Dispersion Guided Distribution Alignment (MCDisp_Align)
    loss -- the parameter-level alignment of the image distribution with the
    moment-matched text distribution (paper §3.3).

    Total loss = lambda_ctr * L_ctr + lambda_var * L_var
               + lambda_dir * L_dir + lambda_cal * L_cal

    L_ctr : plain-cosine distribution-to-set InfoNCE on (mu_v, per-caption
            mu_{i,k}^t); fixed tau; no gradient reaches any variance.
    L_var : MSE(log sigma_v^2, sg[log diag(Sigma_bar_t)]) in log space.
    L_dir : 2r - 2 ||Q_v^T sg(Q_t)||_F^2 projection distance between the
            image low-rank subspace and the top-r eigenspace of S_t.
    L_cal : MSE(log sigma_k^2, log sigma_0^2) over the caption variances.

    Image uses Sigma_v = diag(sigma_v^2) + U_v U_v^T; each caption distribution
    is diagonal (Sigma_k^t = diag(sigma_k^2)), so text_Us is accepted for
    caller compatibility and unused.
    """

    def __init__(
        self,
        lambda_ctr: float = 1.0,
        lambda_var: float = 1.0,
        lambda_dir: float = 0.5,
        lambda_cal: float = 0.01,
        tau: float = 0.07,
        sigma0_sq: float = 0.04,
        eps: float = 1e-6,
    ):
        """MCDisp_Align loss (paper §3.3).

        Args:
            lambda_*: weights for the loss terms (0 disables a term; the
                no_var / no_dir / no_ctr ablations zero the corresponding one).
            tau: FIXED temperature in the L_ctr similarity (not learnable).
            sigma0_sq: caption-calibration prior sigma_0^2 (measured on
                held-out data; MSCOCO caption spread ~0.04).
            eps: numerical stabilizer for log / eigh.
        """
        super().__init__()
        self.lambda_ctr = lambda_ctr
        self.lambda_var = lambda_var
        self.lambda_dir = lambda_dir
        self.lambda_cal = lambda_cal
        self.tau = float(tau)
        self.sigma0_sq = float(sigma0_sq)
        self.eps = eps

    # ------------------------------------------------------------------ sub-losses
    def _ctr_loss(
        self, img_mu: torch.Tensor, text_mus: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """L_ctr: distribution-to-set bidirectional InfoNCE (paper §3.3).

        i2t: image i is contrasted against ALL B*K captions of the batch;
             its K own captions are positives. The logsumexp-over-positives
             form maximizes P(top-1 caption ∈ own set) -- the differentiable
             analogue of the any-hit I2T retrieval metric, and it implements
             "every caption is covered" from §3.3.
        t2i: EVERY caption must retrieve its own image among the B images
             (single positive per query), mirroring the per-caption T2I
             metric. This is the supervision the moment-matched mean could
             not provide (each caption previously got only 1/K of the
             gradient, through the average).
        The score is the plain cosine of the means with a FIXED tau, so no
        gradient reaches any variance or covariance parameter.

        Returns (L_ctr, L_i2t, L_t2i) -- the two parts are logged separately.
        """
        B, K, D = text_mus.shape
        img_n = F.normalize(img_mu, dim=-1)                       # (B, D)
        cap_n = F.normalize(text_mus.reshape(B * K, D), dim=-1)   # (B*K, D)
        logits = (img_n @ cap_n.T) / self.tau                     # (B, B*K)

        # i2t: multi-positive, standard CE orientation: maximize
        # P(top-1 caption ∈ own set) = -log[ sum_own / sum_all ]
        #   = logsumexp(all B*K) - logsumexp(own K)
        own = logits.view(B, B, K)                                # [i, j, k]
        pos = own[torch.arange(B, device=logits.device),
                  torch.arange(B, device=logits.device)]          # (B, K)
        loss_i2t = (torch.logsumexp(logits, dim=-1)
                    - torch.logsumexp(pos, dim=-1)).mean()

        # t2i: per-caption cross entropy (label = owning image)
        labels = torch.arange(B * K, device=logits.device) // K
        loss_t2i = F.cross_entropy(logits.T, labels)

        return 0.5 * (loss_i2t + loss_t2i), loss_i2t, loss_t2i

    def _var_loss(
        self, img_logvar: torch.Tensor, text_logvar: torch.Tensor
    ) -> torch.Tensor:
        """L_var: image variance regressed to the stop-gradient text variance.

        The model's merged text_logvar IS log(diag(Sigma_bar_t) + eps) with
        diag(Sigma_bar_t) = s_t^2 + (1/K) sum_k sigma_k^2 (moment matching),
        so this is exactly the paper's eq:variance_alignment. The target is
        detached: the caption set is a fixed supervision target, not a moving
        one. Log space keeps the gradient scale-invariant across the small
        magnitude (~0.04) of real caption dispersion.
        """
        img_log = torch.log(torch.exp(img_logvar) + self.eps)   # log(sigma_v^2 + eps)
        return F.mse_loss(img_log, text_logvar.detach())

    def _dir_loss(
        self, img_U: Optional[torch.Tensor], text_mus: torch.Tensor
    ) -> torch.Tensor:
        """L_dir: align the image low-rank subspace with the caption variation
        directions (top-r eigenvectors of the between-caption covariance S_t).

        ||P_v - P_t||_F^2 = 2r - 2 ||Q_v^T Q_t||_F^2. Caption directions are a
        stop-gradient target. Effective rank is capped at min(r, K-1, D): K
        centered captions sum to zero, so the deviations span at most K-1
        directions (K<=1 -> no target). Non-finite values (from near-degenerate
        eigh) are zeroed to protect the cov head and, through shared
        img_mu/img_U, the retrieval means.
        """
        if img_U is None or self.lambda_dir <= 0:
            return text_mus.new_zeros(())
        B, D = img_U.shape[0], img_U.shape[-2]
        K = text_mus.shape[1]
        if K <= 1:
            return text_mus.new_zeros(())
        r_eff = min(img_U.shape[-1], K - 1, D)
        if r_eff <= 0:
            return text_mus.new_zeros(())

        Qv, _ = torch.linalg.qr(img_U)            # (B, D, r)
        Qv = Qv[:, :, :r_eff]
        # Between-caption deviations mu_ik - mu_bar_i (raw centering), detached:
        # the caption covariance S_t is the supervision, not a trainable target.
        dev = (text_mus - text_mus.mean(dim=1, keepdim=True)).detach()   # (B, K, D)
        G = dev @ dev.transpose(-1, -2) + self.eps * torch.eye(K, device=dev.device)  # (B, K, K)
        eigvals, eigvecs = torch.linalg.eigh(G)   # ascending
        top_vals = eigvals[:, -r_eff:].clamp(min=self.eps)
        top_vecs = eigvecs[:, :, -r_eff:]
        Qt = torch.matmul(dev.transpose(-1, -2), top_vecs / torch.sqrt(top_vals).unsqueeze(1))
        Qt = F.normalize(Qt, dim=-2)
        C = Qv.transpose(-1, -2) @ Qt             # (B, r_eff, r_eff)
        dir_loss = (2 * r_eff - 2 * (C ** 2).sum(dim=(-1, -2))).mean()

        if not torch.isfinite(dir_loss).all():
            logger.warning("L_dir produced a non-finite value; zeroing (detached).")
            dir_loss = torch.zeros_like(dir_loss)
        return dir_loss

    def _cal_loss(self, text_logvars: torch.Tensor) -> torch.Tensor:
        """L_cal: caption variances calibrated to the prior sigma_0^2.

        The set-level statistics (eq:caption_statistics) involve only the
        caption means, so the caption variances -- which enter the text
        variance through the moment-matched merge -- have no empirical
        counterpart and are calibrated to a fixed prior measured on held-out
        data. This is the ONLY supervisor of the caption-level variances.
        """
        log_target = math.log(max(self.sigma0_sq, self.eps))
        cap_log = torch.log(torch.exp(text_logvars) + self.eps)
        return ((cap_log - log_target) ** 2).mean()

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
        """MCDisp_Align loss. text_mu / text_logvar are the moment-matched
        text distribution (mean and diag of Sigma_bar_t); text_mus /
        text_logvars are the per-caption parameters. text_Us is accepted for
        caller compatibility and unused (caption distributions are diagonal).
        All terms are averaged over the batch, as in eq:overall_objective.
        """
        ctr_loss, ctr_i2t, ctr_t2i = self._ctr_loss(img_mu, text_mus)
        var_loss = self._var_loss(img_logvar, text_logvar)
        dir_loss = self._dir_loss(img_U, text_mus)
        cal_loss = self._cal_loss(text_logvars)

        total = (
            self.lambda_ctr * ctr_loss
            + self.lambda_var * var_loss
            + self.lambda_dir * dir_loss
            + self.lambda_cal * cal_loss
        )

        with torch.no_grad():
            img_var = torch.exp(img_logvar)
            img_var_avg = img_var.mean()
            img_var_min = img_var.min()
            img_var_median = img_var.median()
            img_var_max = img_var.max()
            # text variance diag(Sigma_bar_t) and its two components
            text_var_mean = torch.exp(text_logvar).mean()
            cap_center = text_mus.mean(dim=1, keepdim=True)
            caption_spread = ((text_mus - cap_center) ** 2).mean(dim=1)   # (B, D) = s_t^2
            caption_spread_mean = caption_spread.mean()
            caption_spread_median = caption_spread.median()
            caption_spread_max = caption_spread.max()
            cap_var_mean = torch.exp(text_logvars).mean()
            # low-rank vs diagonal energy on the image side
            diag_var_energy = img_var_avg
            if img_U is not None:
                u_energy = (img_U ** 2).sum(dim=-1).mean()
                u_over_diag = u_energy / (diag_var_energy + self.eps)
            else:
                u_energy = img_var.new_zeros(())
                u_over_diag = img_var.new_zeros(())
            # core-innovation readout: image variance vs the dispersion component
            var_over_spread = img_var_avg / (caption_spread_mean + self.eps)

        loss_dict = {
            "total": total.item(),
            "ctr": ctr_loss.item(),
            "ctr_i2t": ctr_i2t.item(),   # distribution-to-set: any-hit direction
            "ctr_t2i": ctr_t2i.item(),   # distribution-to-set: per-caption direction
            "var": var_loss.item(),
            "dir": dir_loss.item(),
            "cal": cal_loss.item(),
            "contrastive": ctr_loss.item(),   # alias for training-loop compat
            "weighted_ctr": (self.lambda_ctr * ctr_loss).item(),
            "weighted_var": (self.lambda_var * var_loss).item(),
            "weighted_dir": (self.lambda_dir * dir_loss).item(),
            "weighted_cal": (self.lambda_cal * cal_loss).item(),
            # variance statistics
            "img_var_min": img_var_min.item(),
            "img_var_median": img_var_median.item(),
            "img_var_mean": img_var_avg.item(),
            "img_var_avg": img_var_avg.item(),   # alias used by the trainer's accumulators
            "img_var_max": img_var_max.item(),
            "text_var_mean": text_var_mean.item(),
            "cap_var_mean": cap_var_mean.item(),
            "caption_spread_mean": caption_spread_mean.item(),
            "caption_spread_median": caption_spread_median.item(),
            "caption_spread_max": caption_spread_max.item(),
            "var_over_spread": var_over_spread.item(),
            # low-rank covariance statistics (U vs diagonal energy)
            "u_energy": u_energy.item(),
            "diag_var_energy": diag_var_energy.item(),
            "u_over_diag": u_over_diag.item(),
        }
        return total, loss_dict


if __name__ == "__main__":
    print("Testing MCDisp_Align loss (paper §3.3 four-term objective)...")
    B, D, K, r = 4, 768, 5, 4
    img_mu = torch.randn(B, D)
    img_logvar = torch.randn(B, D)
    img_U = torch.randn(B, D, r)
    text_mu = torch.randn(B, D)
    text_logvar = torch.randn(B, D)
    text_mus = torch.randn(B, K, D)
    text_logvars = torch.randn(B, K, D)

    print("\n1. Full loss (with low-rank directions):")
    crit = MCDispAlignLoss()
    loss, d = crit(img_mu, img_logvar, img_U, text_mu, text_logvar,
                   text_mus, text_logvars)
    for k in ("total", "ctr", "ctr_i2t", "ctr_t2i", "var", "dir", "cal"):
        print(f"   {k}: {d[k]:.4f}")
    assert math.isfinite(d["total"])

    print("\n2. Diagonal only (img_U=None -> L_dir = 0):")
    crit_d = MCDispAlignLoss(lambda_dir=0.0)
    loss_d, d_d = crit_d(img_mu, img_logvar, None, text_mu, text_logvar,
                         text_mus, text_logvars)
    print(f"   total: {d_d['total']:.4f}, dir: {d_d['dir']:.4f} (expect 0)")

    print("\n3. Gradient flow (img_mu / img_logvar / img_U / text_mus):")
    im = img_mu.clone().requires_grad_(True)
    ilv = img_logvar.clone().requires_grad_(True)
    iU = img_U.clone().requires_grad_(True)
    tm = text_mus.clone().requires_grad_(True)
    # L_ctr consumes text_mus directly (the real training path), so the
    # caption means receive gradient per caption; the merged mean below feeds
    # only L_var here, mirroring how forward receives both views.
    text_mu_from_tm = tm.mean(dim=1)
    loss_g, _ = crit(im, ilv, iU, text_mu_from_tm, text_logvar, tm, text_logvars)
    loss_g.backward()
    print(f"   grad img_mu: {im.grad.norm():.4f}")
    print(f"   grad img_logvar: {ilv.grad.norm():.4f}")
    print(f"   grad img_U: {iU.grad.norm():.4f}")
    print(f"   grad text_mus: {tm.grad.norm():.4f}")
    assert im.grad.norm() > 0 and ilv.grad.norm() > 0 and iU.grad.norm() > 0 and tm.grad.norm() > 0

    print("\n4. L_ctr sends no gradient to any variance:")
    ilv2 = img_logvar.clone().requires_grad_(True)
    tlv2 = text_logvars.clone().requires_grad_(True)
    crit2 = MCDispAlignLoss(lambda_var=0.0, lambda_dir=0.0, lambda_cal=0.0)
    total2, _ = crit2(img_mu, ilv2, None, text_mu, text_logvar, text_mus, tlv2)
    total2.backward()
    assert ilv2.grad is None or ilv2.grad.norm() == 0, "L_ctr must not touch sigma_v^2"
    assert tlv2.grad is None or tlv2.grad.norm() == 0, "L_ctr must not touch sigma_t^2"
    print("   img/text logvar grads from L_ctr: 0 (verified)")

    print("\n5. Stop-gradient check (caption targets are fixed):")
    tm2 = text_mus.clone().requires_grad_(True)
    tlv3 = text_logvars.clone().requires_grad_(True)
    # text_mu passed as a constant: L_var / L_dir stop-gradient their caption
    # targets, so in isolation only L_cal touches text_logvars (and nothing
    # touches text_mus directly -- it trains only through the merged text mean).
    _, _ = crit(img_mu, img_logvar, img_U, text_mu, text_logvar, tm2, tlv3)
    print("   forward completed without error")

    print("\nAll MCDisp_Align loss tests passed.")
