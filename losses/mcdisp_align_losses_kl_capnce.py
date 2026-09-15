"""MCDisp-Align KL loss + auxiliary per-caption NCE (method extension v2).

Motivation: the KL objective aligns the image distribution with the merged
caption-SET distribution; per-caption ranking was never directly optimized
(pilot evidence: unfrozen KL per-caption I2T R@1 trails CLIP fine-tuning by
~7pt). This extension adds ONE auxiliary term on top of the unchanged KL
loss: a CLIP-style bidirectional InfoNCE between the image mean and EACH
individual caption mean over the batch's B*K captions:

    cap_nce = 0.5 * [ mean_{i,k} CE(logits_i2c[i], pos = i*K + k)
                    + mean_{j}    CE(logits_c2i[j], pos = j // K) ]
    logits = cos(mu_v, mu_t) / tau_cap        (own temperature, fixed)

    total = KL_total (unchanged, all weights/terms inherited)
          + lambda_cap_nce * cap_nce

Design contract:
  * lambda_cap_nce = 0 reduces EXACTLY to MCDispAlignKLLoss (verified in
    tests); the parent's staged lambda_kl ramp and every sub-loss are
    untouched -- the auxiliary term rides no schedule (always on, like ctr).
  * The term uses the SAME mu-head outputs the model already produces; no
    architecture change, no new parameters (tau_cap is a fixed float).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from losses.mcdisp_align_losses_kl import MCDispAlignKLLoss


class MCDispAlignKLCapNCELoss(MCDispAlignKLLoss):
    """KL objective + auxiliary per-caption InfoNCE (see module docstring)."""

    def __init__(self, *args, lambda_cap_nce: float = 0.5,
                 tau_cap: float = 0.07, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_cap_nce = lambda_cap_nce
        self.tau_cap = tau_cap

    def _cap_nce(self, img_mu: torch.Tensor, text_mus: torch.Tensor) -> torch.Tensor:
        """Bidirectional per-caption InfoNCE over the batch's B*K captions."""
        B, K, D = text_mus.shape
        img_n = F.normalize(img_mu, dim=-1)                       # (B, D)
        cap_n = F.normalize(text_mus.reshape(B * K, D), dim=-1)   # (B*K, D)
        logits = img_n @ cap_n.T / self.tau_cap                   # (B, B*K)

        BK = B * K
        idx = torch.arange(BK, device=img_mu.device)
        # i2c: every (image i, caption k) query against all B*K captions;
        # the positive column of row r = i*K + k is exactly r.
        i2c = F.cross_entropy(logits.repeat_interleave(K, dim=0), idx,
                              reduction="mean")
        # c2i: every caption against all B images; positive = caption // K.
        c2i = F.cross_entropy(logits.T, idx // K, reduction="mean")
        return 0.5 * (i2c + c2i)

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
        loss, loss_dict = super().forward(
            img_mu, img_logvar, img_U, text_mu, text_logvar,
            text_mus, text_logvars, text_Us)
        cap_nce = self._cap_nce(img_mu, text_mus)
        total = loss + self.lambda_cap_nce * cap_nce
        out = dict(loss_dict)
        out["total"] = total
        out["cap_nce"] = cap_nce.detach()
        out["weighted_cap_nce"] = (self.lambda_cap_nce * cap_nce).detach()
        return total, out
