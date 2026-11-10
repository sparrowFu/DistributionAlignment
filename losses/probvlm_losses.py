"""
ProbVLM Loss Functions (Upadhyay et al., ICCV 2023)

Faithful port of the official ExplainableML/ProbVLM src/losses.py
(GenGaussLoss + TempCombLoss), plus the project-side helpers:
probvlm_total_loss (batch loss composition against the frozen CLIP
targets) and generalized_gaussian_logvar (per-dimension
generalized-Gaussian -> Gaussian log-variance conversion used at
evaluation time).
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class GenGaussLoss(nn.Module):
    """Negative log-likelihood of a per-dimension generalized Gaussian.

    Port of the official GenGaussLoss (the *active* code path: the residual
    is linear in |mean - target|; the powered form is commented out
    upstream). The eps guards and residual clamp follow the official
    defaults.
    """

    def __init__(
        self,
        reduction: str = 'mean',
        alpha_eps: float = 1e-4,
        beta_eps: float = 1e-4,
        resi_min: float = 1e-4,
        resi_max: float = 1e3,
    ) -> None:
        super(GenGaussLoss, self).__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Reduction not supported: {reduction}")
        self.reduction = reduction
        self.alpha_eps = alpha_eps
        self.beta_eps = beta_eps
        self.resi_min = resi_min
        self.resi_max = resi_max

    def forward(
        self,
        mean: Tensor,
        one_over_alpha: Tensor,
        beta: Tensor,
        target: Tensor,
    ) -> Tensor:
        one_over_alpha1 = one_over_alpha + self.alpha_eps
        beta1 = beta + self.beta_eps

        resi = torch.abs(mean - target)
        resi = (resi * one_over_alpha1 * beta1).clamp(min=self.resi_min, max=self.resi_max)

        log_one_over_alpha = torch.log(one_over_alpha1)
        log_beta = torch.log(beta1)
        lgamma_beta = torch.lgamma(torch.pow(beta1, -1))

        l = resi - log_one_over_alpha + lgamma_beta - log_beta

        if self.reduction == 'mean':
            return l.mean()
        return l.sum()


class TempCombLoss(nn.Module):
    """T1 * L1 + T2 * GenGauss, the official ProbVLM reconstruction loss."""

    def __init__(
        self,
        reduction: str = 'mean',
        alpha_eps: float = 1e-4,
        beta_eps: float = 1e-4,
        resi_min: float = 1e-4,
        resi_max: float = 1e3,
    ) -> None:
        super(TempCombLoss, self).__init__()
        self.L_GenGauss = GenGaussLoss(
            reduction=reduction,
            alpha_eps=alpha_eps, beta_eps=beta_eps,
            resi_min=resi_min, resi_max=resi_max,
        )
        self.L_l1 = nn.L1Loss(reduction=reduction)

    def forward(
        self,
        mean: Tensor,
        one_over_alpha: Tensor,
        beta: Tensor,
        target: Tensor,
        t1: float = 1.0,
        t2: float = 5e-2,
    ) -> Tensor:
        l1 = self.L_l1(mean, target)
        l2 = self.L_GenGauss(mean, one_over_alpha, beta, target)
        return t1 * l1 + t2 * l2


def generalized_gaussian_logvar(
    one_over_alpha: Tensor,
    beta: Tensor,
    eps: float = 1e-4,
    clamp: float = 20.0,
) -> Tensor:
    """Convert generalized-Gaussian parameters to a Gaussian log-variance.

    Var[GGD(mu, alpha, beta)] = alpha^2 * Gamma(3/beta) / Gamma(1/beta)
    per dimension, with alpha = 1 / (1/alpha). Known cases: beta=2 (Gaussian)
    gives sigma^2 = alpha^2 / 2, beta=1 (Laplacian) gives sigma^2 = 2 alpha^2.

    The eps guards match the official loss; the +/-`clamp` bound keeps the
    log-variance finite when beta sits near 0 early in training.
    """
    b = beta + eps
    logvar = (
        -2.0 * torch.log(one_over_alpha + eps)
        + torch.lgamma(3.0 / b)
        - torch.lgamma(1.0 / b)
    )
    return logvar.clamp(min=-clamp, max=clamp)


def probvlm_total_loss(
    outputs: Dict[str, Tensor],
    criterion: TempCombLoss,
    t1: float,
    t2: float,
    cross_modal_lambda: float,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compose the official ProbVLM batch loss from a model forward dict.

    loss = loss_i + loss_t + cross_modal_lambda * (loss_i4t + loss_t4i)

    where the intra-modal terms reconstruct the frozen CLIP features of the
    same modality and the cross-modal terms regress the paired frozen
    features of the *other* modality.
    """
    loss_i = criterion(
        outputs["img_mu"], outputs["img_1alpha"], outputs["img_beta"],
        outputs["i_features"], t1=t1, t2=t2,
    )
    loss_t = criterion(
        outputs["text_mu"], outputs["txt_1alpha"], outputs["txt_beta"],
        outputs["t_features"], t1=t1, t2=t2,
    )
    loss_i4t = criterion(
        outputs["img_mu"], outputs["img_1alpha"], outputs["img_beta"],
        outputs["t_features"], t1=t1, t2=t2,
    )
    loss_t4i = criterion(
        outputs["text_mu"], outputs["txt_1alpha"], outputs["txt_beta"],
        outputs["i_features"], t1=t1, t2=t2,
    )

    loss = loss_i + loss_t + cross_modal_lambda * (loss_i4t + loss_t4i)

    info = {
        "loss": loss.item(),
        "loss_i": loss_i.item(),
        "loss_t": loss_t.item(),
        "loss_i4t": loss_i4t.item(),
        "loss_t4i": loss_t4i.item(),
    }
    return loss, info


if __name__ == "__main__":
    # Numeric sanity checks on random tensors.
    torch.manual_seed(0)
    mu = torch.randn(4, 8)
    one_over_alpha = torch.rand(4, 8)
    beta = torch.rand(4, 8) + 0.5
    target = torch.randn(4, 8)

    L = GenGaussLoss()
    L2 = TempCombLoss()
    print("GenGauss:", L(mu, one_over_alpha, beta, target).item())
    print("TempComb(t1=1, t2=5e-2):", L2(mu, one_over_alpha, beta, target, t1=1.0, t2=5e-2).item())

    # beta=2 => sigma^2 = alpha^2 / 2 => logvar = 2*log(alpha) - log(2)
    # (eps=0.0 for the pure closed form; the default eps shifts by O(1e-4))
    inv_alpha = torch.full((2, 3), 4.0)   # alpha = 1/4
    beta2 = torch.full((2, 3), 2.0)
    expected = 2 * torch.log(torch.tensor(0.25)) - torch.log(torch.tensor(2.0))
    got = generalized_gaussian_logvar(inv_alpha, beta2, eps=0.0)
    assert torch.allclose(got, expected.expand_as(got), atol=1e-5), (got, expected)
    print("beta=2 logvar identity: OK")
