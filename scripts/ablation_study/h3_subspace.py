"""H3 metrics: did the low-rank covariance learn the caption semantic-change
DIRECTIONS? (Plan §6.4.)

  S_sub            normalized subspace overlap ||Qv^T Qt||_F^2 / r_eff in [0,1]
                   (1 = identical subspaces)                            (§6.4.1)
  ExplainedEnergy  tr(Qv^T Ct Qv) / tr(Ct) of the caption-deviation
                   covariance Ct                                    (§6.4.2)
  principal angles mean/median/max of the Qv-Qt principal angles       (§6.4.3)
  U statistics     U/diag energy ratio + participation-ratio effective
                   rank of U (guards against a collapsed-but-present head)
                                                                        (§6.4.4)

The caption target subspaces mirror the training ``_cov_loss`` construction
(raw caption means centered by their set mean; effective rank capped at
min(r, K-1, D)), so train and eval optimize/measure the same object.
"""

from typing import Dict

import torch
import torch.nn.functional as F


def _caption_target_basis(text_mus: torch.Tensor, r_eff: int, eps: float = 1e-6):
    """Q_t (N, D, r_eff): top deviation directions of the raw centered captions."""
    B, K, D = text_mus.shape
    dev = text_mus - text_mus.mean(dim=1, keepdim=True)          # (B, K, D)
    G = dev @ dev.transpose(-1, -2) + eps * torch.eye(K, device=dev.device)
    eigvals, eigvecs = torch.linalg.eigh(G)                      # ascending
    top_vals = eigvals[:, -r_eff:].clamp(min=eps)
    top_vecs = eigvecs[:, :, -r_eff:]
    Qt = torch.matmul(dev.transpose(-1, -2), top_vecs / torch.sqrt(top_vals).unsqueeze(1))
    return F.normalize(Qt, dim=-2)


def _effective_rank(U: torch.Tensor) -> float:
    """Participation ratio (s1^2+s2^2+...)^2 / (s1^4+s2^4+...) of U's singular
    values, per image; returns the mean. Range [1, r]."""
    s = torch.linalg.svdvals(U.float())                          # (N, min(D, r))
    s2 = s ** 2
    num = (s2.sum(dim=-1)) ** 2
    den = (s2 ** 2).sum(dim=-1)
    return float((num / den.clamp(min=1e-12)).mean())


def h3_metrics(
    img_U: torch.Tensor,        # (N, D, r) or None (diagonal-only checkpoints)
    img_var: torch.Tensor,      # (N, D)
    text_mus: torch.Tensor,     # (N, K, D)
) -> Dict[str, float]:
    """All H3 metrics for one checkpoint (plan §6.4). Returns NaN subspace
    values when there is no U head (A4/diagonal-only, mean-only configs)."""
    u_energy_ratio = float("nan")
    if img_U is not None:
        u_energy = (img_U ** 2).sum(dim=-1).mean()
        u_energy_ratio = float(u_energy / (img_var.mean() + 1e-12))

    if img_U is None:
        return {
            "s_sub": float("nan"), "explained_energy": float("nan"),
            "principal_angle_mean_deg": float("nan"),
            "principal_angle_median_deg": float("nan"),
            "principal_angle_max_deg": float("nan"),
            "u_over_diag": float("nan"), "u_effective_rank": float("nan"),
        }

    N, D, r = img_U.shape
    K = text_mus.shape[1]
    if K <= 1:
        raise ValueError("H3 subspace metrics are undefined for K<=1 (no deviation directions)")

    r_eff = min(r, K - 1, D)
    Qv, _ = torch.linalg.qr(img_U.float())
    Qv = Qv[:, :, :r_eff]
    Qt = _caption_target_basis(text_mus.float(), r_eff)

    C = Qv.transpose(-1, -2) @ Qt                    # (N, r_eff, r_eff)
    s_sub = float(((C ** 2).sum(dim=(-1, -2)) / r_eff).mean())

    dev = text_mus.float() - text_mus.float().mean(dim=1, keepdim=True)
    Ct = torch.matmul(dev.transpose(-1, -2), dev) / K          # (N, D, D)
    tr_all = Ct.diagonal(dim1=-2, dim2=-1).sum(dim=-1)         # (N,)
    # trace(Qv^T Ct Qv): operand dims (a,d1),(d1,d2),(d2,a)
    proj = torch.einsum("nij,njk,nki->n", Qv.transpose(-1, -2), Ct, Qv)
    explained = float((proj / tr_all.clamp(min=1e-12)).mean())

    sv = torch.linalg.svdvals(C).clamp(0.0, 1.0)               # cosines of principal angles
    angles = torch.rad2deg(torch.arccos(sv))                   # (N, r_eff)
    return {
        "s_sub": s_sub,
        "explained_energy": explained,
        "principal_angle_mean_deg": float(angles.mean()),
        "principal_angle_median_deg": float(angles.median()),
        "principal_angle_max_deg": float(angles.max()),
        "u_over_diag": u_energy_ratio,
        "u_effective_rank": _effective_rank(img_U),
    }
