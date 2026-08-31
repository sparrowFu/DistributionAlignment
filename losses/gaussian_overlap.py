"""Pairwise Gaussian-overlap scores between image and caption Gaussians.

score_ij = log ∫ p_v(z|x_i) p_t(z|c_j) dz  (公共常数 D·log2π 舍去)
         = -0.5 [ δ_ij^T C_ij^{-1} δ_ij + log|C_ij| ],
C_ij = diag(d_i^v + d_j^t) + U_i U_i^T  (caption 侧对角).

Woodbury / matrix-determinant-lemma 形式（O(B·N·(D+r²))，无 D×D 逆）:
  v = U^T diag(1/c) δ,  M = I_r + U^T diag(1/c) U,
  mahal = δ^T diag(1/c) δ − v^T M^{-1} v,
  log|C| = Σ_d log c + log|M|.
整体除以 2D 做逐维归一化：量级不随嵌入维度漂移（A11 允许并对齐缩放
Mahalanobis 与 logdet）；再除以 tau_match 进入对比 logits。
"""

import torch


def gaussian_overlap_scores(
    img_mu: torch.Tensor,        # (B, D)
    img_diag_var: torch.Tensor,  # (B, D)  对角分量 d_v（非完整方差）
    img_U: torch.Tensor,         # (B, D, r) 或 None
    cap_mu: torch.Tensor,        # (N, D)
    cap_diag_var: torch.Tensor,  # (N, D)
    per_dim_norm: bool = True,
) -> torch.Tensor:
    """返回 (B, N) 分数，越高越匹配。输入均为原始分布坐标，方差为正数。"""
    B, D = img_mu.shape
    delta = img_mu[:, None, :] - cap_mu[None, :, :]            # (B, N, D)
    c = img_diag_var[:, None, :] + cap_diag_var[None, :, :]    # (B, N, D)
    inv_c = 1.0 / c
    mahal = (delta * delta * inv_c).sum(-1)                    # (B, N)
    logdet = torch.log(c).sum(-1)                              # (B, N)

    if img_U is not None:
        U = img_U
        v = torch.einsum("bdr,bnd,bnd->bnr", U, inv_c, delta)  # (B, N, r)
        M = torch.einsum("bdr,bnd,bds->bnrs", U, inv_c, U)     # (B, N, r, r)
        M = M + torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
        w = torch.linalg.solve(M, v.unsqueeze(-1)).squeeze(-1)  # (B, N, r)
        mahal = mahal - (v * w).sum(-1)
        chol = torch.linalg.cholesky(M)
        logdet = logdet + 2.0 * torch.log(
            torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)

    score = -0.5 * (mahal + logdet)
    if per_dim_norm:
        score = score / D
    return score
