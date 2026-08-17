"""Full-Gaussian-likelihood retrieval scorer (plan §6.4.5, §3.5).

    score(x_i, c_j) = log N(mu_j^t ; mu_i^v, Sigma_i^v),
    Sigma_i^v = diag(sigma_i^2) + U_i U_i^T

COORDINATE CONSISTENCY (plan §3.5): means, diagonal variances and U are all
used in the model's RAW output space -- no L2 normalization of the means
(which would rescale the space without transforming Sigma). The log-determinant
uses the exact low-rank identity

    log|diag(s) + U U^T| = sum_d log s_d + log det(I_r + U^T diag(1/s) U)

and the quadratic term reuses the Woodbury solve pattern from the training
loss. One (N, M) score matrix serves both directions: rows rank captions
(I2T), columns rank images (T2I). Diagonal-only checkpoints (U=None) skip the
low-rank correction.
"""

from typing import Optional

import torch


@torch.no_grad()
def likelihood_sim_rows(
    img_mu: torch.Tensor,       # (N, D) raw means
    img_logvar: torch.Tensor,   # (N, D)
    cap_mu_flat: torch.Tensor,  # (M, D) raw caption means
    img_U: Optional[torch.Tensor] = None,   # (N, D, r)
    eps: float = 1e-6,
    img_chunk: int = 64,
    cap_chunk: int = 2048,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """(N, M) log N(cap_mu_j; img_mu_i, Sigma_i) for every image-caption pair.

    Doubly chunked (images x captions); peak compute memory ~
    ``img_chunk * cap_chunk * D`` floats; the output matrix itself is
    ``N * M * dtype`` (float32 by default -- likelihood values reach
    |score|~1e3 at D=768 where float16's ~1.0 resolution would corrupt
    close rankings).
    """
    N, D = img_mu.shape
    M = cap_mu_flat.shape[0]
    out = torch.empty(N, M, dtype=dtype)

    imu = img_mu.to(device)
    ilv = img_logvar.to(device)
    iU = img_U.to(device) if img_U is not None else None
    cap = cap_mu_flat.to(device)

    const = -0.5 * D * torch.log(torch.tensor(2.0 * torch.pi, device=device))

    for s in range(0, N, img_chunk):
        e = min(s + img_chunk, N)
        var = torch.exp(ilv[s:e])                              # (c, D)
        inv = 1.0 / (var + eps)
        rows = torch.empty(e - s, M, dtype=torch.float32, device=device)
        # per-image log-determinant (c,)
        if iU is not None:
            U = iU[s:e]                                        # (c, D, r)
            W = inv.unsqueeze(-1) * U                          # (c, D, r) = D^-1 U
            S = torch.eye(U.shape[-1], device=device).unsqueeze(0) + U.transpose(-1, -2) @ W
            signs, ldets = torch.slogdet(S)
            logdet = torch.log(var + eps).sum(dim=-1) + ldets.view(-1)
        else:
            U, W, S = None, None, None
            logdet = torch.log(var + eps).sum(dim=-1)
        for cs in range(0, M, cap_chunk):
            ce = min(cs + cap_chunk, M)
            diff = cap[cs:ce].unsqueeze(0) - imu[s:e].unsqueeze(1)      # (c, m, D)
            a = inv.unsqueeze(1) * diff                                 # D^-1 d
            quad_diag = (diff * a).sum(dim=-1)                          # (c, m)
            if U is not None:
                UtA = torch.einsum("cdr,cmd->cmr", U, a)                # (c, m, r)
                z = torch.linalg.solve(S, UtA.transpose(-1, -2)).transpose(-1, -2)
                corr = torch.einsum("cdr,cmr->cmd", W, z)
                quad = quad_diag - (diff * corr).sum(dim=-1)
            else:
                quad = quad_diag
            rows[:, cs:ce] = const - 0.5 * logdet.unsqueeze(1) - 0.5 * quad
        out[s:e] = rows.to(dtype).cpu()
    return out
