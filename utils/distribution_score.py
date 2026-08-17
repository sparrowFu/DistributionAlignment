"""Distribution-likelihood matching score: log-likelihood of the text mean under the image Gaussian.

The retrieval/matching score is the log-likelihood of the text mean under the
image Gaussian distribution:

    score(image, text) = log N(text_mean ; image_mean, Sigma_image)

with Sigma_image = diag(sigma^2) + U U^T (U is the low-rank covariance factor).
Higher score = better match. This single function is used by BOTH the training
contrastive loss and every retrieval evaluation, so train and eval share one
definition of "match".

Sigma^{-1} is applied via the Woodbury identity and log|Sigma| via the
matrix-determinant lemma, so only an r x r solve / determinant is needed
(r = covariance rank) -- the diagonal + low-rank structure is exact, not
approximated.
"""

from typing import Optional

import torch


def image_text_loglik_matrix(
    img_mean: torch.Tensor,          # (N, D) image means (distribution centers)
    img_var: torch.Tensor,           # (N, D) image diagonal variances sigma^2 (> 0)
    img_U: Optional[torch.Tensor],   # (N, D, r) low-rank factor, or None (diagonal)
    text_mean: torch.Tensor,         # (M, D) text means (points to score)
    eps: float = 1e-6,
    per_dim_normalize: bool = True,
    use_logdet: bool = True,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Return the (N, M) score matrix S[n, m] = log N(text_m ; img_n, Sigma_n).

    Higher = better match. Means are assumed already L2-normalized by the caller
    (so means, var, and U live in the same normalized space).

    Args:
        img_mean/img_var/img_U: per-image Gaussian parameters. img_U None means
            Sigma = diag(img_var).
        text_mean: text means scored under each image distribution.
        eps: numerical floor on var (also guards the Woodbury matrix).
        per_dim_normalize: divide the Mahalanobis term by D, decoupling its
            scale from dimensionality and from the variance magnitude.
        use_logdet: include the -0.5 * log|Sigma| normalization term.
        chunk_size: chunk over the N (image) axis to bound peak memory.

    Returns:
        (N, M) matrix; higher = better match.
    """
    N, D = img_mean.shape
    M = text_mean.shape[0]
    r = 0 if img_U is None else img_U.shape[-1]
    norm = float(D) if per_dim_normalize else 1.0

    var = torch.clamp(img_var, min=eps)       # (N, D) floor at eps
    inv_var = 1.0 / var                       # (N, D)
    logdet_diag = torch.log(var).sum(dim=-1)  # (N,)

    eye_r = (torch.eye(r, device=img_mean.device, dtype=img_mean.dtype)
             if r > 0 else None)

    scores = img_mean.new_zeros(N, M)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        img_mean_c = img_mean[start:end]                  # (C, D)
        diff = text_mean.unsqueeze(0) - img_mean_c.unsqueeze(1)   # (C, M, D)
        inv_var_c = inv_var[start:end].unsqueeze(1)       # (C, 1, D)
        a = diff * inv_var_c                              # (C, M, D) = D^{-1} diff
        term1 = (diff * a).sum(dim=-1)                    # (C, M) = diff^T D^{-1} diff

        if img_U is not None:
            U_c = img_U[start:end]                        # (C, D, r)
            Winv = inv_var[start:end].unsqueeze(-1) * U_c  # (C, D, r) = D^{-1} U
            S = eye_r + U_c.transpose(-1, -2) @ Winv      # (C, r, r) = I + U^T D^{-1} U
            b = U_c.transpose(-1, -2) @ a.transpose(-1, -2)  # (C, r, M) = U^T D^{-1} diff
            z = torch.linalg.solve(S, b)                  # (C, r, M) = S^{-1} b
            term2 = (b * z).sum(dim=1)                    # (C, M) = b^T S^{-1} b
            mahal = term1 - term2                         # (C, M)
            logdet_S = torch.linalg.slogdet(S)[1]         # (C,)
        else:
            mahal = term1                                 # (C, M)
            logdet_S = torch.zeros(end - start, device=img_mean.device, dtype=img_mean.dtype)

        logdet = logdet_diag[start:end] + logdet_S        # (C,) log|Sigma|
        s = -0.5 * mahal / norm                           # (C, M)
        if use_logdet:
            s = s - 0.5 * logdet.unsqueeze(1)             # (C, M)
        scores[start:end] = s

    return scores
