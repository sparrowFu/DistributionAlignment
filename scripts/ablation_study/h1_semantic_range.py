"""H1 metrics: does the image variance represent the multi-caption semantic
range? (Plan §6.2.)

All targets use the POPULATION caption spread (denominator K, plan §3.4) --
the same definition as the training loss ``_var_loss`` (asserted by a unit
test), so there is no train/eval estimator mismatch.

  E_var        log-space variance matching error        (§6.2.1, lower=better)
  rho_sem      sample-level Spearman(mean sigma^2, mean s^2) (+ Pearson)
               (§6.2.2)
  pairwise-div correlation vs mean sigma^2              (§6.2.3)
  per-dim      median/IQR/valid-fraction/median-r2 of the per-dimension
               corr(sigma^2_d, s^2_d)                   (§6.2.4)
  stratified   5 equal-frequency semantic-range bins: per-bin I2T R@1,
               Spearman(bin, R@1), linear-trend R^2, big-vs-small range gap
               (§6.2.5)
"""

from typing import Dict

import torch
import torch.nn.functional as F


def population_caption_spread(text_mus: torch.Tensor) -> torch.Tensor:
    """s^2 = (1/K) sum_k (mu_k - mean)^2 per dim, (N, D). Denominator K (§3.4)."""
    center = text_mus.mean(dim=1, keepdim=True)
    return ((text_mus - center) ** 2).mean(dim=1)


def e_var(img_var: torch.Tensor, spread: torch.Tensor, eps: float = 1e-6) -> float:
    """Log-space variance matching error E_var (plan §6.2.1)."""
    return ((torch.log(img_var + eps) - torch.log(spread + eps)) ** 2).mean().item()


def _rank_average(x: torch.Tensor) -> torch.Tensor:
    """Average ranks (1-based) with tie handling, no scipy dependency."""
    n = x.numel()
    order = torch.argsort(x)
    sx = x[order]
    out = torch.empty(n, dtype=torch.float64)
    start = 0
    for i in range(1, n + 1):
        if i == n or sx[i] != sx[start]:
            out[start:i] = (start + 1 + i) / 2.0     # ranks start+1 .. i
            start = i
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = out
    return ranks


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0 or not torch.isfinite(denom):
        return float("nan")
    return float((a * b).sum() / denom)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    return _pearson(_rank_average(a), _rank_average(b))


def pairwise_diversity(text_mus: torch.Tensor) -> torch.Tensor:
    """Div_i = 1 - mean_{p<q} cos(mu_p, mu_q), (N,). Plan §6.2.3."""
    N, K, D = text_mus.shape
    mu = F.normalize(text_mus, dim=-1)
    div = torch.zeros(N, dtype=torch.float64)
    for p in range(K):
        for q in range(p + 1, K):
            div += 1.0 - (mu[:, p, :] * mu[:, q, :]).sum(dim=-1).double()
    return div / max(K * (K - 1) // 2, 1)


def per_dim_consistency(img_var: torch.Tensor, spread: torch.Tensor,
                        min_target_std: float = 1e-8) -> Dict[str, float]:
    """Per-dimension corr(sigma^2_d, s^2_d) statistics (plan §6.2.4).

    Near-constant target dims are excluded (their correlation is undefined);
    the excluded fraction is reported as the complement of ``valid_dim_frac``.
    """
    N, D = img_var.shape
    corrs, r2s = [], []
    valid = 0
    for d in range(D):
        t = spread[:, d]
        if t.std() < min_target_std or img_var[:, d].std() < min_target_std:
            continue
        valid += 1
        r = _pearson(img_var[:, d], t)
        if r == r:                       # skip NaN
            corrs.append(r)
            r2s.append(r * r)
    if not corrs:
        return {"median_corr": float("nan"), "iqr_corr": float("nan"),
                "valid_dim_frac": 0.0, "median_r2": float("nan")}
    c = torch.tensor(corrs)
    q25, q50, q75 = torch.quantile(c, torch.tensor([0.25, 0.5, 0.75]))
    return {
        "median_corr": float(q50),
        "iqr_corr": float(q75 - q25),
        "valid_dim_frac": valid / D,
        "median_r2": float(torch.tensor(r2s).median()),
    }


def stratified_retrieval(
    range_per_image: torch.Tensor,
    i2t_hit_at_1: torch.Tensor,
    n_bins: int = 5,
) -> Dict[str, float]:
    """Semantic-range-binned I2T R@1 + trend stats (plan §6.2.5).

    Args:
        range_per_image: (N,) semantic range per image (e.g. mean_d sigma^2).
        i2t_hit_at_1:    (N,) bool, I2T R@1 hit under the primary scorer.
    """
    N = range_per_image.numel()
    order = torch.argsort(range_per_image)
    bin_edges = [int(round(b * N / n_bins)) for b in range(n_bins + 1)]
    bin_r1, bin_idx = [], []
    for b in range(n_bins):
        sel = order[bin_edges[b]:bin_edges[b + 1]]
        bin_r1.append(i2t_hit_at_1[sel].float().mean().item())
        bin_idx.append(b)
    # Spearman(bin index, bin R@1) over the n_bins points
    bi = torch.tensor(bin_idx, dtype=torch.float64)
    br = torch.tensor(bin_r1, dtype=torch.float64)
    rho = spearman(bi, br)
    # linear trend R^2
    xm, ym = bi.mean(), br.mean()
    sxx = ((bi - xm) ** 2).sum()
    r2 = float("nan")
    if sxx > 0:
        sxy = ((bi - xm) * (br - ym)).sum()
        slope = sxy / sxx
        ss_res = ((br - ym - slope * (bi - xm)) ** 2).sum()
        ss_tot = ((br - ym) ** 2).sum()
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        **{f"bin{b}_i2t_r1": v for b, v in enumerate(bin_r1)},
        "spearman_bin_r1": rho,
        "trend_r2": r2,
        "big_minus_small_r1": bin_r1[-1] - bin_r1[0],
    }


def h1_metrics(
    img_var: torch.Tensor,          # (N, D) predicted image variance
    text_mus: torch.Tensor,         # (N, K, D) per-caption means
    i2t_hit_at_1: torch.Tensor,     # (N,) bool primary-scorer I2T R@1 hits
) -> Dict[str, float]:
    """All H1 metrics for one checkpoint's features (plan §6.2)."""
    spread = population_caption_spread(text_mus)
    range_img = img_var.mean(dim=1).double()          # mean_d sigma^2
    range_tgt = spread.mean(dim=1).double()           # mean_d s^2
    div = pairwise_diversity(text_mus)
    return {
        "e_var": e_var(img_var, spread),
        "rho_sem_spearman": spearman(range_img, range_tgt),
        "rho_sem_pearson": _pearson(range_img, range_tgt),
        "div_corr_spearman": spearman(range_img, div),
        "div_corr_pearson": _pearson(range_img, div),
        **per_dim_consistency(img_var, spread),
        **stratified_retrieval(range_img, i2t_hit_at_1),
    }
