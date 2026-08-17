"""H2 metrics: does the image distribution cover the caption SET? (Plan §6.3.)

  CoverageRate        fraction of positive (image, own-caption) pairs whose
                     per-D-normalized squared Mahalanobis distance is within
                     the coverage margin m_pos            (§6.3.1)
  Mahalanobis stats   mean / median / q90 of d^2/D over positive pairs, plus
                     the high-diversity (top-quartile Div) subset (§6.3.2)
  PairCount@K / Worst-caption Rank come from the shared ranking core
                     (§6.3.3 / §6.3.4)
  hard subset         high caption-diversity quartile: I2T R@1, PairCount@10,
                     CoverageRate, WorstRank median            (§6.3.5)

The Mahalanobis distance reuses the training loss's Woodbury implementation
(``MCDispAlignLoss._mahalanobis``) so train and eval measure the SAME metric.
"""

from typing import Dict

import torch

from losses.mcdisp_align_losses import MCDispAlignLoss
from scripts.ablation_study.h1_semantic_range import pairwise_diversity


def positive_mahalanobis(
    img_mu: torch.Tensor,      # (N, D)
    img_var: torch.Tensor,     # (N, D)
    img_U: torch.Tensor,       # (N, D, r) or None
    text_mus: torch.Tensor,    # (N, K, D)
    eps: float = 1e-6,
    chunk: int = 512,
) -> torch.Tensor:
    """(N, K) per-D-normalized squared Mahalanobis of each own caption under
    its image's Sigma_v = diag(sigma^2) + U U^T."""
    N, K, D = text_mus.shape
    out = torch.empty(N, K)
    mahal = MCDispAlignLoss._mahalanobis
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        diff = (text_mus[s:e] - img_mu[s:e].unsqueeze(1)).reshape((e - s) * K, D)
        var = img_var[s:e].unsqueeze(1).expand(e - s, K, D).reshape((e - s) * K, D)
        U = None
        if img_U is not None:
            r = img_U.shape[-1]
            U = img_U[s:e].unsqueeze(1).expand(e - s, K, D, r).reshape((e - s) * K, D, r)
        out[s:e] = (mahal(diff, var, U, eps) / D).reshape(e - s, K)
    return out


def h2_metrics(
    img_mu: torch.Tensor,
    img_var: torch.Tensor,
    img_U: torch.Tensor,
    text_mus: torch.Tensor,
    ranking: Dict[str, object],       # from mc_ranking.ranks_from_sim (mc scorer)
    m_pos: float = 1.0,
    k_pair: int = 10,
) -> Dict[str, float]:
    """All H2 metrics for one checkpoint (plan §6.3)."""
    d2 = positive_mahalanobis(img_mu, img_var, img_U, text_mus)
    flat = d2.flatten()
    covered = d2 <= m_pos

    div = pairwise_diversity(text_mus)
    hard_thr = torch.quantile(div, 0.75)
    hard = div >= hard_thr

    i2t1 = ranking["i2t_hit"][1]
    worst = ranking["worst_rank"].double()

    def _q(t: torch.Tensor, q: float) -> float:
        return float(torch.quantile(t.float(), q))

    return {
        # §6.3.1 positive coverage rate
        "coverage_rate": covered.float().mean().item(),
        # §6.3.2 Mahalanobis distance statistics
        "mahal_mean": flat.mean().item(),
        "mahal_median": _q(flat, 0.5),
        "mahal_q90": _q(flat, 0.9),
        "mahal_mean_highdiv": d2[hard].mean().item() if hard.any() else float("nan"),
        # §6.3.3 pair counts (mc scorer; cosine variant available via ranking calls)
        **{f"pair_count@{k}": v for k, v in ranking["pair_count"].items()},
        # §6.3.4 worst-caption rank
        "worst_rank_median": _q(worst, 0.5),
        "worst_rank_q90": _q(worst, 0.9),
        # §6.3.5 high-diversity hard subset
        "hard_i2t_r1": i2t1[hard].float().mean().item() if hard.any() else float("nan"),
        "hard_pair_count@10": (ranking["pos_ranks"][hard] <= k_pair).sum(dim=1).float().mean().item()
        if hard.any() else float("nan"),
        "hard_coverage_rate": covered[hard].float().mean().item() if hard.any() else float("nan"),
        "hard_worst_rank_median": _q(worst[hard], 0.5) if hard.any() else float("nan"),
    }
