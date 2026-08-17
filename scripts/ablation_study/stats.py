"""Statistics for the ablation study (plan §9).

  aggregate_seeds    mean / std / delta-vs-Full / %-change / 95% CI (t-based,
                     small-n critical values; no scipy dependency)
  paired_bootstrap   plan §9.3: resample the SAME test queries for two systems,
                     I2T and T2I separately; percentile CI + two-sided
                     bootstrap p-value for delta > 0
  holm_bonferroni    multiple-comparison correction over the three primary
                     contrasts (F vs A1/A2/A3)
"""

from typing import Dict, Sequence, Tuple

import torch


# t_{0.975, df} for df = n_seeds - 1 (n up to 8); fallback 1.96 for larger n.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def _t_crit(n: int) -> float:
    return _T975.get(n - 1, 1.96)


def aggregate_seeds(
    per_seed: Sequence[Dict[str, float]],
    baseline: float = None,
    keys: Sequence[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate a metric dict across seeds.

    For each metric key present in every record: mean, std, 95% CI half-width,
    and (when ``baseline`` is given) absolute and % delta vs the Full mean.
    """
    keys = keys or (sorted(set.intersection(*[set(r) for r in per_seed]))
                    if per_seed else [])
    n = len(per_seed)
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = torch.tensor([float(r[k]) for r in per_seed], dtype=torch.float64)
        mean = float(vals.mean())
        std = float(vals.std(unbiased=True)) if n > 1 else 0.0
        ci = _t_crit(n) * std / (n ** 0.5) if n > 1 else 0.0
        entry = {"mean": mean, "std": std, "ci95": ci, "n_seeds": n}
        if baseline is not None:
            entry["delta_vs_full"] = mean - baseline
            entry["pct_vs_full"] = (mean - baseline) / baseline * 100.0 if baseline != 0 else float("nan")
        out[k] = entry
    return out


def paired_bootstrap(
    hits_a: torch.Tensor,
    hits_b: torch.Tensor,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired bootstrap over the same query set (plan §9.3).

    Args:
        hits_a, hits_b: (Q,) 0/1 per-query hit vectors for two systems, same
            queries in the same order (e.g. F vs A1 I2T R@1 hits).
        n_boot: resampling rounds.
    """
    a = hits_a.to(torch.float64)
    b = hits_b.to(torch.float64)
    q = a.numel()
    delta = float((a - b).mean())
    g = torch.Generator().manual_seed(seed)
    boots = []
    for _ in range(n_boot):
        idx = torch.randint(0, q, (q,), generator=g)
        boots.append(float((a[idx] - b[idx]).mean()))
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    # two-sided bootstrap p-value for delta == 0
    beyond = sum(1 for x in boots if (x > 0) != (delta > 0))
    p = 2.0 * min(beyond, n_boot - beyond) / n_boot
    return {
        "delta": delta,
        "ci_low": lo,
        "ci_high": hi,
        "ci_excludes_zero": not (lo <= 0.0 <= hi),
        "p_boot": max(p, 1.0 / n_boot),
    }


def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values (plan §9.3)."""
    items = [(k, p) for k, p in p_values.items() if p == p]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adjusted[k] = min(1.0, running)
    for k, p in p_values.items():
        if p != p:                      # NaN metrics stay NaN
            adjusted[k] = float("nan")
    return adjusted


def bootstrap_primary_contrasts(
    contrast_hits: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    n_boot: int = 1000,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Run paired bootstrap for the three primary contrasts and Holm-correct.

    Args:
        contrast_hits: {contrast_name: (i2t_a, i2t_b, t2i_a, t2i_b)} hit
            vectors, a = Full, b = the weakened config.
    Returns:
        (raw bootstrap results per contrast+direction, Holm-adjusted p-values
        over the per-contrast mR deltas).
    """
    raw: Dict[str, Dict[str, float]] = {}
    p_vals: Dict[str, float] = {}
    for name, (i2t_a, i2t_b, t2i_a, t2i_b) in contrast_hits.items():
        r_i2t = paired_bootstrap(i2t_a, i2t_b, n_boot=n_boot)
        r_t2i = paired_bootstrap(t2i_a, t2i_b, n_boot=n_boot)
        raw[f"{name}:i2t"] = r_i2t
        raw[f"{name}:t2i"] = r_t2i
        raw[f"{name}:mr"] = {
            "delta": (r_i2t["delta"] + r_t2i["delta"]) / 2.0,
            "ci_low": (r_i2t["ci_low"] + r_t2i["ci_low"]) / 2.0,
            "ci_high": (r_i2t["ci_high"] + r_t2i["ci_high"]) / 2.0,
            "ci_excludes_zero": r_i2t["ci_excludes_zero"] and r_t2i["ci_excludes_zero"],
        }
        p_vals[name] = max(r_i2t["p_boot"], r_t2i["p_boot"])
    return raw, holm_bonferroni(p_vals)
