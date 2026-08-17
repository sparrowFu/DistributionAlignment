"""Inference-stage scorer ablation + interventions (plan §7).

All on ONE Full checkpoint -- no retraining:

  Scorers (§7 table):
    cosine                          means only
    semantic-range-weighted cosine  means + scalar semantic range (the
                                    uncertainty-discounted training score)
    full Gaussian likelihood        means + per-dim variance + U (raw space)

  Interventions (§7.1), applied to the image side (the text side is the
  candidate side of I2T and the query side of T2I):
    constant sigma^2   every image gets the train-set mean variance vector
    shuffled sigma^2   image variance vectors are randomly permuted across
                       images (sample-mismatched ranges)
    zero U / shuffled U analogous for the low-rank factors

  Side-awareness (§7.2): for the RANGE-WEIGHTED scorer the query-side scale is
  a row constant that cannot change that row's ranking -- an image-side sigma
  intervention can only move T2I (sigma as candidate discount) and a text-side
  one only I2T. ``query_side_invariance_check`` asserts exactly this property
  and doubles as an alignment unit test for the scorer implementation. The
  LIKELIHOOD scorer is NOT row-constant (Mahalanobis + logdet vary per
  candidate), so interventions move both directions there.
"""

from typing import Dict

import torch

from scripts.ablation_study import mc_ranking
from scripts.ablation_study.gaussian_scorer import likelihood_sim_rows


def query_side_invariance_check(
    img_mu: torch.Tensor, img_logvar: torch.Tensor,
    cap_mu_flat: torch.Tensor, cap_lv_flat: torch.Tensor,
    K: int, tau: float = 0.07,
) -> Dict[str, bool]:
    """Plan §7.2: permuting the QUERY-side variance must not change that
    direction's ranking under the discounted-cosine scorer.

    I2T queries are images: permuting image sigma^2 leaves every image's ROW
    scaled by a constant -> identical I2T ranking. T2I queries are captions:
    permuting caption sigma^2 leaves every caption's row constant-scaled ->
    identical T2I ranking. Returns which invariances hold (both must be True;
    a False means a scorer/alignment bug).
    """
    g = torch.Generator().manual_seed(0)
    N = img_mu.shape[0]
    # float32: the invariance must survive rounding, not be broken by it.
    f32 = torch.float32

    sim = mc_ranking.mc_sim_rows(img_mu, img_logvar, cap_mu_flat, cap_lv_flat,
                                 tau=tau, dtype=f32)
    r0 = mc_ranking.ranks_from_sim(sim, K, k_values=(1, 5, 10))

    perm = torch.randperm(N, generator=g)
    sim_perm_img = mc_ranking.mc_sim_rows(
        img_mu, img_logvar[perm], cap_mu_flat, cap_lv_flat, tau=tau, dtype=f32)
    r1 = mc_ranking.ranks_from_sim(sim_perm_img, K, k_values=(1, 5, 10))

    # Permute ONLY the caption query variances (means stay put): each caption's
    # scale is a row constant for its own T2I ranking, so T2I must be unchanged.
    perm_cap = torch.randperm(cap_mu_flat.shape[0], generator=g)
    sim_perm_cap = mc_ranking.mc_sim_rows(
        img_mu, img_logvar, cap_mu_flat, cap_lv_flat[perm_cap], tau=tau, dtype=f32)
    r2 = mc_ranking.ranks_from_sim(sim_perm_cap, K, k_values=(1, 5, 10))

    return {
        "i2t_invariant_to_img_query_sigma": bool(
            torch.equal(r0["i2t_hit"][1], r1["i2t_hit"][1])),
        "t2i_invariant_to_cap_query_sigma": bool(
            torch.equal(r0["t2i_hit"][1], r2["t2i_hit"][1])),
    }


def intervention_suite(
    img_mu: torch.Tensor, img_logvar: torch.Tensor,
    text_mus: torch.Tensor, text_logvars: torch.Tensor,
    img_U: torch.Tensor = None,
    k_values=(1, 5, 10), tau: float = 0.07,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Scorer x intervention table (plan §7, result table §12.4).

    Returns {setting_name: {"i2t_r1", "t2i_r1", "mr", ...}} where
    ``mr`` is the plan's 6-recall mean for that setting.
    """
    N, K, D = text_mus.shape
    cap_mu = text_mus.reshape(N * K, D)
    cap_lv = text_logvars.reshape(N * K, D)
    g = torch.Generator().manual_seed(seed)

    img_var = torch.exp(img_logvar)
    perm = torch.randperm(N, generator=g)

    def _summarize(sim, name, out):
        r = mc_ranking.ranks_from_sim(sim, K, k_values=k_values)
        row = {f"i2t_r@{k}": r["i2t_r"][k] for k in k_values}
        row.update({f"t2i_r@{k}": r["t2i_r"][k] for k in k_values})
        row["mr"] = mc_ranking.mr_from(r, k_values)
        out[name] = row

    out: Dict[str, Dict[str, float]] = {}

    # ---- scorer table on the TRUE parameters ----
    sim_cos = mc_ranking.cos_sim_rows(img_mu, img_logvar, cap_mu, cap_lv)
    _summarize(sim_cos, "scorer:cosine", out)
    sim_mc = mc_ranking.mc_sim_rows(img_mu, img_logvar, cap_mu, cap_lv, tau=tau)
    _summarize(sim_mc, "scorer:range_weighted", out)
    sim_lik = likelihood_sim_rows(img_mu, img_logvar, cap_mu, img_U)
    _summarize(sim_lik, "scorer:gaussian_likelihood", out)

    # ---- sigma^2 interventions (image side), range-weighted scorer ----
    mean_lv = img_logvar.mean(dim=0, keepdim=True).expand(N, D).contiguous()
    _summarize(
        mc_ranking.mc_sim_rows(img_mu, mean_lv, cap_mu, cap_lv, tau=tau),
        "range_weighted:constant_img_sigma", out)
    _summarize(
        mc_ranking.mc_sim_rows(img_mu, img_logvar[perm], cap_mu, cap_lv, tau=tau),
        "range_weighted:shuffled_img_sigma", out)

    # ---- U interventions, likelihood scorer (U only enters here) ----
    if img_U is not None:
        zero_U = torch.zeros_like(img_U)
        _summarize(
            likelihood_sim_rows(img_mu, img_logvar, cap_mu, zero_U),
            "likelihood:zero_U", out)
        _summarize(
            likelihood_sim_rows(img_mu, img_logvar, cap_mu, img_U[perm]),
            "likelihood:shuffled_U", out)
        # constant sigma^2 under the likelihood scorer, for completeness
        _summarize(
            likelihood_sim_rows(img_mu, mean_lv, cap_mu, img_U),
            "likelihood:constant_img_sigma", out)
        _summarize(
            likelihood_sim_rows(img_mu, img_logvar[perm], cap_mu, img_U[perm]),
            "likelihood:shuffled_img_sigma", out)

    return out
