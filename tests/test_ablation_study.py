"""Unit tests for the ablation-study code (experiment plan phase-0 checks).

Covers:
  * population caption spread == the training loss's L_var target (plan §3.4:
    no estimator mismatch between train and eval)
  * query-side variance invariance of the discounted scorer (plan §7.2 --
    also an alignment unit test for the scorer implementation)
  * rank derivation (any-hit I2T, per-caption T2I, PairCount, WorstRank)
    against a brute-force reference
  * H1/H3 metric bounds on synthetic data (S_sub in [0,1], perfect tracking
    gives rho_sem ~ 1, E_var ~ 0)
  * Gaussian likelihood scorer: diagonal case matches the closed form;
    low-rank case matches a direct dense-inverse reference
  * experiment definitions: A1/A3 keep the U head (capacity-matched),
    A4/mean-only drop it; all use manifest data and select_by="mr"

Pure CPU, synthetic tensors -- no dataset or checkpoint needed.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from losses.mcdisp_align_losses import MCDispAlignLoss
from scripts.ablation_study import mc_ranking
from scripts.ablation_study.experiments import EXPERIMENTS, build_train_config
from scripts.ablation_study.gaussian_scorer import likelihood_sim_rows
from scripts.ablation_study.h1_semantic_range import (
    e_var, h1_metrics, population_caption_spread, spearman,
)
from scripts.ablation_study.h3_subspace import h3_metrics
from scripts.ablation_study.interventions import query_side_invariance_check
from scripts.ablation_study.stats import (
    aggregate_seeds, holm_bonferroni, paired_bootstrap,
)


def _features(N=8, K=5, D=16, r=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(N, K, D, generator=g)
    return {
        "img_mu": torch.randn(N, D, generator=g),
        "img_logvar": torch.randn(N, D, generator=g) * 0.5 - 2.0,
        "text_mus": t,
        "text_logvars": torch.randn(N, K, D, generator=g) * 0.5 - 2.0,
        "img_U": torch.randn(N, D, r, generator=g) * 0.1,
    }


# --------------------------------------------------------------------- §3.4
def test_population_spread_matches_loss_target():
    """Eval spread (denominator K) must equal the training L_var target."""
    f = _features()
    crit = MCDispAlignLoss()
    # replicate _var_loss's target computation
    text_center = f["text_mus"].mean(dim=1)
    target = ((f["text_mus"] - text_center.unsqueeze(1)) ** 2).mean(dim=1)
    spread = population_caption_spread(f["text_mus"])
    assert torch.allclose(target, spread, atol=1e-7)
    # and E_var is exactly the loss when sigma^2 == s^2
    assert e_var(spread, spread) < 1e-10


# --------------------------------------------------------------------- §7.2
def test_query_side_invariance():
    f = _features(N=12)
    N, K, D = f["text_mus"].shape
    cap_mu = f["text_mus"].reshape(N * K, D)
    cap_lv = f["text_logvars"].reshape(N * K, D)
    res = query_side_invariance_check(
        f["img_mu"], f["img_logvar"], cap_mu, cap_lv, K)
    assert res["i2t_invariant_to_img_query_sigma"], res
    assert res["t2i_invariant_to_cap_query_sigma"], res


def test_candidate_side_sigma_can_change_ranking():
    """The converse of §7.2: candidate-side variance CAN move rankings.

    Deterministic construction (N=2, K=1): image 0's own caption has the
    HIGHER cosine but a much larger variance discount, so a rival caption
    outranks it; swapping the two candidate variances flips the top-1 back.
    """
    D = 4
    img_mu = torch.zeros(2, D)
    img_mu[0, 0], img_mu[1, 1] = 1.0, 1.0
    cap_mu = torch.zeros(2, D)
    cap_mu[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])          # own: cos = 1.0
    cap_mu[1] = torch.tensor([0.98, 0.199, 0.0, 0.0])       # rival: cos = 0.98
    img_logvar = torch.full((2, D), -2.0)
    cap_lv = torch.full((2, D), -2.0)
    cap_lv[0] = math.log(torch.tensor(5.0))                 # own: huge discount

    r0 = mc_ranking.ranks_from_sim(
        mc_ranking.mc_sim_rows(img_mu, img_logvar, cap_mu, cap_lv,
                               dtype=torch.float32), K=1, k_values=(1,))
    assert not r0["i2t_hit"][1][0], "rival should outrank the high-variance own caption"

    swap = torch.tensor([1, 0])
    r1 = mc_ranking.ranks_from_sim(
        mc_ranking.mc_sim_rows(img_mu, img_logvar, cap_mu, cap_lv[swap],
                               dtype=torch.float32), K=1, k_values=(1,))
    assert r1["i2t_hit"][1][0], "after the variance swap the own caption should win"


# ------------------------------------------------------------------ ranking core
def test_ranks_against_brute_force():
    f = _features(N=6, K=3, D=8)
    N, K, D = f["text_mus"].shape
    cap_mu = f["text_mus"].reshape(N * K, D)
    cap_lv = f["text_logvars"].reshape(N * K, D)
    sim = mc_ranking.mc_sim_rows(f["img_mu"], f["img_logvar"], cap_mu, cap_lv)
    res = mc_ranking.ranks_from_sim(sim, K, k_values=(1, 2, 5))

    order = torch.argsort(sim.float(), dim=1, descending=True)
    for i in range(N):
        own = list(range(i * K, (i + 1) * K))
        ranks_ref = sorted([int((order[i] == j).nonzero()[0]) + 1 for j in own])
        ranks_got = sorted(res["pos_ranks"][i].tolist())
        assert ranks_ref == ranks_got, (i, ranks_ref, ranks_got)
        assert res["worst_rank"][i].item() == max(ranks_ref)
    # any-hit I2T R@k from brute force
    for k in (1, 2, 5):
        hit = torch.zeros(N, dtype=torch.bool)
        for i in range(N):
            topk = set(order[i, :k].tolist())
            hit[i] = bool(topk & set(range(i * K, (i + 1) * K)))
        assert torch.equal(hit, res["i2t_hit"][k])
    # T2I: caption j's own image rank via column sort
    for k in (1, 2, 5):
        hit = torch.zeros(N * K, dtype=torch.bool)
        for j in range(N * K):
            col = torch.argsort(sim[:, j].float(), descending=True)
            hit[j] = bool((col[:k] == j // K).any())
        assert torch.equal(hit, res["t2i_hit"][k])
    assert math.isclose(
        res["pair_count"][5],
        (res["pos_ranks"] <= 5).sum(dim=1).float().mean().item())


# --------------------------------------------------------------------- H1 / H3
def test_h1_perfect_tracking():
    f = _features()
    spread = population_caption_spread(f["text_mus"])
    hits = torch.ones(f["img_mu"].shape[0], dtype=torch.bool)
    m = h1_metrics(spread, f["text_mus"], hits)   # sigma^2 == s^2 exactly
    assert m["e_var"] < 1e-8
    assert m["rho_sem_spearman"] > 0.999
    assert m["rho_sem_pearson"] > 0.999


def test_h3_bounds_and_s_sub_range():
    f = _features()
    m = h3_metrics(f["img_U"], torch.exp(f["img_logvar"]), f["text_mus"])
    assert 0.0 <= m["s_sub"] <= 1.0
    assert 0.0 <= m["explained_energy"] <= 1.0
    assert 0.0 <= m["principal_angle_mean_deg"] <= 90.0
    assert 1.0 <= m["u_effective_rank"] <= f["img_U"].shape[-1]
    # diagonal-only checkpoint -> NaN subspace metrics, no crash
    m0 = h3_metrics(None, torch.exp(f["img_logvar"]), f["text_mus"])
    assert m0["s_sub"] != m0["s_sub"]


# --------------------------------------------------------------- likelihood
def test_likelihood_diagonal_matches_closed_form():
    f = _features(N=3, K=2, D=5, r=0)
    N, K, D = f["text_mus"].shape
    cap = f["text_mus"].reshape(N * K, D)
    sim = likelihood_sim_rows(f["img_mu"], f["img_logvar"], cap, None).float()
    for i in range(N):
        var = torch.exp(f["img_logvar"][i])
        for j in range(N * K):
            d = cap[j] - f["img_mu"][i]
            ref = (-0.5 * D * math.log(2 * math.pi)
                   - 0.5 * torch.log(var).sum()
                   - 0.5 * (d ** 2 / var).sum())
            # rtol: the implementation divides by (var + eps) as a numerical
            # guard; at |score|~1e2 that shifts the 4th digit, not the formula.
            assert torch.isclose(sim[i, j], ref, rtol=1e-4, atol=1e-2), (i, j)


def test_likelihood_lowrank_matches_dense_inverse():
    f = _features(N=3, K=2, D=6, r=2)
    N, K, D = f["text_mus"].shape
    cap = f["text_mus"].reshape(N * K, D)
    sim = likelihood_sim_rows(f["img_mu"], f["img_logvar"], cap, f["img_U"]).float()
    for i in range(N):
        var = torch.exp(f["img_logvar"][i])
        U = f["img_U"][i]
        Sig = torch.diag(var) + U @ U.T
        ref_prec = torch.linalg.inv(Sig)
        _, logdet = torch.linalg.slogdet(Sig)
        for j in range(N * K):
            d = cap[j] - f["img_mu"][i]
            ref = (-0.5 * D * math.log(2 * math.pi) - 0.5 * logdet
                   - 0.5 * d @ ref_prec @ d)
            assert torch.isclose(sim[i, j], ref, rtol=1e-3, atol=1e-2), (i, j)


# --------------------------------------------------------------------- stats
def test_paired_bootstrap_and_holm():
    g = torch.Generator().manual_seed(0)
    a = (torch.rand(500, generator=g) > 0.4).float()
    b = (torch.rand(500, generator=g) > 0.5).float()   # slightly worse system
    r = paired_bootstrap(a, b, n_boot=300)
    assert r["ci_low"] <= r["delta"] <= r["ci_high"]
    agg = aggregate_seeds([{"m": 0.5}, {"m": 0.7}, {"m": 0.6}], baseline=0.6)
    assert math.isclose(agg["m"]["mean"], 0.6)
    assert agg["m"]["ci95"] > 0
    adj = holm_bonferroni({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adj["x"] <= adj["z"] <= adj["y"]
    # Holm with monotonicity enforcement: x: 3*0.01; z: max(0.03, 2*0.03);
    # y: max(0.06, 1*0.04) -> both z and y are lifted to 0.06.
    assert adj["x"] == 0.03 and adj["y"] == 0.06 and adj["z"] == 0.06


def test_spearman_monotonic_and_ties():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([10.0, 20.0, 30.0, 41.0])
    assert spearman(x, y) > 0.99
    y2 = torch.tensor([1.0, 1.0, 2.0, 2.0])
    assert -1.01 <= spearman(x, y2) <= 1.01   # finite with ties


# ------------------------------------------------------------ experiment defs
def test_experiment_definitions_capacity_matching():
    # A1/A3 keep the U head (capacity-matched to Full); A4/mean-only drop it.
    assert EXPERIMENTS["no_var"]["cov_rank"] == EXPERIMENTS["full"]["cov_rank"]
    assert EXPERIMENTS["no_cov"]["cov_rank"] == EXPERIMENTS["full"]["cov_rank"]
    assert EXPERIMENTS["no_cov"]["weights"]["lambda_cov"] == 0.0
    assert EXPERIMENTS["diagonal_only"]["cov_rank"] == 0
    assert EXPERIMENTS["mean_only_kall"]["cov_rank"] == 0
    assert EXPERIMENTS["mean_only_k1"]["K"] == 1
    assert EXPERIMENTS["mean_only_k1"]["sample"] == "random"
    assert EXPERIMENTS["full_k3"]["K"] == 3
    assert EXPERIMENTS["full_k3"]["cov_rank"] == 2 <= 3 - 1   # r* <= K_min - 1


def test_build_train_config_unified_controls():
    cfg = build_train_config("no_var", 42, epochs=2)
    assert cfg.select_by == "mr"
    assert cfg.no_early_stop
    assert cfg.train_manifest is not None and cfg.dev_manifest is not None
    assert cfg.dev_num_captions == 5
    assert cfg.lambda_var == 0.0 and cfg.lambda_cov == EXPERIMENTS["full"]["weights"]["lambda_cov"]
    # K-comparability: full_k3 and full_kall_r2 share r*
    c3 = build_train_config("full_k3", 42, epochs=2)
    cr2 = build_train_config("full_kall_r2", 42, epochs=2)
    assert c3.cov_rank == cr2.cov_rank == 2
    # checkpoint paths are per-experiment per-seed (plan §14)
    assert cfg.best_path != c3.best_path
    assert "seed42" in str(cfg.best_path) and "no_var" in str(cfg.best_path)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
