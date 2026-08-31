"""
Tests for the four-group objective: L_match (gaussian overlap by default,
cosine as the cosine_match ablation baseline), L_mu, the dispersion group
(L_var + R_prior), and L_dir with the spectral rank guard.

The first six tests are the COSINE-branch contract. Pure CPU, follows the
repo's main() runner pattern.

Reference behavior (design doc §4.1):
  L_match = (L_i2t + L_t2i) / 2
  L_i2t = (1/B) Σᵢ [ logsumexp_all_BK − logsumexp_own_K ]  (= −log P(top-1 ∈ own))
  L_t2i = per-caption cross entropy over the B images (label = own image)
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from losses.gaussian_overlap import gaussian_overlap_scores
from losses.mcdisp_align_losses import MCDispAlignLoss

TAU = 0.07


def _cosine_only(**kwargs) -> MCDispAlignLoss:
    """Criterion with only the cosine L_match active (mu/var/reg/dir zeroed)."""
    return MCDispAlignLoss(match_score="cosine", lambda_mu=0.0, lambda_var=0.0,
                           lambda_reg=0.0, lambda_dir=0.0, **kwargs)


def _gauss_match_only(**kwargs) -> MCDispAlignLoss:
    """Criterion with only the gaussian L_match active (A16 isolation)."""
    return MCDispAlignLoss(match_score="gaussian", lambda_mu=0.0, lambda_var=0.0,
                           lambda_reg=0.0, lambda_dir=0.0, **kwargs)


def _forward_match(crit, img_mu, text_mus):
    """Call forward with dummy variance tensors; return the loss dict."""
    B, K, D = text_mus.shape
    _, d = crit(
        img_mu, torch.zeros(B, D), None,
        torch.zeros(B, D), torch.zeros(B, D),
        text_mus, torch.zeros(B, K, D),
    )
    return d


# ---------------------------------------------------------------------- cosine L_match

def test_orthogonal_hand_check():
    """B orthonormal image means; every caption equals its image vector.
    Own cosine = 1, cross cosine = 0 -> closed-form values:
      L_i2t = log(K·e^{1/τ} + K(B−1)) − (log K + 1/τ)
      L_t2i = log(e^{1/τ} + (B−1)) − 1/τ
    """
    B, K, D = 3, 2, 4
    img_mu = torch.zeros(B, D)
    img_mu[0, 0] = img_mu[1, 1] = img_mu[2, 2] = 1.0
    text_mus = img_mu.unsqueeze(1).repeat(1, K, 1)          # (B, K, D)

    d = _forward_match(_cosine_only(), img_mu, text_mus)

    e = math.exp(1.0 / TAU)
    exp_i2t = math.log(K * e + K * (B - 1)) - (math.log(K) + 1.0 / TAU)
    exp_t2i = math.log(e + (B - 1)) - 1.0 / TAU
    assert abs(d["match_i2t"] - exp_i2t) < 1e-4, (d["match_i2t"], exp_i2t)
    assert abs(d["match_t2i"] - exp_t2i) < 1e-4, (d["match_t2i"], exp_t2i)
    assert abs(d["match"] - 0.5 * (exp_i2t + exp_t2i)) < 1e-4


def test_per_caption_gradient_nonzero():
    """Root-cause fix: EVERY caption mean receives gradient from L_match.
    (The old mean-only InfoNCE sent none: text_mus was not an endpoint.)"""
    torch.manual_seed(0)
    B, K, D = 4, 5, 8
    img_mu = torch.randn(B, D, requires_grad=True)
    text_mus = torch.randn(B, K, D, requires_grad=True)
    crit = _cosine_only()
    loss, _ = crit(
        img_mu, torch.zeros(B, D), None,
        torch.zeros(B, D), torch.zeros(B, D),
        text_mus, torch.zeros(B, K, D),
    )
    loss.backward()
    assert img_mu.grad is not None and img_mu.grad.norm() > 0
    for i in range(B):
        for k in range(K):
            assert text_mus.grad[i, k].norm() > 0, f"caption ({i},{k}) got no grad"


def test_cosine_match_no_gradient_to_variances():
    """Cosine-branch claim preserved: the score involves only means."""
    torch.manual_seed(1)
    B, K, D = 4, 5, 8
    img_lv = torch.randn(B, D, requires_grad=True)
    txt_lv = torch.randn(B, K, D, requires_grad=True)
    crit = _cosine_only()
    loss, _ = crit(
        torch.randn(B, D), img_lv, None,
        torch.zeros(B, D), torch.zeros(B, D),
        torch.randn(B, K, D), txt_lv,
    )
    loss.backward()
    assert img_lv.grad is None or img_lv.grad.norm() == 0
    assert txt_lv.grad is None or txt_lv.grad.norm() == 0


def test_k1_matches_pairwise_infonce():
    """Boundary: K=1 degenerates to the classic bidirectional InfoNCE
    on (μᵥ, μᵗ) — i.e. the OLD formula. Guards against sign/form drift."""
    torch.manual_seed(2)
    B, D = 4, 8
    img_mu = torch.randn(B, D)
    cap = torch.randn(B, 1, D)
    d = _forward_match(_cosine_only(), img_mu, cap)

    img_n = F.normalize(img_mu, dim=-1)
    cap_n = F.normalize(cap.squeeze(1), dim=-1)
    sim = (img_n @ cap_n.T) / TAU
    labels = torch.arange(B)
    expected = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))
    assert abs(d["match"] - expected.item()) < 1e-5


def test_lambda_match_zero_no_match_grad():
    torch.manual_seed(3)
    B, K, D = 4, 5, 8
    img_mu = torch.randn(B, D, requires_grad=True)
    crit = _cosine_only(lambda_match=0.0)
    loss, _ = crit(
        img_mu, torch.zeros(B, D), None,
        torch.zeros(B, D), torch.zeros(B, D),
        torch.randn(B, K, D), torch.zeros(B, K, D),
    )
    loss.backward()
    assert img_mu.grad is None or img_mu.grad.norm() == 0


def test_b1_degenerates_to_zero():
    """B=1 (reachable: no drop_last + filter_none_collate can shrink batches):
    own set = entire gallery, single-image gallery -> both terms exactly 0."""
    torch.manual_seed(4)
    K, D = 5, 8
    img_mu = torch.randn(1, D)
    text_mus = torch.randn(1, K, D)
    d = _forward_match(_cosine_only(), img_mu, text_mus)
    assert d["match_i2t"] == 0.0 and d["match_t2i"] == 0.0 and d["match"] == 0.0


# ------------------------------------------------------------------ gaussian L_match (A16)

def test_match_gaussian_ranks_own_above_foreign():
    """The gaussian overlap score must rank captions drawn near the image's own
    Gaussian above foreign images' captions, and the per-caption t2i direction
    must beat the random baseline ln(B)."""
    torch.manual_seed(5)
    B, K, D = 3, 2, 8
    img_mu = torch.zeros(B, D)
    img_mu[torch.arange(B), torch.arange(B)] = 1.0        # orthonormal image means
    text_mus = img_mu.unsqueeze(1) + 0.05 * torch.randn(B, K, D)  # own: small offset
    # (a foreign caption of image j sits on a DIFFERENT basis vector -> large
    #  Mahalanobis distance from image i)

    img_logvar = math.log(0.01) * torch.ones(B, D)
    text_logvars = math.log(0.01) * torch.ones(B, K, D)
    _, d = _gauss_match_only()(
        img_mu, img_logvar, None,
        torch.zeros(B, D), torch.zeros(B, D),
        text_mus, text_logvars,
    )

    # independent check through the scorer itself: own blocks vs foreign blocks
    scores = gaussian_overlap_scores(
        img_mu, torch.exp(img_logvar), None,
        text_mus.reshape(B * K, D), torch.exp(text_logvars.reshape(B * K, D)))
    blocks = scores.reshape(B, B, K)
    idx = torch.arange(B)
    own_mean = blocks[idx, idx].mean().item()
    foreign_mean = blocks[~torch.eye(B, dtype=torch.bool)].mean().item()
    assert own_mean > foreign_mean + 1.0, (own_mean, foreign_mean)
    assert d["match_t2i"] < math.log(B), (d["match_t2i"], math.log(B))


def test_match_supervises_caption_variance():
    """A16 core acceptance: the gaussian L_match sends gradient to the caption
    variances (text_logvars NOT detached) -- the matching objective itself
    supervises sigma^2 -- and to every other distribution parameter."""
    torch.manual_seed(6)
    B, K, D, r = 4, 5, 8, 3
    crit = _gauss_match_only()
    img_mu = torch.randn(B, D, requires_grad=True)
    img_logvar = (-4 + 0.5 * torch.randn(B, D)).requires_grad_(True)
    img_U = (0.1 * torch.randn(B, D, r)).requires_grad_(True)
    text_mus = torch.randn(B, K, D, requires_grad=True)
    text_logvars = (-4 + 0.5 * torch.randn(B, K, D)).requires_grad_(True)
    loss, _ = crit(
        img_mu, img_logvar, img_U,
        torch.zeros(B, D), torch.zeros(B, D),
        text_mus, text_logvars,
    )
    loss.backward()
    for name, t in (("text_logvars", text_logvars), ("img_mu", img_mu),
                    ("img_logvar", img_logvar), ("img_U", img_U),
                    ("text_mus", text_mus)):
        assert t.grad is not None, f"{name} got no grad at all"
        assert torch.isfinite(t.grad).all(), f"{name} grad not finite"
        assert t.grad.norm() > 0, f"{name} got no gradient from gaussian L_match"


# ------------------------------------------------------------------------ L_mu (A01)

def test_mu_loss_raw_coordinate():
    """A01: L_mu aligns the centers in RAW coordinates: equal means -> 0, a 10x
    scaled image mean -> clearly positive (cosine would see no difference); the
    text target is detached; lambda_mu=0 removes the mu gradient entirely."""
    torch.manual_seed(7)
    B, K, D = 3, 4, 8
    img_mu = torch.randn(B, D)
    text_mu = img_mu.clone()
    crit = MCDispAlignLoss(match_score="cosine", lambda_match=0.0,
                           lambda_mu=1.0, lambda_var=0.0, lambda_reg=0.0,
                           lambda_dir=0.0)
    _, d = crit(img_mu, torch.zeros(B, D), None, text_mu, torch.zeros(B, D),
                torch.randn(B, K, D), torch.zeros(B, K, D))
    assert d["mu"] == 0.0

    _, d10 = crit(img_mu * 10.0, torch.zeros(B, D), None, text_mu,
                  torch.zeros(B, D), torch.randn(B, K, D), torch.zeros(B, K, D))
    assert d10["mu"] > 1.0, d10["mu"]        # mean((10x − x)^2) = 81·mean(x^2)

    # detached target: text_mu never receives gradient; img_mu does
    # (im != tm so the mse is live and its gradient nonzero)
    tm = text_mu.clone().requires_grad_(True)
    im = (img_mu + 0.1 * torch.randn(B, D)).requires_grad_(True)
    loss, _ = crit(im, torch.zeros(B, D), None, tm, torch.zeros(B, D),
                   torch.randn(B, K, D), torch.zeros(B, K, D))
    loss.backward()
    assert tm.grad is None
    assert im.grad is not None and im.grad.norm() > 0

    # with lambda_mu=0 (everything else 0 too) no mu-term grad reaches img_mu
    crit0 = MCDispAlignLoss(match_score="cosine", lambda_match=0.0,
                            lambda_mu=0.0, lambda_var=0.0, lambda_reg=0.0,
                            lambda_dir=0.0)
    im2 = img_mu.clone().requires_grad_(True)
    loss0, _ = crit0(im2, torch.zeros(B, D), None, text_mu, torch.zeros(B, D),
                     torch.randn(B, K, D), torch.zeros(B, K, D))
    loss0.backward()
    assert im2.grad is None or im2.grad.norm() == 0


# ----------------------------------------------------------------- L_var (A02) / R_prior

def test_var_loss_full_marginal():
    """A02: L_var regresses the FULL image marginal variance d_v + Σ_r U²_r,
    not just the diagonal: matching the marginal exactly -> ~0; keeping the
    diagonal fixed while doubling U (same column space!) -> clearly positive,
    which the old diag-only L_var could not see."""
    torch.manual_seed(8)
    B, D, K, r = 3, 8, 4, 2
    d_v = 0.5 + torch.rand(B, D)                    # diagonal component
    img_logvar = torch.log(d_v)
    img_U = torch.randn(B, D, r)
    u_energy = (img_U ** 2).sum(-1)                 # (B, D) = Σ_r U²
    text_logvar = torch.log(d_v + u_energy)         # exact marginal target

    crit = MCDispAlignLoss(match_score="cosine", lambda_match=0.0,
                           lambda_mu=0.0, lambda_var=1.0, lambda_reg=0.0,
                           lambda_dir=0.0)
    _, d = crit(torch.randn(B, D), img_logvar, img_U, torch.zeros(B, D),
                text_logvar, torch.randn(B, K, D), torch.zeros(B, K, D))
    assert d["var"] < 1e-8, d["var"]                # exact match -> ~0 (eps gap only)

    # double U: same subspace (rank preserved), diagonal unchanged -> the old
    # diag-only formula would stay at ~0; the full marginal catches the growth
    _, d2 = crit(torch.randn(B, D), img_logvar, img_U * 2.0, torch.zeros(B, D),
                 text_logvar, torch.randn(B, K, D), torch.zeros(B, K, D))
    assert d2["var"] > 0.1, d2["var"]

    # img_U=None degenerates to the pure-diagonal formula
    _, d3 = crit(torch.randn(B, D), img_logvar, None, torch.zeros(B, D),
                 text_logvar, torch.randn(B, K, D), torch.zeros(B, K, D))
    expected = F.mse_loss(torch.log(d_v + 1e-6), text_logvar)
    assert abs(d3["var"] - expected.item()) < 1e-9

    # gradient reaches img_U through the marginal
    iU = img_U.clone().requires_grad_(True)
    loss, _ = crit(torch.randn(B, D), img_logvar, iU, torch.zeros(B, D),
                   text_logvar, torch.randn(B, K, D), torch.zeros(B, K, D))
    loss.backward()
    assert iU.grad is not None and iU.grad.norm() > 0


def test_prior_renamed_cal():
    """R_prior (ex-L_cal): the same formula under the new name; with every
    other group disabled it is the only gradient source of text_logvars, and
    lambda_reg=0 leaves text_logvars with no gradient at all."""
    torch.manual_seed(9)
    B, K, D = 3, 4, 8
    text_logvars = (-4 + torch.randn(B, K, D)).requires_grad_(True)
    crit = MCDispAlignLoss(match_score="cosine", lambda_match=0.0,
                           lambda_mu=0.0, lambda_var=0.0, lambda_reg=1.0,
                           lambda_dir=0.0, sigma0_sq=0.04)
    total, d = crit(torch.randn(B, D), torch.zeros(B, D), None,
                    torch.zeros(B, D), torch.zeros(B, D),
                    torch.randn(B, K, D), text_logvars)
    log_target = math.log(0.04)
    lv = text_logvars.detach()
    expected = ((torch.log(torch.exp(lv) + 1e-6) - log_target) ** 2).mean()
    assert abs(d["reg"] - expected.item()) < 1e-9
    total.backward()
    assert text_logvars.grad is not None and text_logvars.grad.norm() > 0

    # lambda_reg=0 and everything else 0 (cosine match touches no logvar):
    # no gradient at all reaches text_logvars
    crit0 = MCDispAlignLoss(match_score="cosine", lambda_match=0.0,
                            lambda_mu=0.0, lambda_var=0.0, lambda_reg=0.0,
                            lambda_dir=0.0)
    tlv0 = text_logvars.detach().clone().requires_grad_(True)
    loss0, _ = crit0(torch.randn(B, D), torch.zeros(B, D), None,
                     torch.zeros(B, D), torch.zeros(B, D),
                     torch.randn(B, K, D), tlv0)
    loss0.backward()
    assert tlv0.grad is None or tlv0.grad.norm() == 0


# ------------------------------------------------------------------- L_dir rank guard (A05)

def test_dir_rank_guard():
    """A05: L_dir is computed only on samples whose caption deviation spectrum
    actually has rank >= r_eff. Collapsed captions (every eigenvalue equal to
    the eps ridge, none exceeding the eps-floored threshold) are skipped --
    dir_valid = 0, loss 0.0 -- not charged the old constant 2r."""
    torch.manual_seed(10)
    B, K, D, r = 4, 4, 8, 3                         # r_eff = min(3, K-1=3, 8) = 3
    img_U = torch.randn(B, D, r)
    img_mu = torch.randn(B, D)
    img_logvar = torch.zeros(B, D)
    healthy = torch.randn(B, K, D) * 0.5
    crit = MCDispAlignLoss(match_score="cosine")    # dir is match-score agnostic

    # (a) all K captions identical -> rank 0 < r_eff on every sample
    collapsed = healthy[:, :1].repeat(1, K, 1)
    _, d_a = crit(img_mu, img_logvar, img_U, torch.zeros(B, D),
                  torch.zeros(B, D), collapsed, torch.zeros(B, K, D))
    assert d_a["dir"] == 0.0 and d_a["dir_valid"] == 0 and d_a["dir_total"] == B

    # (b) half the batch collapsed, half healthy -> healthy-only mean
    mixed = collapsed.clone()
    mixed[2:] = healthy[2:]
    _, d_b = crit(img_mu, img_logvar, img_U, torch.zeros(B, D),
                  torch.zeros(B, D), mixed, torch.zeros(B, K, D))
    assert d_b["dir_valid"] == 2 and d_b["dir_total"] == B
    # per-sample independence: the mixed-batch value equals the healthy-only mean
    _, d_h = crit(img_mu[2:], img_logvar[2:], img_U[2:], torch.zeros(2, D),
                  torch.zeros(2, D), healthy[2:], torch.zeros(2, K, D))
    assert d_h["dir_valid"] == 2
    assert abs(d_b["dir"] - d_h["dir"]) < 1e-6, (d_b["dir"], d_h["dir"])

    # (c) K=1 -> no between-caption variation at all
    _, d_c = crit(img_mu, img_logvar, img_U, torch.zeros(B, D),
                  torch.zeros(B, D), healthy[:, :1], torch.zeros(B, 1, D))
    assert d_c["dir"] == 0.0 and d_c["dir_valid"] == 0


# ---------------------------------------------------------------------- bookkeeping

def test_legacy_aliases_removed():
    """A04 cleanup: the temporary aliases are gone -- lambda_ctr / lambda_cal
    are no longer accepted kwargs, and no ctr*/cal/img_var_*/u_* keys remain
    in the loss dict (the trainer consumes the four-group keys only)."""
    torch.manual_seed(13)
    B, D, K = 2, 8, 3
    try:
        MCDispAlignLoss(lambda_ctr=0.7, lambda_cal=0.05)
    except TypeError:
        pass
    else:
        raise AssertionError("lambda_ctr/lambda_cal kwargs must be removed")

    crit = MCDispAlignLoss()
    _, d = crit(torch.randn(B, D), torch.zeros(B, D), None,
                torch.zeros(B, D), torch.zeros(B, D),
                torch.randn(B, K, D), torch.zeros(B, K, D))
    for legacy in ("ctr", "ctr_i2t", "ctr_t2i", "cal", "weighted_ctr",
                   "weighted_cal", "img_var_avg", "img_var_min",
                   "img_var_median", "img_var_mean", "img_var_max",
                   "u_energy", "diag_var_energy", "u_over_diag"):
        assert legacy not in d, legacy


def test_match_uses_diag_not_marginal_in_scorer():
    """Wiring pin: the scorer must receive the DIAGONAL component (it adds
    U itself); passing the full marginal would double-count U U^T."""
    torch.manual_seed(12)
    B, K, D, r = 2, 1, 4, 2
    img_mu = torch.randn(B, D)
    img_logvar = -2 + 0.3 * torch.randn(B, D)
    img_U = 0.5 * torch.randn(B, D, r)
    text_mus = img_mu.unsqueeze(1) + 0.2 * torch.randn(B, K, D)
    text_logvars = -2 + 0.3 * torch.randn(B, K, D)

    crit = _gauss_match_only()
    _, d = crit(img_mu, img_logvar, img_U,
                torch.zeros(B, D), torch.zeros(B, D),
                text_mus, text_logvars)

    # independent rebuild with the DIAGONAL variance fed to the scorer
    # (gaussian_overlap_scores adds the U U^T term itself)
    logits = gaussian_overlap_scores(
        img_mu, torch.exp(img_logvar), img_U,
        text_mus.reshape(B * K, D),
        torch.exp(text_logvars.reshape(B * K, D)),
    ) / crit.tau_match
    own = logits.reshape(B, B, K)
    idx = torch.arange(B)
    pos = own[idx, idx]                                       # (B, K) own captions
    exp_i2t = (torch.logsumexp(logits, dim=-1)
               - torch.logsumexp(pos, dim=-1)).mean()
    labels = torch.arange(B * K) // K
    exp_t2i = F.cross_entropy(logits.T, labels)
    assert abs(d["match_i2t"] - exp_i2t.item()) < 1e-6, (d["match_i2t"], exp_i2t.item())
    assert abs(d["match_t2i"] - exp_t2i.item()) < 1e-6, (d["match_t2i"], exp_t2i.item())


def test_total_equals_weighted_atomic_sum():
    """total is exactly the sum of the five weighted atomics, and disp is the
    dispersion group's weighted sum (checked at non-default weights)."""
    torch.manual_seed(11)
    B, K, D, r = 4, 5, 8, 3
    crit = MCDispAlignLoss(lambda_match=0.7, lambda_mu=0.3, lambda_var=1.2,
                           lambda_reg=0.05, lambda_dir=0.5)   # gaussian match
    _, d = crit(torch.randn(B, D), 0.1 * torch.randn(B, D), torch.randn(B, D, r),
                torch.randn(B, D), 0.1 * torch.randn(B, D),
                torch.randn(B, K, D), -3 + 0.1 * torch.randn(B, K, D))
    assert d["dir_valid"] > 0                         # guard actually passing
    atoms = (d["weighted_match"] + d["weighted_mu"] + d["weighted_var"]
             + d["weighted_reg"] + d["weighted_dir"])
    assert abs(d["total"] - atoms) < 1e-6, (d["total"], atoms)
    assert abs(d["disp"] - (d["weighted_var"] + d["weighted_reg"])) < 1e-6


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
