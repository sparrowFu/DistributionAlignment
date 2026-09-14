"""Math tests for utils/mcdisp_coverage_score.py (plan §12.1).

Reference computations construct Sigma explicitly and use
torch.linalg.solve -- they deliberately do NOT reuse the Woodbury formulas
under test.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.mcdisp_coverage_score import (
    combine_coverage_score, coverage_score_parts, merge_topk, prepare_image_covariance,
    recalls_from_topk, sweep_all_lambdas, topk_stable, validate_features,
)

torch.manual_seed(0)


def _ref_q(delta, var, U, eps):
    """Independent reference: explicit Sigma + eps I, solved directly."""
    B, M, D = delta.shape
    Sigma = torch.diag_embed(var + eps)
    if U is not None:
        Sigma = Sigma + U @ U.transpose(-1, -2)
    Sigma = Sigma.unsqueeze(1).expand(B, M, D, D)
    return (torch.einsum("bmi,bmij,bmj->bm", delta, torch.linalg.inv(Sigma), delta)) / D


def _rand_case(B=4, D=9, r=3):
    img_mu = torch.randn(B, D, dtype=torch.float64)
    var = torch.rand(B, D, dtype=torch.float64) + 0.3
    U = torch.randn(B, D, r, dtype=torch.float64) * 0.5
    return img_mu, var, U


class TestMahalanobis:
    def test_woodbury_matches_direct_lowrank(self):
        img_mu, var, U = _rand_case()
        delta = torch.randn(4, 7, img_mu.shape[-1], dtype=torch.float64)
        state = prepare_image_covariance(torch.log(var), U, eps=1e-6)
        from utils.mcdisp_coverage_score import _quad_per_dim
        got = _quad_per_dim(delta, state["inv_var"].unsqueeze(1),
                            state["U_scaled"].unsqueeze(1), state["chol"].unsqueeze(1))
        ref = _ref_q(delta, var, U, 1e-6)
        assert torch.allclose(got, ref, rtol=1e-8, atol=1e-8)

    def test_diagonal_only_matches_perdim(self):
        img_mu, var, _ = _rand_case()
        delta = torch.randn(4, 5, img_mu.shape[-1], dtype=torch.float64)
        state = prepare_image_covariance(torch.log(var), None, eps=1e-6)
        from utils.mcdisp_coverage_score import _quad_per_dim
        got = _quad_per_dim(delta, state["inv_var"].unsqueeze(1), None, None)
        ref = (delta ** 2 / (var.unsqueeze(1) + 1e-6)).sum(-1) / delta.shape[-1]
        assert torch.allclose(got, ref, rtol=1e-10)

    def test_lambda_zero_returns_cosine_untouched(self):
        C = torch.randn(3, 4)
        P = torch.full((3, 4), float("inf"))          # hostile penalty
        out = combine_coverage_score(C, P, penalty_weight=0.0)
        assert out is C and torch.equal(out, C)

    def test_penalty_exactness(self):
        img_mu = torch.zeros(1, 4)
        var = torch.ones(1, 4)
        cap = torch.tensor([[0.5, 0, 0, 0], [3.0, 0, 0, 0]])
        state = prepare_image_covariance(torch.log(var), None)
        C, q, P = coverage_score_parts(img_mu, cap, state, coverage_margin=0.1)
        # q = d^2/(var+eps)/D; below margin -> P == 0; above -> exact difference
        expected = [0.0, 9.0 / (4 * (1 + 1e-6)) - 0.1]
        assert P[0, 0] == 0.0
        assert torch.allclose(P[0, 1], torch.tensor(expected[1]), rtol=1e-6)
        lam = 0.3
        S = combine_coverage_score(C, P, penalty_weight=lam)
        assert torch.allclose(S[0, 1], C[0, 1] - lam * expected[1], rtol=1e-6)

    def test_outofboundary_candidate_can_be_overtaken(self):
        # Tilted covariance (rank-1 U along (1,-1)): candidate A has HIGHER
        # cosine but lies across the tight direction (outside the ellipsoid);
        # candidate B has LOWER cosine but sits along the loose direction
        # (inside). A penalized score must rank B above A.
        img_mu = torch.tensor([[1.0, 0.0]])
        var = torch.tensor([[0.01, 0.01]])
        U = torch.tensor([[[1.0], [-1.0]]])
        A = torch.tensor([[1.3, 0.3]])    # delta (0.3, 0.3) orthogonal to U -> far outside
        B = torch.tensor([[1.5, -0.5]])   # delta (0.5, -0.5) along U -> inside
        state = prepare_image_covariance(torch.log(var), U)
        CA, qA, PA = coverage_score_parts(img_mu, A, state, coverage_margin=1.0)
        CB, qB, PB = coverage_score_parts(img_mu, B, state, coverage_margin=1.0)
        assert CA[0, 0] > CB[0, 0]        # cosine prefers A
        assert PA[0, 0] > 0 and PB[0, 0] == 0
        lam = 0.05
        SA = combine_coverage_score(CA, PA, penalty_weight=lam)[0, 0]
        SB = combine_coverage_score(CB, PB, penalty_weight=lam)[0, 0]
        assert SA < SB                    # penalty flips the ranking

    def test_distance_uses_raw_coordinates_not_normalized(self):
        # non-uniform scaling of the means must change q exactly as raw math says
        img_mu = torch.tensor([[1.0, 2.0]])
        var = torch.tensor([[0.5, 0.1]])
        cap = torch.tensor([[1.5, 2.4]])
        state = prepare_image_covariance(torch.log(var), None)
        _, q, _ = coverage_score_parts(img_mu, cap, state, coverage_margin=1e9)
        d = cap[0] - img_mu[0]
        ref = float((d ** 2 / (var[0] + 1e-6)).sum() / 2)
        assert torch.allclose(q[0, 0], torch.tensor(ref), rtol=1e-6)
        # sanity: had the means been L2-normalized first, the value would differ
        d_n = torch.nn.functional.normalize(cap, dim=-1)[0] - torch.nn.functional.normalize(img_mu, dim=-1)[0]
        ref_n = float((d_n ** 2 / (var[0] + 1e-6)).sum() / 2)
        assert not math.isclose(ref, ref_n, rel_tol=1e-3)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            validate_features(torch.zeros(2, 3), torch.full((2, 3), float("nan")), None)
        with pytest.raises(ValueError):
            validate_features(torch.zeros(2, 3), torch.zeros(2, 3),
                              torch.zeros(2, 3, 0))  # r=0 must be passed as None


class TestTopk:
    def test_tie_break_by_index(self):
        S = torch.tensor([[0.5, 0.5, 0.5, 0.2]])
        _, idx = topk_stable(S, 3)
        assert idx.tolist() == [[0, 1, 2]]

    def test_merge_equals_full_sort(self):
        torch.manual_seed(1)
        S = torch.randn(6, 200)
        full_v, full_i = topk_stable(S, 10)
        rv = torch.full((6, 10), -float("inf"))
        ri = torch.full((6, 10), -1, dtype=torch.long)
        for cs in range(0, 200, 37):
            v, i = topk_stable(S[:, cs:cs + 37], 10)
            rv, ri = merge_topk(rv, ri, v, i + cs, 10)
        assert torch.equal(ri, full_i)
        assert torch.allclose(rv, full_v)

    def test_ties_across_chunks_merge_by_global_index(self):
        a = torch.full((1, 2), 0.7)
        ai = torch.tensor([[150, 90]])
        rv = torch.full((1, 2), -float("inf"))
        ri = torch.full((1, 2), -1, dtype=torch.long)
        rv, ri = merge_topk(rv, ri, a, ai, 2)
        assert ri.tolist() == [[90, 150]]


class TestSweepAndRecall:
    def _feats(self, N=3, K=5, D=6):
        torch.manual_seed(2)
        img_mu = torch.randn(N, D)
        base = img_mu.repeat_interleave(K, dim=0)
        caption_mu = base + 0.05 * torch.randn(N * K, D)
        img_logvar = torch.full((N, D), -2.0)
        U = 0.1 * torch.randn(N, D, 2)
        caption_image_ids = torch.arange(N).repeat_interleave(K)
        return img_mu, img_logvar, U, caption_mu, caption_image_ids

    def test_chunked_sweep_equals_full_matrix(self):
        img_mu, img_logvar, U, cap_mu, cids = self._feats(N=7, K=5, D=6)
        runs, stats = sweep_all_lambdas(img_mu, img_logvar, U, cap_mu,
                                        penalty_grid=[0.0, 0.3], coverage_margin=0.5,
                                        image_chunk=2, caption_chunk=9, topk=10)
        state = prepare_image_covariance(img_logvar, U)
        C, q, P = coverage_score_parts(img_mu, cap_mu, state, coverage_margin=0.5)
        for lam in (0.0, 0.3):
            S = combine_coverage_score(C, P, penalty_weight=lam)
            _, full_i = topk_stable(S, 10)
            assert torch.equal(runs[lam].img_topk_idx, full_i)
            _, full_ti = topk_stable(S.T, 10)
            # N=7 < topk=10: the sweep keeps -1 padding beyond the candidate
            # count; compare the populated prefix only.
            n_pop = full_ti.shape[1]
            assert torch.equal(runs[lam].cap_topk_idx[:, :n_pop], full_ti)
            assert (runs[lam].cap_topk_idx[:, n_pop:] == -1).all()

    def test_recall_denominators_and_t2i_columns(self):
        # construct scores where image i's own captions all rank top for i,
        # and caption c's own image ranks top for c -> all recalls 1.0
        img_mu, img_logvar, U, cap_mu, cids = self._feats()
        runs, _ = sweep_all_lambdas(img_mu, img_logvar, U, cap_mu,
                                    penalty_grid=[0.0], coverage_margin=1e9)
        r = recalls_from_topk(runs[0.0].img_topk_idx, runs[0.0].cap_topk_idx, cids)
        assert r["mc_i2t@1"] == pytest.approx(1.0)
        assert r["mc_t2i@1"] == pytest.approx(1.0)
        assert r["mr"] == pytest.approx(1.0)
        # row vs column semantics + denominators: move 3 of image 2's five
        # captions next to image 0. Image 2 still retrieves its own remaining
        # captions first (I2T R@1 = 1.0 over N=3 queries), but those 3 moved
        # captions retrieve image 0 (T2I R@1 = 12/15 -- only reachable with
        # the M=N*K denominator, and distinct from the row metric).
        cap2 = cap_mu.clone()
        cap2[10:13] = img_mu[0] + 0.05 * torch.randn(3, cap_mu.shape[1])
        runs_m, _ = sweep_all_lambdas(img_mu, img_logvar, U, cap2,
                                      penalty_grid=[0.0], coverage_margin=1e9)
        r3 = recalls_from_topk(runs_m[0.0].img_topk_idx, runs_m[0.0].cap_topk_idx, cids)
        assert r3["mc_i2t@1"] == pytest.approx(1.0)
        assert r3["mc_t2i@1"] == pytest.approx(12 / 15)
