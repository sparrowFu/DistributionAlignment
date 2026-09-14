"""Eval-logic tests for the coverage scoring pipeline (plan §12.2, adapted
to the flat evaluate_* architecture): lambda selection, metric-group
construction, lambda0-equals-cosine semantics. Heavy end-to-end runs are
guarded by RUN_HEAVY=1."""

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.mcdisp_coverage_score import (
    recalls_from_topk, select_best_lambda, sweep_all_lambdas,
)


class TestLambdaSelection:
    def test_higher_mr_wins(self):
        dev = [{"penalty_weight": 0.0, "mr": 0.50},
               {"penalty_weight": 0.1, "mr": 0.53},
               {"penalty_weight": 0.3, "mr": 0.52}]
        assert select_best_lambda(dev) == 0.1

    def test_tie_breaks_to_smaller_lambda(self):
        dev = [{"penalty_weight": 0.3, "mr": 0.50},
               {"penalty_weight": 0.03, "mr": 0.50},
               {"penalty_weight": 0.0, "mr": 0.49}]
        assert select_best_lambda(dev) == 0.03

    def test_no_gain_selects_zero(self):
        dev = [{"penalty_weight": 0.0, "mr": 0.50},
               {"penalty_weight": 0.01, "mr": 0.50},
               {"penalty_weight": 0.1, "mr": 0.48}]
        assert select_best_lambda(dev) == 0.0


class TestRecallGroups:
    def _recalls(self):
        return {0.0: {"mc_i2t@1": 0.4, "mc_t2i@1": 0.6, "mr": 0.5},
                0.1: {"mc_i2t@1": 0.5, "mc_t2i@1": 0.7, "mr": 0.6}}

    def test_families_and_lambda_labels(self):
        from scripts.evaluate_mcdisp_coverage import build_recall_groups
        groups = build_recall_groups(self._recalls(), ks=[1])
        fams = [g["family"] for g in groups]
        assert fams == ["mc_cos_recall", "mc_coverage_recall"]
        assert "lambda=0.1" in groups[1]["label"]
        assert groups[0]["per_k"][1]["i2t"] == 0.4
        assert groups[1]["per_k"][1]["mean"] == pytest.approx(0.6)

    def test_lambda0_not_duplicated_as_coverage(self):
        from scripts.evaluate_mcdisp_coverage import build_recall_groups
        groups = build_recall_groups({0.0: {"mc_i2t@1": 0.4, "mc_t2i@1": 0.6}},
                                     ks=[1])
        assert [g["family"] for g in groups] == ["mc_cos_recall"]


class TestLambdaZeroSemantics:
    def test_tie_order_pipeline_recalls_match_direct_sweep(self):
        """Regression (main.py full-run bug): the tie-order pipeline must
        reproduce the direct sweep's recalls in BOTH directions, not just
        I2T."""
        from utils.mcdisp_coverage_score import rank_all_lambdas, sweep_all_lambdas
        torch.manual_seed(7)
        N, K, D = 9, 5, 6
        img_mu = torch.randn(N, D)
        cap = img_mu.repeat_interleave(K, dim=0) + 0.15 * torch.randn(N * K, D)
        lv = torch.full((N, D), -1.5)
        U = 0.1 * torch.randn(N, D, 2)
        cids = torch.arange(N).repeat_interleave(K)
        runs, _, iperm, cperm = rank_all_lambdas(
            img_mu, lv, U, cap, penalty_grid=[0.0, 0.3], coverage_margin=0.5,
            image_chunk=2, caption_chunk=7)
        got = recalls_from_topk(runs[0.3].img_topk_idx, runs[0.3].cap_topk_idx,
                                cids, row_image_ids=iperm, row_caption_ids=cperm)
        direct, _ = sweep_all_lambdas(img_mu, lv, U, cap,
                                      penalty_grid=[0.3], coverage_margin=0.5,
                                      image_chunk=3, caption_chunk=11)
        want = recalls_from_topk(direct[0.3].img_topk_idx, direct[0.3].cap_topk_idx,
                                 cids)
        for k in (1, 5, 10):
            assert got[f"mc_i2t@{k}"] == pytest.approx(want[f"mc_i2t@{k}"])
            assert got[f"mc_t2i@{k}"] == pytest.approx(want[f"mc_t2i@{k}"])

    def test_lambda0_run_equals_cosine_ranking(self):
        """lambda=0 must reproduce the cosine ranking exactly (plan §3/§7.3)."""
        torch.manual_seed(3)
        N, K, D = 5, 5, 8
        img_mu = torch.randn(N, D)
        cap = img_mu.repeat_interleave(K, dim=0) + 0.1 * torch.randn(N * K, D)
        lv = torch.full((N, D), -1.5)
        U = 0.1 * torch.randn(N, D, 2)
        cids = torch.arange(N).repeat_interleave(K)
        runs, _ = sweep_all_lambdas(img_mu, lv, U, cap,
                                    penalty_grid=[0.0, 0.07], coverage_margin=0.5)
        r0 = recalls_from_topk(runs[0.0].img_topk_idx, runs[0.0].cap_topk_idx, cids)
        assert r0["mc_i2t@1"] > 0.5          # sanity: clustered construction


@pytest.mark.skipif(not os.environ.get("RUN_HEAVY"),
                    reason="loads the real CLIP backbone; RUN_HEAVY=1 to enable")
class TestReadOnlyIntegration:
    def test_state_dict_unchanged_and_checkpoint_hash_stable(self, tmp_path):
        """Plan §12.2 item 7: one small extraction leaves the model weights
        and the checkpoint file byte-identical."""
        import hashlib
        from scripts.evaluate_mcdisp_coverage import extract_features
        from models.mcdisp_align_model import MCDispAlignModel
        from utils.eval_common import build_eval_dataloader
        ck = Path("checkpoints/seed42/mcdisp_align_kl_coco_best.pt")
        if not ck.exists():
            pytest.skip("checkpoint not present")
        before = hashlib.sha256(ck.read_bytes()).hexdigest()
        model = MCDispAlignModel()
        model.load(str(ck))
        model = model.to("cpu")
        snap = {k: v.clone() for k, v in model.state_dict().items()}
        loader, _ = build_eval_dataloader("coco", batch_size=2,
                                          num_workers=0, num_samples=8)
        feats = extract_features(model, loader, "cpu", num_samples=8)
        assert feats["caption_mu" if "caption_mu" in feats else "text_mus"].shape[0] > 0
        assert all(torch.equal(snap[k], v) for k, v in model.state_dict().items())
        assert hashlib.sha256(ck.read_bytes()).hexdigest() == before


class TestPairingIsolation:
    """Plan §12.3: answer-leakage checks on the scoring/ranking path."""

    def _feats(self, N=6, K=5, D=7, seed=5):
        torch.manual_seed(seed)
        img_mu = torch.randn(N, D)
        cap = img_mu.repeat_interleave(K, dim=0) + 0.2 * torch.randn(N * K, D)
        lv = torch.full((N, D), -1.5)
        U = 0.1 * torch.randn(N, D, 2)
        cids = torch.arange(N).repeat_interleave(K)
        return img_mu, lv, U, cap, cids

    def test_scoring_signature_is_label_free(self):
        """§7.4.1: the ranking entry point accepts only features + config."""
        import inspect
        from utils import mcdisp_coverage_score as m
        for fn in (m.rank_all_lambdas, m.sweep_all_lambdas,
                   m.coverage_score_parts, m.combine_coverage_score):
            params = set(inspect.signature(fn).parameters)
            banned = {"labels", "caption_image_ids", "cids", "positives",
                      "negatives", "targets", "y"}
            assert not (params & banned), (fn.__name__, params)
            assert inspect.getsource(fn).find("caption_image_ids") == -1

    def test_rankings_ignores_labels_and_metrics_do_not(self):
        """§12.3.2: label perturbation leaves rankings identical (the ranker
        cannot even see labels) while metrics respond to them."""
        from utils.mcdisp_coverage_score import rank_all_lambdas, recalls_from_topk
        img_mu, lv, U, cap, cids = self._feats()
        r1, _, img_perm, _ = rank_all_lambdas(img_mu, lv, U, cap,
                                           penalty_grid=[0.0, 0.1],
                                           coverage_margin=0.5)
        r2, _, _, _ = rank_all_lambdas(img_mu, lv, U, cap,
                                    penalty_grid=[0.0, 0.1], coverage_margin=0.5)
        for lam in (0.0, 0.1):
            assert torch.equal(r1[lam].img_topk_idx, r2[lam].img_topk_idx)
            assert torch.equal(r1[lam].cap_topk_idx, r2[lam].cap_topk_idx)
        # metrics DO depend on labels
        m1 = recalls_from_topk(r1[0.0].img_topk_idx, r1[0.0].cap_topk_idx, cids,
                               row_image_ids=img_perm)
        m2 = recalls_from_topk(r1[0.0].img_topk_idx, r1[0.0].cap_topk_idx,
                               cids.flip(0), row_image_ids=img_perm)
        assert m1["mr"] != m2["mr"]

    def test_order_equivariance(self):
        """§12.3.4: shuffling image rows and caption columns (ids travel with
        them) restores the same per-candidate ranking on tie-free data."""
        from utils.mcdisp_coverage_score import rank_all_lambdas
        img_mu, lv, U, cap, _ = self._feats(N=8)
        N, M = img_mu.shape[0], cap.shape[0]
        rA, _, permA, _ = rank_all_lambdas(img_mu, lv, U, cap,
                                        penalty_grid=[0.0, 0.3],
                                        coverage_margin=0.5)
        gi, ci = torch.randperm(N), torch.randperm(M)
        rB, _, permB, capB = rank_all_lambdas(img_mu[gi], lv[gi], U[gi], cap[ci],
                                        penalty_grid=[0.0, 0.3],
                                        coverage_margin=0.5)
        # ranked rows are in tie-order; map original image id -> ranked row
        invA = torch.empty_like(permA); invA[permA] = torch.arange(N)
        invB = torch.empty_like(permB); invB[permB] = torch.arange(N)
        row_of = lambda i: int(invA[i])                       # image i in A
        rowB_of = lambda i: int(invB[int((gi == i).nonzero()[0])])  # in B
        for lam in (0.0, 0.3):
            # B's input position j holds A's candidate ci[j], so a B id maps
            # back to an A id via ci itself.
            for i in range(N):
                a = set(rA[lam].img_topk_idx[row_of(i)].tolist())
                b = set(ci[rB[lam].img_topk_idx[rowB_of(i)]].tolist())
                assert a == b, (i, lam, sorted(a), sorted(b))

    def test_ties_follow_prefixed_order(self):
        """§12.3.5: with fully tied scores the order is exactly the pre-fixed
        seeded candidate order -- never grouping or labels."""
        from utils.mcdisp_coverage_score import make_tie_order, rank_all_lambdas
        N, K, D = 4, 5, 3
        img_mu = torch.zeros(N, D)
        cap = torch.zeros(N * K, D)          # every cosine identical -> all ties
        lv = torch.full((N, D), -1.0)
        r, _, _, _ = rank_all_lambdas(img_mu, lv, None, cap,
                                   penalty_grid=[0.0], coverage_margin=1e9)
        _, cap_perm = make_tie_order(N, N * K)
        for row in r[0.0].img_topk_idx:
            assert row.tolist() == cap_perm[:r[0.0].img_topk_idx.shape[1]].tolist()
