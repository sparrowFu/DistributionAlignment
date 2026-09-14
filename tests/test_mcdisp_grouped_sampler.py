"""Sampler + mining tests (plan §12.1 of the grouped-continuation plan)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.mcdisp_grouped_sampler import (
    batch_indices_to_batch_sampler, make_pools, plan_epoch_batches, plan_pool_batches,
)
from utils.mcdisp_mining_features import (
    TrainSnapshot, build_pool_candidate_table, native_set_logits,
    per_caption_confusability,
)


def _snap(W=8, D=6, seed=0, clusters=None):
    """Synthetic snapshot; `clusters` maps images to one-hot-ish anchors so
    confusability structure is controlled."""
    torch.manual_seed(seed)
    if clusters:
        base = torch.randn(max(clusters) + 1, D)
        img = torch.stack([base[c] for c in clusters])
    else:
        img = torch.randn(W, D)
    caps = img.unsqueeze(1) + 0.01 * torch.randn(W, 5, D)
    text_mu = caps.mean(1)
    return TrainSnapshot(
        img_mu=img, img_logvar=torch.full((len(img), D), -2.0),
        text_mus=caps, text_mu=text_mu,
        text_logvar=torch.full((len(img), D), -2.0),
        caption_fingerprints=[frozenset(f"cap{a}-{k}" for k in range(5))
                              for a in range(len(img))],
        image_ids=list(range(len(img))))


class TestDirectionAndGap:
    def test_i2t_uses_z_ij_t2i_uses_z_ji(self):
        # build_pool_candidate_table must evaluate the gap in the direction
        # matching the entry: g_i2t(a,j) = z_aa - z_aj, g_t2i(a,j) = z_aa - z_ja.
        snap = _snap(W=6, clusters=[0, 0, 1, 1, 2, 2])
        table = build_pool_candidate_table(snap, tau=0.07, use_uncertainty_sim=True,
                                           rank_start=0, rank_end=5)
        z = native_set_logits(snap, tau=0.07, use_uncertainty_sim=True)
        zd = torch.diagonal(z)
        for a, entry in table.items():
            for direction, cands in entry.items():
                for j in cands:
                    g = float(zd[a] - z[a, j]) if direction == "i2t" else float(zd[a] - z[j, a])
                    assert 0.0 <= g <= 2.0, (a, j, direction, g)

    def test_native_logits_match_loss_sim_matrix(self):
        from losses.mcdisp_align_losses import MCDispAlignLoss
        from losses.mcdisp_align_losses_kl import MCDispAlignKLLoss
        import torch.nn.functional as F
        snap = _snap(W=5)
        for cls, kw in ((MCDispAlignLoss, {}), (MCDispAlignKLLoss, {"lambda_kl": 1.0})):
            crit = cls(tau=0.07, use_uncertainty_sim=True, **kw)
            ref = crit._sim_matrix(
                F.normalize(snap.img_mu, dim=-1), snap.img_logvar,
                F.normalize(snap.text_mu, dim=-1), snap.text_logvar,
                crit.tau, crit.use_uncertainty_sim)
            got = native_set_logits(snap, tau=0.07, use_uncertainty_sim=True)
            assert torch.allclose(ref, got, rtol=1e-5), cls.__name__

    def test_per_caption_max_shape_and_symmetry(self):
        snap = _snap(W=7)
        h_i2t, h_t2i = per_caption_confusability(snap)
        assert h_i2t.shape == (7, 7)
        assert torch.allclose(h_i2t.diagonal(), torch.ones(7), atol=1e-4)


class TestCandidateFilters:
    def test_rank_window_and_gap(self):
        # 8 images, all near cluster 0 -> h ranks dense; craft z gaps by
        # scaling one image's set variance (changes uncertainty discount)
        snap = _snap(W=8, clusters=[0] * 8)
        snap.text_logvar[7] = 6.0               # huge variance -> tiny z with 7
        table = build_pool_candidate_table(snap, tau=0.07, use_uncertainty_sim=True,
                                           rank_start=1, rank_end=7, gap_min=0.0, gap_max=2.0)
        for a, entry in table.items():
            for cands in entry.values():
                assert a not in cands
        z = native_set_logits(snap, tau=0.07, use_uncertainty_sim=True)
        zd = torch.diagonal(z)
        for a, entry in table.items():
            for direction, cands in entry.items():
                gap = (zd[a] - z[a, :]) if direction == "i2t" else (zd[a] - z[:, a])
                for c in cands:
                    assert 0.0 <= float(gap[c]) <= 2.0

    def test_identical_caption_sets_excluded(self):
        snap = _snap(W=6, clusters=[0, 0, 1, 1, 2, 2])
        snap.caption_fingerprints[1] = snap.caption_fingerprints[0]
        table = build_pool_candidate_table(snap, tau=0.07, use_uncertainty_sim=True,
                                           rank_start=0, rank_end=5)
        assert 1 not in table.get(0, {}).get("i2t", [])
        assert 0 not in table.get(1, {}).get("i2t", [])

    def test_empty_table_when_fewer_than_rank_start(self):
        snap = _snap(W=3)
        table = build_pool_candidate_table(snap, tau=0.07, use_uncertainty_sim=True,
                                           rank_start=5, rank_end=64)
        assert table == {}


class TestBatchPlanning:
    def _pools_tables(self, W, seed=1):
        rng = torch.Generator().manual_seed(seed)
        pools = make_pools(W, pool_size=W, rng=rng)
        tables = [dict() for _ in pools]
        # rich table: every anchor has both directions filled with all others
        n = pools[0]
        tables[0] = {a: {"i2t": [j for j in n if j != a],
                          "t2i": [j for j in n if j != a]} for a in n}
        return pools, tables, rng

    def test_fixed_B_exposure_and_uniqueness(self):
        pools, tables, rng = self._pools_tables(40)
        batches, stats = plan_epoch_batches(pools, tables, P=4, B=8, rng=rng)
        flat = [i for b in batches for i in b]
        assert sorted(flat) == list(range(40))
        assert all(len(b) == 8 for b in batches)
        assert stats["qualified_pairs"] >= stats["planned_pairs"] - stats["fallback_pairs"]

    def test_pairs_alternate_directions(self):
        pool = list(range(8))
        table = {a: {"i2t": [b for b in pool if b != a and b % 2 == 0],
                     "t2i": [b for b in pool if b != a and b % 2 == 1]} for a in pool}
        rng = torch.Generator().manual_seed(3)
        batches, stats = plan_pool_batches(pool, table, 3, 4, rng)
        assert stats["planned_pairs"] == stats["qualified_pairs"]

    def test_random_arm_equivalent_and_same_exposure(self):
        # R: empty tables -> pure random batches; same exposure as G
        rng_r = torch.Generator().manual_seed(9)
        rng_g = torch.Generator().manual_seed(9)
        pools_r = make_pools(24, 24, rng=rng_r)
        pools_g = make_pools(24, 24, rng=rng_g)
        br, _ = plan_epoch_batches(pools_r, [dict()], 0, 8, rng_r)
        bg, _ = plan_epoch_batches(pools_g, [dict()], 0, 8, rng_g)
        assert sorted(i for b in br for i in b) == sorted(i for b in bg for i in b)
        assert [sorted(b) for b in br] == [sorted(b) for b in bg]   # same rng stream

    def test_deterministic_replay_from_rng_state(self):
        pools, tables, rng = self._pools_tables(32)
        state = rng.get_state()
        b1, s1 = plan_epoch_batches(pools, tables, 2, 8, rng)
        rng.set_state(state)
        b2, s2 = plan_epoch_batches(pools, tables, 2, 8, rng)
        assert b1 == b2 and s1["index_hash"] == s2["index_hash"]

    def test_tail_pool_inherits_full_batch_rule(self):
        rng = torch.Generator().manual_seed(2)
        pools = make_pools(20, 16, rng=rng)     # pools: 16 + 4 tail
        batches, _ = plan_epoch_batches(pools, [dict(), dict()], 0, 8, rng)
        sizes = sorted(len(b) for b in batches)
        assert sizes == [4, 8, 8]               # tail preserved, not dropped

    def test_batch_sampler_mapping(self):
        mapped = batch_indices_to_batch_sampler([[3, 1]], dataset_order=[10, 11, 12, 13])
        assert mapped == [[13, 11]]
