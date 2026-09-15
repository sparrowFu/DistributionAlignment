"""Tests for the KL + auxiliary per-caption NCE loss (kl_capnce)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from losses.mcdisp_align_losses_kl import MCDispAlignKLLoss
from losses.mcdisp_align_losses_kl_capnce import MCDispAlignKLCapNCELoss


def _batch(B=4, K=5, D=16, seed=0, aligned=True):
    torch.manual_seed(seed)
    img_mu = torch.randn(B, D)
    base = img_mu.repeat_interleave(K, dim=0) if aligned else torch.randn(B * K, D)
    text_mus = (base + 0.05 * torch.randn(B * K, D)).view(B, K, D)
    img_lv = torch.full((B, D), -2.0)
    txt_lv = torch.full((B, D), -2.0)
    cap_lv = torch.full((B, K, D), -2.0)
    merged_mu = text_mus.mean(1)
    merged_lv = torch.log(text_mus.var(dim=1, unbiased=False) + 1e-6)
    return dict(img_mu=img_mu, img_logvar=img_lv, img_U=None,
                text_mu=merged_mu, text_logvar=merged_lv,
                text_mus=text_mus, text_logvars=cap_lv, text_Us=None)


def _kwargs():
    return dict(lambda_ctr=1.0, lambda_kl=1.0, tau=0.07)


class TestCapNCE:
    def test_lambda0_exactly_equals_parent(self):
        for seed in range(3):
            b = _batch(seed=seed)
            parent = MCDispAlignKLLoss(**_kwargs())
            child = MCDispAlignKLCapNCELoss(lambda_cap_nce=0.0, **_kwargs())
            lp, dp = parent(**b)
            lc, dc = child(**b)
            assert torch.allclose(lp, lc, rtol=1e-6)
            def _val(x):
                return float(x.item()) if torch.is_tensor(x) else float(x)
            for k, v in dp.items():
                assert _val(v) == pytest.approx(_val(dc[k]), rel=1e-6), k

    def test_aligned_lower_than_shuffled(self):
        good = MCDispAlignKLCapNCELoss(lambda_cap_nce=1.0, **_kwargs())
        b1 = _batch(aligned=True)
        b2 = _batch(aligned=False)
        _, d1 = good(**b1)
        _, d2 = good(**b2)
        assert d1["cap_nce"] < d2["cap_nce"]

    def test_weighted_total_accounting(self):
        child = MCDispAlignKLCapNCELoss(lambda_cap_nce=0.5, **_kwargs())
        parent = MCDispAlignKLLoss(**_kwargs())
        b = _batch()
        lp, _ = parent(**b)
        lc, dc = child(**b)
        assert torch.allclose(lc, lp + 0.5 * dc["cap_nce"], rtol=1e-5)

    def test_dict_has_new_keys_and_parent_keys(self):
        child = MCDispAlignKLCapNCELoss(**_kwargs())
        _, d = child(**_batch())
        for k in ("total", "set_nce", "mu", "var", "cap_nce", "weighted_cap_nce"):
            assert k in d

    def test_trainer_branch_constructs(self):
        from utils.mcdisp_align_trainer import MCDispAlignTrainConfig
        cfg = MCDispAlignTrainConfig(loss_name="kl_capnce")
        assert cfg.lambda_cap_nce == 0.5 and cfg.tau_cap == 0.07
        assert cfg.ckpt_model_name.endswith("_kl_capnce")
