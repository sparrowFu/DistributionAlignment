"""
Tests for ProbVLM (B4) pure components: the GenGauss/TempComb losses, the
generalized-Gaussian -> Gaussian log-variance conversion, the BayesCap MLP
adapter, and the total-loss composition. Pure-function tests only (no CLIP
weights, no GPU, no dataset) per the tests/ convention; the end-to-end
training/evaluation is covered by the scripts.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from losses.probvlm_losses import (
    GenGaussLoss,
    TempCombLoss,
    generalized_gaussian_logvar,
    probvlm_total_loss,
)
from models.probvlm_model import BayesCapMLP


def _gg_params(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(shape, generator=g)
    one_over_alpha = torch.rand(shape, generator=g)
    beta = torch.rand(shape, generator=g) + 0.5
    target = torch.randn(shape, generator=g)
    return mu, one_over_alpha, beta, target


class TestGenGaussLoss:
    def test_matches_manual_formula(self):
        mu, one_over_alpha, beta, target = _gg_params((4, 16))
        loss = GenGaussLoss()(mu, one_over_alpha, beta, target)

        one_over_alpha1 = one_over_alpha + 1e-4
        beta1 = beta + 1e-4
        resi = (torch.abs(mu - target) * one_over_alpha1 * beta1).clamp(1e-4, 1e3)
        manual = (
            resi - torch.log(one_over_alpha1)
            + torch.lgamma(torch.pow(beta1, -1)) - torch.log(beta1)
        ).mean()

        assert torch.allclose(loss, manual, atol=1e-6)

    def test_sum_reduction(self):
        mu, one_over_alpha, beta, target = _gg_params((4, 16), seed=1)
        mean_loss = GenGaussLoss()(mu, one_over_alpha, beta, target)
        sum_loss = GenGaussLoss(reduction="sum")(mu, one_over_alpha, beta, target)
        assert torch.allclose(mean_loss * mu.numel(), sum_loss, atol=1e-4)

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError):
            GenGaussLoss(reduction="max")

    def test_residual_clamp_saturates(self):
        # |mu - target| huge -> resi hits resi_max, loss saturates
        mu = torch.zeros(2, 8)
        target = torch.full((2, 8), 1e6)
        one_over_alpha = torch.ones(2, 8)
        beta = torch.ones(2, 8)
        loss = GenGaussLoss()(mu, one_over_alpha, beta, target)
        expected = (
            1e3 - math.log(1 + 1e-4)
            + math.lgamma(1.0 / (1 + 1e-4)) - math.log(1 + 1e-4)
        )
        assert torch.allclose(loss, torch.tensor(expected), atol=1e-4)


class TestTempCombLoss:
    def test_linear_composition(self):
        for seed in (0, 1, 2):
            mu, one_over_alpha, beta, target = _gg_params((4, 16), seed=seed)
            t1, t2 = 0.7, 0.03
            got = TempCombLoss()(mu, one_over_alpha, beta, target, t1=t1, t2=t2)
            expected = (
                t1 * nn.L1Loss()(mu, target)
                + t2 * GenGaussLoss()(mu, one_over_alpha, beta, target)
            )
            assert torch.allclose(got, expected, atol=1e-6)


class TestGeneralizedGaussianLogvar:
    def test_beta2_gaussian_case(self):
        # beta=2: sigma^2 = alpha^2 * Gamma(1.5)/Gamma(0.5) = alpha^2 / 2
        # (eps=0.0 for the pure closed form; the default eps shifts the value
        # by O(1e-4), covered by the finiteness tests below)
        inv_alpha = torch.full((2, 3), 4.0)   # alpha = 1/4
        beta = torch.full((2, 3), 2.0)
        got = generalized_gaussian_logvar(inv_alpha, beta, eps=0.0)
        expected = 2 * math.log(0.25) - math.log(2.0)
        assert torch.allclose(got, torch.full_like(got, expected), atol=1e-5)

    def test_beta1_laplacian_case(self):
        # beta=1: sigma^2 = alpha^2 * Gamma(3)/Gamma(1) = 2 alpha^2
        inv_alpha = torch.full((2, 3), 4.0)   # alpha = 1/4
        beta = torch.full((2, 3), 1.0)
        got = generalized_gaussian_logvar(inv_alpha, beta, eps=0.0)
        expected = 2 * math.log(0.25) + math.log(2.0)
        assert torch.allclose(got, torch.full_like(got, expected), atol=1e-5)

    def test_shape_preserved_and_finite(self):
        mu, one_over_alpha, beta, _ = _gg_params((4, 16), seed=3)
        logvar = generalized_gaussian_logvar(torch.relu(one_over_alpha), torch.relu(beta))
        assert logvar.shape == one_over_alpha.shape
        assert torch.isfinite(logvar).all()

    def test_tiny_beta_clamped(self):
        # beta -> 0 would make lgamma(3/beta) explode; the clamp keeps it finite
        inv_alpha = torch.full((2, 3), 4.0)
        beta = torch.full((2, 3), 1e-12)
        logvar = generalized_gaussian_logvar(inv_alpha, beta)
        assert torch.isfinite(logvar).all()
        assert (logvar <= 20.0).all()


class TestBayesCapMLP:
    def test_trunk_structure(self):
        mlp = BayesCapMLP(8, 8, hid_dim=4, num_layers=3, p_drop=0.1)
        trunk_types = [type(mod) for mod in mlp.mod]
        assert trunk_types == [
            nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Dropout, nn.Linear,
        ]
        assert mlp.mod[0].in_features == 8 and mlp.mod[0].out_features == 4
        assert mlp.mod[2].in_features == 4 and mlp.mod[2].out_features == 4
        assert mlp.mod[5].in_features == 4 and mlp.mod[5].out_features == 8

    def test_output_shapes_and_signs(self):
        mlp = BayesCapMLP(8, 8, hid_dim=4, num_layers=3, p_drop=0.1)
        mlp.eval()
        x = torch.randn(5, 8)
        with torch.no_grad():
            mu, one_over_alpha, beta = mlp(x)
        assert mu.shape == (5, 8) and one_over_alpha.shape == (5, 8) and beta.shape == (5, 8)
        assert (one_over_alpha >= 0).all() and (beta >= 0).all()

    def test_block_activations(self):
        # mu is unconstrained (ends with Linear); alpha/beta end with ReLU
        mlp = BayesCapMLP(8, 8, hid_dim=4, num_layers=3, p_drop=0.1)
        assert isinstance(mlp.block_mu[-1], nn.Linear)
        assert isinstance(mlp.block_alpha[-1], nn.ReLU)
        assert isinstance(mlp.block_beta[-1], nn.ReLU)

    def test_invalid_num_layers_raises(self):
        with pytest.raises(ValueError):
            BayesCapMLP(8, 8, hid_dim=4, num_layers=2)

    def test_mismatched_dims_raise(self):
        with pytest.raises(ValueError):
            BayesCapMLP(8, 6, hid_dim=4, num_layers=3)


class TestProbVLMTotalLoss:
    def _outputs(self, seed=0):
        mu, one_over_alpha, beta, target = _gg_params((4, 16), seed=seed)
        t_mu, t_1alpha, t_beta, t_target = _gg_params((4, 16), seed=seed + 100)
        return {
            "img_mu": mu, "img_1alpha": one_over_alpha, "img_beta": beta,
            "text_mu": t_mu, "txt_1alpha": t_1alpha, "txt_beta": t_beta,
            "i_features": target, "t_features": t_target,
        }

    def test_zero_lambda_drops_cross_modal(self):
        outputs = self._outputs(seed=0)
        criterion = TempCombLoss()
        loss, info = probvlm_total_loss(outputs, criterion, t1=1.0, t2=5e-2,
                                        cross_modal_lambda=0.0)
        loss_i = criterion(outputs["img_mu"], outputs["img_1alpha"],
                           outputs["img_beta"], outputs["i_features"], t1=1.0, t2=5e-2)
        loss_t = criterion(outputs["text_mu"], outputs["txt_1alpha"],
                           outputs["txt_beta"], outputs["t_features"], t1=1.0, t2=5e-2)
        assert torch.allclose(loss, loss_i + loss_t, atol=1e-6)
        assert set(info) == {"loss", "loss_i", "loss_t", "loss_i4t", "loss_t4i"}

    def test_lambda_composition(self):
        outputs = self._outputs(seed=1)
        criterion = TempCombLoss()
        lam = 1e-4
        loss, info = probvlm_total_loss(outputs, criterion, t1=1.0, t2=5e-2,
                                        cross_modal_lambda=lam)
        loss_i4t = criterion(outputs["img_mu"], outputs["img_1alpha"],
                             outputs["img_beta"], outputs["t_features"], t1=1.0, t2=5e-2)
        loss_t4i = criterion(outputs["text_mu"], outputs["txt_1alpha"],
                             outputs["txt_beta"], outputs["i_features"], t1=1.0, t2=5e-2)
        expected = (info["loss_i"] + info["loss_t"]
                    + lam * (info["loss_i4t"] + info["loss_t4i"]))
        assert loss.item() == pytest.approx(expected, rel=1e-6)
        assert loss_i4t.item() == pytest.approx(info["loss_i4t"], rel=1e-6)
        assert loss_t4i.item() == pytest.approx(info["loss_t4i"], rel=1e-6)
