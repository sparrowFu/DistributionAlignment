"""Tests for --sg-set-variance (stop-gradient on the L_set text-set variance).

Two guarantees:
1. FORWARD EQUIVALENCE: with identical inputs, the loss values with
   sg_set_variance=True and False are exactly equal (the switch changes only
   the gradient path, never a computed number).
2. GRADIENT CLOSURE: in the trained regime (aligned positives, hard
   negatives), the collapse gradients d L_set / d(caption norm scale) and
   d L_set / d(caption spread scale) are exactly 0 with the switch on, and
   strictly positive with it off (gradient descent actively shrinks caption
   norms/spread through the moment-matched v_C).
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from losses.mcdisp_align_losses_kl import MCDispAlignKLLoss


def _make_inputs(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    B, K, D = 32, 5, 128
    common = torch.randn(D, generator=g)
    centers = common + 0.55 * torch.randn(B, D, generator=g)
    tm = centers.unsqueeze(1) + 0.18 * torch.randn(B, K, D, generator=g)
    img_mu = centers + 0.10 * torch.randn(B, D, generator=g)
    img_lv = torch.full((B, D), -3.2)
    tlv = torch.full((B, K, D), -3.2)
    return img_mu, img_lv, tm, tlv


def _merged(tm, tlv):
    mu_c = tm.mean(1)
    var_c = (tlv.exp() + tm ** 2).mean(1) - mu_c ** 2
    return mu_c, var_c.clamp_min(1e-6).log()


def test_forward_values_identical():
    img_mu, img_lv, tm, tlv = _make_inputs()
    mu_c, lv_c = _merged(tm, tlv)
    out = []
    for sg in (False, True):
        crit = MCDispAlignKLLoss(sg_set_variance=sg)
        crit.uncertainty_grad_alpha = 1.0
        out.append(crit._set_nce(img_mu, img_lv, mu_c, lv_c).item())
    assert out[0] == out[1], f"forward values differ: {out}"


def test_collapse_gradient_closed_when_enabled():
    img_mu, img_lv, tm_base, tlv = _make_inputs()
    results = {}
    for sg in (False, True):
        crit = MCDispAlignKLLoss(sg_set_variance=sg)
        crit.uncertainty_grad_alpha = 1.0
        r = torch.tensor(1.0, requires_grad=True)   # caption norm scale
        s = torch.tensor(1.0, requires_grad=True)   # caption spread scale
        c = tm_base.mean(1, keepdim=True)
        tm = (c + s * (tm_base - c)) * r
        mu_c, lv_c = _merged(tm, tlv)
        crit._set_nce(img_mu, img_lv, mu_c, lv_c).backward()
        results[sg] = (r.grad.item(), s.grad.item())
    off_r, off_s = results[False]
    on_r, on_s = results[True]
    # sanity: the collapse channel is actually open in the paper path
    assert off_r > 0 and off_s > 0, f"expected positive collapse gradients, got {results[False]}"
    # the switch closes it (to floating-point zero; ~1e-8 vs ~4e-2 when open)
    assert abs(on_r) < 1e-6 and abs(on_s) < 1e-6, f"gradients not zeroed: {results[True]}"


def test_matching_gradient_through_means_is_preserved():
    """sg must NOT block the caption-mean gradient through the cosine numerator
    (the actual alignment signal): per-caption means still receive gradient."""
    img_mu, img_lv, tm, tlv = _make_inputs()
    crit = MCDispAlignKLLoss(sg_set_variance=True)
    crit.uncertainty_grad_alpha = 1.0
    tm_g = tm.clone().requires_grad_(True)
    mu_c, lv_c = _merged(tm_g, tlv)
    crit._set_nce(img_mu, img_lv, mu_c, lv_c).backward()
    assert tm_g.grad is not None and tm_g.grad.abs().sum() > 0
