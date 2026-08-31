"""
Tests for the distribution-to-set L_ctr (paper §3.3): caption-level
bidirectional InfoNCE. Pure CPU, follows the repo's main() runner pattern.

Reference behavior (design doc §4.1):
  L_ctr = (L_i2t + L_t2i) / 2
  L_i2t = (1/B) Σᵢ [ logsumexp_all_BK − logsumexp_own_K ]  (= −log P(top-1 ∈ own))
  L_t2i = per-caption cross entropy over the B images (label = own image)
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from losses.mcdisp_align_losses import MCDispAlignLoss

TAU = 0.07


def _ctr_only(**kwargs) -> MCDispAlignLoss:
    """Criterion with only L_ctr active (var/dir/cal zeroed)."""
    return MCDispAlignLoss(lambda_var=0.0, lambda_dir=0.0, lambda_cal=0.0, **kwargs)


def _forward_ctr(crit, img_mu, text_mus):
    """Call forward with dummy variance tensors; return the loss dict."""
    B, K, D = text_mus.shape
    _, d = crit(
        img_mu, torch.zeros(B, D), None,
        torch.zeros(B, D), torch.zeros(B, D),
        text_mus, torch.zeros(B, K, D),
    )
    return d


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

    d = _forward_ctr(_ctr_only(), img_mu, text_mus)

    e = math.exp(1.0 / TAU)
    exp_i2t = math.log(K * e + K * (B - 1)) - (math.log(K) + 1.0 / TAU)
    exp_t2i = math.log(e + (B - 1)) - 1.0 / TAU
    assert abs(d["ctr_i2t"] - exp_i2t) < 1e-4, (d["ctr_i2t"], exp_i2t)
    assert abs(d["ctr_t2i"] - exp_t2i) < 1e-4, (d["ctr_t2i"], exp_t2i)
    assert abs(d["ctr"] - 0.5 * (exp_i2t + exp_t2i)) < 1e-4


def test_per_caption_gradient_nonzero():
    """Root-cause fix: EVERY caption mean receives gradient from L_ctr.
    (The old mean-only InfoNCE sent none: text_mus was not an endpoint.)"""
    torch.manual_seed(0)
    B, K, D = 4, 5, 8
    img_mu = torch.randn(B, D, requires_grad=True)
    text_mus = torch.randn(B, K, D, requires_grad=True)
    crit = _ctr_only()
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


def test_ctr_no_gradient_to_variances():
    """Paper claim preserved: the contrastive score involves only means."""
    torch.manual_seed(1)
    B, K, D = 4, 5, 8
    img_lv = torch.randn(B, D, requires_grad=True)
    txt_lv = torch.randn(B, K, D, requires_grad=True)
    crit = _ctr_only()
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
    d = _forward_ctr(_ctr_only(), img_mu, cap)

    img_n = F.normalize(img_mu, dim=-1)
    cap_n = F.normalize(cap.squeeze(1), dim=-1)
    sim = (img_n @ cap_n.T) / TAU
    labels = torch.arange(B)
    expected = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))
    assert abs(d["ctr"] - expected.item()) < 1e-5


def test_lambda_ctr_zero_no_contrast_grad():
    torch.manual_seed(3)
    B, K, D = 4, 5, 8
    img_mu = torch.randn(B, D, requires_grad=True)
    crit = MCDispAlignLoss(lambda_ctr=0.0, lambda_var=0.0, lambda_dir=0.0, lambda_cal=0.0)
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
    d = _forward_ctr(_ctr_only(), img_mu, text_mus)
    assert d["ctr_i2t"] == 0.0 and d["ctr_t2i"] == 0.0 and d["ctr"] == 0.0


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
