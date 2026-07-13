"""
Tests for MSDALoss._variance_target — the L_var target computation.

The fix: in "rescaled" mode the caption_spread target is rescaled so its batch
mean matches img_var's batch mean, which removes the large mean offset that
otherwise drowns the per-image signal and collapses the variance head.

Runnable without pytest:
    python tests/test_msdaloss_variance.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from losses.dist_align_losses import MSDALoss


def _close(a, b, tol=1e-6):
    assert torch.allclose(a, b, atol=tol), f"\n{a}\n!=\n{b}\n(within {tol})"


def test_raw_returns_caption_spread_unchanged():
    loss = MSDALoss()
    img_var = torch.rand(4, 8) * 0.1 + 0.1        # ~0.1..0.2
    caption_spread = torch.rand(4, 8) * 0.005      # ~0..0.005
    target = loss._variance_target(img_var, caption_spread, "raw")
    _close(target, caption_spread)                 # identity
    assert not target.requires_grad, "raw target must be detached"


def test_rescaled_matches_img_var_mean():
    """rescaled target mean == img_var mean (the 36x offset is killed).

    A tiny residual (~eps/mean) is expected from the numerical-stabilizer in the
    denominator; assert relative match well below the original 36x gap.
    """
    loss = MSDALoss()
    img_var = torch.rand(4, 8) * 0.05 + 0.10       # mean ~0.125
    caption_spread = torch.rand(4, 8) * 0.004      # mean ~0.002 (~60x smaller)
    target = loss._variance_target(img_var, caption_spread, "rescaled")
    rel = abs(target.mean().item() - img_var.mean().item()) / img_var.mean().item()
    assert rel < 1e-2, f"mean not matched: rel={rel:.4f}"


def test_rescaled_preserves_per_image_variation():
    """rescaling is a constant factor, so per-image relative pattern is preserved."""
    loss = MSDALoss()
    img_var = torch.rand(4, 8) * 0.05 + 0.10
    caption_spread = torch.rand(4, 8) * 0.004 + 1e-4
    target = loss._variance_target(img_var, caption_spread, "rescaled")
    # target / mean(target)  ==  caption_spread / mean(caption_spread)
    rel_t = target / (target.mean() + 1e-12)
    rel_c = caption_spread / (caption_spread.mean() + 1e-12)
    _close(rel_t, rel_c, tol=1e-5)
    # And the CV (signal we want the head to learn) is preserved:
    cv_c = caption_spread.std() / caption_spread.mean()
    cv_t = target.std() / target.mean()
    assert abs(cv_c - cv_t) < 1e-4, f"CV changed: {cv_c} -> {cv_t}"


def test_rescaled_detached():
    loss = MSDALoss()
    img_var = torch.rand(4, 8, requires_grad=True)
    caption_spread = torch.rand(4, 8)
    target = loss._variance_target(img_var, caption_spread, "rescaled")
    assert not target.requires_grad, "rescaled target must be detached (it's a target)"


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
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
