"""Tests for the rewritten MSDALoss (likelihood-based L_set).

Runnable without pytest:
    python tests/test_msdaloss_likelihood.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from losses.dist_align_losses import MSDALoss


def _inputs(B=4, D=8, K=5, r=4, requires_grad=True):
    g = lambda *s: torch.randn(*s, requires_grad=requires_grad)
    return dict(
        img_mu=g(B, D), img_logvar=g(B, D), img_U=g(B, D, r),
        text_mu_bar=g(B, D), text_logvar_bar=g(B, D),
        text_mus=g(B, K, D), text_logvars=g(B, K, D), text_Us=g(B, K, D, r),
    )


def test_loss_dict_keeps_only_surviving_keys():
    _, d = MSDALoss()(**_inputs())
    for k in ("total", "set_nce", "var", "cov", "img_var_avg"):
        assert k in d, f"missing {k}"
    for k in ("cover", "mu", "reg"):
        assert k not in d, f"removed key {k} still present"


def test_gradient_flows_to_heads():
    out = _inputs()
    total, _ = MSDALoss()(**out)
    total.backward()
    for name in ("img_mu", "img_logvar", "img_U", "text_mus"):
        grad = out[name].grad
        assert grad is not None and grad.abs().sum() > 0, f"no grad to {name}"


def test_diagonal_mode_is_finite():
    out = _inputs(r=4)
    out["img_U"] = None
    out["text_Us"] = None
    total, _ = MSDALoss()(**out)
    assert torch.isfinite(total)


def test_tau_is_learnable_parameter_by_default():
    loss = MSDALoss(learnable_tau=True)
    assert isinstance(loss.tau, torch.nn.Parameter)


def test_main_self_test_runs():
    """The module __main__ block must still execute (sanity)."""
    import subprocess
    res = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent.parent / "losses" / "dist_align_losses.py")],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr


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
