"""
Tests for utils/lr_scheduler.py (cosine + linear warmup schedule).

Each test_* function uses asserts; __main__ runs them all and reports.
"""

import math
import sys
from pathlib import Path

# Make project root importable (utils/ lives at project root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.lr_scheduler import cosine_warmup_factor, apply_lr_for_epoch


def _close(a, b, tol=1e-9):
    assert math.isclose(a, b, abs_tol=tol), f"{a} != {b} (within {tol})"


def test_warmup_linear_ramp():
    # warmup=3 over total=10: factor at epoch k == (k+1)/3
    _close(cosine_warmup_factor(0, 10, 3, 0.02), 1.0 / 3.0)
    _close(cosine_warmup_factor(1, 10, 3, 0.02), 2.0 / 3.0)
    _close(cosine_warmup_factor(2, 10, 3, 0.02), 1.0)  # last warmup epoch -> peak


def test_warmup_one_epoch_starts_at_full():
    # warmup=1: epoch 0 factor == 1.0 (single-epoch warmup reaches peak immediately)
    _close(cosine_warmup_factor(0, 10, 1, 0.02), 1.0)


def test_no_warmup_starts_at_peak():
    # warmup=0, min_ratio=0: pure cosine, epoch 0 == 1.0, midpoint == 0.5
    _close(cosine_warmup_factor(0, 10, 0, 0.0), 1.0)
    _close(cosine_warmup_factor(5, 10, 0, 0.0), 0.5)


def test_floor_never_below_min_ratio():
    # factor must be >= min_ratio for every epoch (cosine term in [0,1])
    min_ratio = 0.1
    factors = [cosine_warmup_factor(e, 10, 0, min_ratio) for e in range(10)]
    assert min(factors) >= min_ratio, f"factor {min(factors)} < min_ratio {min_ratio}"
    assert max(factors) <= 1.0 + 1e-12


def test_cosine_monotonic_after_warmup():
    # After warmup, factor is non-increasing across epochs.
    factors = [cosine_warmup_factor(e, 10, 2, 0.02) for e in range(2, 10)]
    for a, b in zip(factors, factors[1:]):
        assert b <= a + 1e-12, f"not non-increasing: {a} -> {b}"


def test_total_epochs_zero_returns_one():
    _close(cosine_warmup_factor(0, 0, 1, 0.02), 1.0)
    _close(cosine_warmup_factor(5, 0, 1, 0.02), 1.0)


def test_apply_lr_disabled_does_not_mutate():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    base_lrs = [0.1]
    before = [g["lr"] for g in opt.param_groups]
    factor = apply_lr_for_epoch(opt, base_lrs, 5, 10, 1, 0.02, scheduler="none")
    _close(factor, 1.0)
    assert [g["lr"] for g in opt.param_groups] == before, "disabled scheduler mutated lr"


def test_apply_lr_sets_per_group():
    # Two param groups with different base lrs; both scaled by the same factor.
    p = [torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))]
    opt = torch.optim.SGD([{"params": [p[0]], "lr": 0.5}, {"params": [p[1]], "lr": 0.05}])
    base_lrs = [0.5, 0.05]
    # warmup=0, min_ratio=0, total=10: epoch 5 -> factor 0.5
    factor = apply_lr_for_epoch(opt, base_lrs, 5, 10, 0, 0.0, scheduler="cosine")
    _close(factor, 0.5)
    _close(opt.param_groups[0]["lr"], 0.25)   # 0.5 * 0.5
    _close(opt.param_groups[1]["lr"], 0.025)  # 0.05 * 0.5


def main():
    tests = [
        v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]
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
