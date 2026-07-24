"""
Tests for utils/dist_align_trainer.stage_multipliers — the staged MSDA loss
schedule. (The shared trainer's other exports — create_optimizer / train_epoch /
evaluate / run_dist_align_training — need a real model + data, so they are
covered by the end-to-end training scripts rather than unit tests.)

Runnable without pytest:
    python tests/test_dist_align_trainer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dist_align_trainer import stage_multipliers


def _ones():
    return {"ctr": 1.0, "mu": 1.0, "var": 1.0, "cover": 1.0, "cov": 1.0, "reg": 1.0}


def test_no_staged_is_all_ones():
    for epoch in range(5):
        m = stage_multipliers(epoch, 10, no_staged=True)
        assert m == _ones(), f"no_staged epoch {epoch}: {m}"


def test_warmup_disables_var_cover_cov():
    # total=10 -> warmup_end=2, so epochs 0,1 are warmup
    m0 = stage_multipliers(0, 10, no_staged=False)
    m1 = stage_multipliers(1, 10, no_staged=False)
    for m in (m0, m1):
        assert m["ctr"] == 1.0 and m["mu"] == 1.0 and m["reg"] == 1.0
        assert m["var"] == 0.0 and m["cover"] == 0.0 and m["cov"] == 0.0


def test_main_disables_only_cov():
    # total=10 -> main_end=8, so epochs 2..7 are main: var+cover on, cov off
    for epoch in range(2, 8):
        m = stage_multipliers(epoch, 10, no_staged=False)
        assert m["var"] == 1.0 and m["cover"] == 1.0, f"main epoch {epoch}: {m}"
        assert m["cov"] == 0.0, f"main epoch {epoch} cov should be 0: {m}"


def test_full_ramps_cov_to_one():
    # total=10 -> full from epoch 8; full_len = 10-8 = 2
    # epoch 8 -> cov = (8-8+1)/2 = 0.5 ; epoch 9 -> cov = (9-8+1)/2 = 1.0
    assert stage_multipliers(8, 10, no_staged=False)["cov"] == 0.5
    assert stage_multipliers(9, 10, no_staged=False)["cov"] == 1.0


def test_reg_always_on():
    for epoch in range(10):
        for no_staged in (True, False):
            assert stage_multipliers(epoch, 10, no_staged)["reg"] == 1.0


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
