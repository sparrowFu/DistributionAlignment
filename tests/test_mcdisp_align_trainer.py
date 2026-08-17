"""
Tests for the 5-stage MCDisp_Align schedule (stage_multipliers, var_ramp, alpha_schedule). The trainer's end-to-end exports need a real model + data and are covered by the training scripts.

For total=10 the 5 stages (each 2 epochs) are:
  [0,2) warmup | [2,4) var_bootstrap | [4,6) pos_coverage |
  [6,8) neg_repulsion | [8,10) full (cov ramp)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.mcdisp_align_trainer import stage_multipliers, var_ramp, alpha_schedule


def _ones():
    return {"ctr": 1.0, "mu": 1.0, "reg": 1.0,
            "var": 1.0, "cover_pos": 1.0, "cover_neg": 1.0, "cov": 1.0,
            "stage": "full"}


def test_no_staged_is_all_ones():
    for epoch in range(5):
        m = stage_multipliers(epoch, 10, no_staged=True)
        # drop the stage tag for the equality check (it is "full")
        assert m == _ones(), f"no_staged epoch {epoch}: {m}"


def test_warmup_disables_var_cover_cov():
    for epoch in (0, 1):
        m = stage_multipliers(epoch, 10, no_staged=False)
        assert m["stage"] == "warmup"
        assert m["ctr"] == 1.0 and m["mu"] == 1.0 and m["reg"] == 1.0
        assert m["var"] == 0.0 and m["cover_pos"] == 0.0 and m["cover_neg"] == 0.0 and m["cov"] == 0.0


def test_var_bootstrap_only_var():
    for epoch in (2, 3):
        m = stage_multipliers(epoch, 10, no_staged=False)
        assert m["stage"] == "var_bootstrap", f"epoch {epoch}: {m}"
        assert m["var"] == 1.0
        assert m["cover_pos"] == 0.0 and m["cover_neg"] == 0.0 and m["cov"] == 0.0


def test_pos_coverage_adds_cover_pos():
    for epoch in (4, 5):
        m = stage_multipliers(epoch, 10, no_staged=False)
        assert m["stage"] == "pos_coverage", f"epoch {epoch}: {m}"
        assert m["var"] == 1.0 and m["cover_pos"] == 1.0
        assert m["cover_neg"] == 0.0 and m["cov"] == 0.0


def test_neg_repulsion_adds_cover_neg():
    for epoch in (6, 7):
        m = stage_multipliers(epoch, 10, no_staged=False)
        assert m["stage"] == "neg_repulsion", f"epoch {epoch}: {m}"
        assert m["var"] == 1.0 and m["cover_pos"] == 1.0 and m["cover_neg"] == 1.0
        assert m["cov"] == 0.0


def test_full_ramps_cov_to_one():
    # full from epoch 8; full_len = 10-8 = 2 -> epoch8 cov=0.5, epoch9 cov=1.0
    m8 = stage_multipliers(8, 10, no_staged=False)
    m9 = stage_multipliers(9, 10, no_staged=False)
    assert m8["stage"] == "full" and m8["cov"] == 0.5
    assert m9["cov"] == 1.0


def test_reg_always_on():
    for epoch in range(10):
        for no_staged in (True, False):
            assert stage_multipliers(epoch, 10, no_staged)["reg"] == 1.0


def test_var_ramp_per_step():
    spe = 50  # steps per epoch
    assert var_ramp(0, 0, spe, 10) == 0.0          # warmup: off
    assert var_ramp(1, spe - 1, spe, 10) == 0.0    # still warmup
    assert var_ramp(2, 0, spe, 10) == 0.05         # first var_bootstrap step
    assert var_ramp(4, 0, spe, 10) == 1.0          # after var_bootstrap: full
    # monotonic increase across var_bootstrap
    prev = var_ramp(2, 0, spe, 10)
    for step in range(1, spe):
        cur = var_ramp(2, step, spe, 10)
        assert cur >= prev - 1e-9
        prev = cur
    # last var_bootstrap step is close to (but the stage reaches 1.0 only at epoch 4)
    assert var_ramp(3, spe - 1, spe, 10) > 0.9


def test_alpha_schedule_per_step():
    spe = 50
    assert alpha_schedule(0, 0, spe, 10) == 0.0     # warmup: block L_set->sigma^2 grad
    assert alpha_schedule(2, 0, spe, 10) == 0.0     # start of var_bootstrap still 0
    assert alpha_schedule(4, 0, spe, 10) == 1.0     # after var_bootstrap: full grad
    assert 0.0 < alpha_schedule(3, spe - 1, spe, 10) < 1.0


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
