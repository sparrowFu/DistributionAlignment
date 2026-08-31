"""A07: centered moment matching. Old form mean(var+mu^2)-mean(mu)^2 loses
effective digits when |mu| >> spread; centered form mean(var)+mean(dev^2)
is exact to fp32. K=1 must equal the caption's own variance."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from models.mcdisp_align_model import MCDispAlignModel


def _merge(mus, logvars):
    m = MCDispAlignModel.__new__(MCDispAlignModel)   # 不加载 CLIP，只借用方法
    return m._moment_matching(mus, logvars, None)


def test_large_center_small_spread():
    """float32 下旧式约 1e-6，中心化应得 ≈ 2/3 + 0.04 = 0.7067。"""
    mus = torch.tensor([[[9999.0], [10000.0], [10001.0]]])        # (1,3,1)
    logvars = torch.full((1, 3, 1), 0.04).log()
    _, combined_logvar = _merge(mus, logvars)
    v = torch.exp(combined_logvar)
    assert abs(v.item() - (2.0 / 3.0 + 0.04)) < 1e-3, v.item()


def test_k1_equals_own_variance():
    mus = torch.randn(4, 1, 8).double()
    logvars = torch.randn(4, 1, 8).double()
    mu_out, lv_out = _merge(mus, logvars)
    assert torch.allclose(mu_out, mus[:, 0, :], atol=1e-9)
    assert torch.allclose(torch.exp(lv_out), torch.exp(logvars[:, 0, :]), rtol=1e-6)


def test_translation_invariance():
    torch.manual_seed(0)
    mus = torch.randn(3, 5, 6)
    logvars = torch.randn(3, 5, 6)
    _, lv1 = _merge(mus, logvars)
    _, lv2 = _merge(mus + 100.0, logvars)
    assert torch.allclose(lv1, lv2, atol=1e-4)


def test_matches_explicit_formula():
    torch.manual_seed(1)
    mus, logvars = torch.randn(2, 5, 7).double(), torch.randn(2, 5, 7).double()
    mu_out, lv_out = _merge(mus, logvars)
    dev = mus - mus.mean(1, keepdim=True)
    expect_var = torch.exp(logvars).mean(1) + (dev ** 2).mean(1)
    assert torch.allclose(mu_out, mus.mean(1), atol=1e-9)
    assert torch.allclose(torch.exp(lv_out), expect_var, rtol=1e-8)


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa
            failed += 1; print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
