"""gaussian_overlap_scores 对照密集 C^{-1} 直接计算 + A16 数据梯度路径。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from losses.gaussian_overlap import gaussian_overlap_scores


def _dense_score(mu_v, d_v, U, mu_t, d_t):
    """独立基准：直接构 C 求 Cholesky（O(D^3)，仅小维度测试用）。"""
    B, D = mu_v.shape
    N = mu_t.shape[0]
    out = torch.empty(B, N, dtype=mu_v.dtype)
    for i in range(B):
        Sv = torch.diag(d_v[i])
        if U is not None:
            Sv = Sv + U[i] @ U[i].T
        for j in range(N):
            St = torch.diag(d_t[j])
            C = Sv + St
            d_ = mu_v[i] - mu_t[j]
            L = torch.linalg.cholesky(C)
            sol = torch.cholesky_solve(d_.unsqueeze(-1), L).squeeze(-1)
            mahal = (d_ * sol).sum()
            logdet = 2.0 * torch.log(torch.diag(L)).sum()
            out[i, j] = -0.5 * (mahal + logdet) / D
    return out


def test_matches_dense_reference():
    torch.manual_seed(0)
    B, N, D, r = 3, 7, 5, 2
    mu_v, d_v = torch.randn(B, D).double(), torch.rand(B, D).double() + 0.05
    U = torch.randn(B, D, r).double() * 0.3
    mu_t, d_t = torch.randn(N, D).double(), torch.rand(N, D).double() + 0.05
    got = gaussian_overlap_scores(mu_v, d_v, U, mu_t, d_t)
    ref = _dense_score(mu_v, d_v, U, mu_t, d_t)
    assert torch.allclose(got, ref, rtol=1e-8, atol=1e-8), (got - ref).abs().max()


def test_diagonal_only_matches_dense():
    torch.manual_seed(1)
    B, N, D = 2, 4, 6
    mu_v, d_v = torch.randn(B, D).double(), torch.rand(B, D).double() + 0.05
    mu_t, d_t = torch.randn(N, D).double(), torch.rand(N, D).double() + 0.05
    got = gaussian_overlap_scores(mu_v, d_v, None, mu_t, d_t)
    ref = _dense_score(mu_v, d_v, None, mu_t, d_t)
    assert torch.allclose(got, ref, rtol=1e-8, atol=1e-8)


def test_positive_overlaps_negative():
    """配对更近的分数应更高。"""
    torch.manual_seed(2)
    B, D = 2, 8
    mu_v = torch.zeros(B, D)
    mu_v[1] = 5.0                       # 图 1 远离 caption
    d_v = torch.full((B, D), 0.05)
    mu_t = torch.zeros(1, D)
    d_t = torch.full((1, D), 0.05)
    s = gaussian_overlap_scores(mu_v, d_v, None, mu_t, d_t)
    assert s[0, 0] > s[1, 0]


def test_gradient_reaches_all_distribution_params():
    """A16 验收核心：仅此分数即可让 caption 均值/方差、图像均值/方差/U
    全部拿到有限非零梯度（一般非退化批次）。"""
    torch.manual_seed(3)
    B, K, D, r = 3, 2, 6, 2
    mu_v = torch.randn(B, D, requires_grad=True)
    d_v = (torch.rand(B, D) * 0.1 + 0.05).log().detach().requires_grad_(True)
    U = (torch.randn(B, D, r) * 0.3).requires_grad_(True)
    cap_mu = torch.randn(B * K, D, requires_grad=True)
    cap_d = (torch.rand(B * K, D) * 0.1 + 0.05).log().detach().requires_grad_(True)

    s = gaussian_overlap_scores(mu_v, d_v.exp(), U, cap_mu, cap_d.exp())
    total = s.diagonal().sum() - s.mean()
    total.backward()
    for name, g in [("mu_v", mu_v), ("log_d_v", d_v), ("U", U),
                    ("cap_mu", cap_mu), ("cap_d", cap_d)]:
        assert g.grad is not None and torch.isfinite(g.grad).all() \
            and g.grad.norm() > 0, name


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
