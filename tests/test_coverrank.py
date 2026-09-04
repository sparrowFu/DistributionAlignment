"""
Tests for the cover-rank multi-caption retrieval metric
(utils/retrieval.py: compute_multicaption_coverrank).

  1. Exactness on a handcrafted score matrix: per-image cover ranks are known,
     so mean / median / censored mean / covered fraction are hand-computable.
  2. Consistency with all-hit: covered@K must equal allhit@K from
     compute_multicaption_allhit on the SAME matrix (R_i == K <=> top-K is
     exactly the own set).
  3. Chunking invariance (chunk_rows=1 == one big chunk).
  4. Censoring: an image whose last own caption sits beyond k_max is excluded
     from mean/median and counted as k_max in the censored mean.
  5. Functional: orthogonal anchors + small noise -> every R_i == K.

Pure CPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.retrieval import compute_multicaption_allhit, compute_multicaption_coverrank


def _matrix_scorer(S: torch.Tensor):
    return lambda sl: S[sl]


def _build_matrix():
    """N=3, K=2 (gallery 0..5, image-major). Rankings (descending):
      img0: c00 c01 ...        -> own ranks {1,2}  -> R=2
      img1: c10 c00 c11 ...    -> own ranks {1,3}  -> R=3
      img2: c00 c01 c20 c21 .. -> own ranks {3,4}  -> R=4
    """
    order = {0: [0, 1, 2, 3, 4, 5],
             1: [2, 0, 3, 1, 4, 5],
             2: [0, 1, 4, 5, 2, 3]}
    S = torch.zeros(3, 6)
    for i, ranking in order.items():
        for rank, col in enumerate(ranking):
            S[i, col] = 6.0 - rank
    return S


def test_exact_values():
    S = _build_matrix()
    m = compute_multicaption_coverrank(
        {"m": _matrix_scorer(S)}, n_images=3, k_per_image=2, k_max=6)
    assert abs(m["m_coverrank_mean@6"] - (2 + 3 + 4) / 3) < 1e-12
    assert abs(m["m_coverrank_median@6"] - 3.0) < 1e-12
    assert m["m_coverrank_censored_mean@6"] == m["m_coverrank_mean@6"]  # all covered
    assert abs(m["m_covered@6"] - 1.0) < 1e-12


def test_consistency_with_allhit():
    S = _build_matrix()
    cov = compute_multicaption_coverrank(
        {"m": _matrix_scorer(S)}, n_images=3, k_per_image=2, k_max=2)
    ah = compute_multicaption_allhit(
        {"m": _matrix_scorer(S)}, n_images=3, k_per_image=2, k_hit=2)
    # covered@K == allhit@K: only img0 (R=2) has its whole set inside top-2
    assert abs(cov["m_covered@2"] - ah["m_allhit@2"]) < 1e-12
    assert abs(cov["m_covered@2"] - 1 / 3) < 1e-12


def test_chunking_invariance():
    g = torch.Generator().manual_seed(0)
    N, K, D = 10, 3, 8
    S = torch.randn(N, D, generator=g) @ torch.randn(N * K, D, generator=g).T
    full = compute_multicaption_coverrank(
        {"m": _matrix_scorer(S)}, N, K, k_max=15, chunk_rows=512)
    chunked = compute_multicaption_coverrank(
        {"m": _matrix_scorer(S)}, N, K, k_max=15, chunk_rows=1)
    assert full == chunked


def test_censoring():
    S = _build_matrix()
    # k_max=3: img0 (R=2) and img1 (R=3) covered; img2's last own at rank 4
    m = compute_multicaption_coverrank(
        {"m": _matrix_scorer(S)}, n_images=3, k_per_image=2, k_max=3)
    assert abs(m["m_covered@3"] - 2 / 3) < 1e-12
    assert abs(m["m_coverrank_mean@3"] - (2 + 3) / 2) < 1e-12   # covered only
    assert abs(m["m_coverrank_censored_mean@3"] - (2 + 3 + 3) / 3) < 1e-12


def test_functional_orthogonal_anchors():
    g = torch.Generator().manual_seed(1)
    N, K, D = 6, 3, 64
    img = torch.eye(N, D) + 0.01 * torch.randn(N, D, generator=g)
    text = img.unsqueeze(1) + 0.05 * torch.randn(N, K, D, generator=g)
    img_n = torch.nn.functional.normalize(img, dim=-1)
    cap_n = torch.nn.functional.normalize(text.reshape(N * K, D), dim=-1)
    scorer = lambda sl: img_n[sl] @ cap_n.T                # noqa: E731

    m = compute_multicaption_coverrank({"cos": scorer}, N, K, k_max=N * K)
    assert abs(m[f"cos_coverrank_mean@{N * K}"] - K) < 1e-12  # every R_i == K
    assert abs(m[f"cos_covered@{N * K}"] - 1.0) < 1e-12


def test_k_max_validation():
    S = torch.zeros(2, 4)
    for bad in (1, 5):
        try:
            compute_multicaption_coverrank(
                {"m": _matrix_scorer(S)}, n_images=2, k_per_image=2, k_max=bad)
            raise AssertionError(f"k_max={bad} must raise")
        except ValueError:
            pass


if __name__ == "__main__":
    test_exact_values()
    test_consistency_with_allhit()
    test_chunking_invariance()
    test_censoring()
    test_functional_orthogonal_anchors()
    test_k_max_validation()
    print("All cover-rank metric tests passed.")
