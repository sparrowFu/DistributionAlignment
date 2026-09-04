"""
Tests for the all-hit@K multi-caption retrieval metric
(utils/retrieval.py: compute_multicaption_allhit).

  1. Exactness on handcrafted score matrices: the top-K rankings are known
     per image, so all-hit / any-hit / paircount have hand-computable values.
  2. Chunking invariance: chunk_rows=1 (one row per scorer call) must give
     byte-identical results to a single chunk.
  3. k_hit semantics: k_hit < K (precision-style) and k_hit > K (all-hit is
     0 by construction, any-hit saturates at the any-hit@K value).
  4. A functional cosine check: orthogonal images with own captions = image
     + tiny noise must retrieve the whole own set (all-hit == 1.0), and a
     swapped caption must break it.

Pure CPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.retrieval import compute_multicaption_allhit


def _matrix_scorer(S: torch.Tensor):
    """Scorer over a FIXED (N, N*K) score matrix, ignoring the slice content
    except its range (mirrors how real scorers slice their image-side state)."""
    return lambda sl: S[sl]


def test_exact_values_handcrafted_matrix():
    # N=3 images, K=2 captions -> gallery indices 0..5 (image-major: i*K + k).
    # Rankings per image (descending score):
    #   img0: c00 c01 c10 c11 c20 c21   -> top-2 = own set         (all-hit)
    #   img1: c10 c00 c11 c20 c21 c01   -> top-2 = 1 own, 1 foreign
    #   img2: c00 c10 c20 c01 c11 c21   -> top-2 = 0 own
    order = [[0, 1, 2, 3, 4, 5],
             [2, 0, 3, 1, 4, 5],
             [0, 2, 4, 1, 3, 5]]
    N, K = 3, 2
    S = torch.zeros(N, N * K)
    for i, ranking in enumerate(order):
        for rank, col in enumerate(ranking):
            S[i, col] = float(len(ranking) - rank)   # higher = ranked earlier

    metrics = compute_multicaption_allhit(
        {"m": _matrix_scorer(S)}, n_images=N, k_per_image=K, k_hit=K)

    assert abs(metrics["m_allhit@2"] - 1 / 3) < 1e-12      # only img0
    assert abs(metrics["m_anyhit@2"] - 2 / 3) < 1e-12      # img0, img1
    assert abs(metrics["m_paircount@2"] - 3 / 3) < 1e-12   # 2 + 1 + 0 = 3 / N


def test_chunking_invariance():
    g = torch.Generator().manual_seed(0)
    N, K, D = 11, 3, 8
    img = torch.randn(N, D)
    text = torch.randn(N, K, D)
    cap = text.reshape(N * K, D)
    S = img @ cap.T                                        # fixed score matrix

    full = compute_multicaption_allhit(
        {"m": _matrix_scorer(S)}, N, K, chunk_rows=512)
    chunked = compute_multicaption_allhit(
        {"m": _matrix_scorer(S)}, N, K, chunk_rows=1)
    assert full == chunked


def test_k_hit_semantics():
    # img0 ranking: c00 c01 c10 c11 c20 c21 (K=3)
    # img1 ranking: c10 c11 c13(foreign) c12 c20 ...  -> one foreign at rank 3
    N, K = 2, 3
    S = torch.zeros(N, N * K)
    # img0: own 0,1,2 first
    for rank, col in enumerate([0, 1, 2, 3, 4, 5]):
        S[0, col] = 6.0 - rank
    # img1 (own = 3,4,5): c10 c11 c00(foreign) c12 c01 c02
    for rank, col in enumerate([3, 4, 0, 5, 1, 2]):
        S[1, col] = 6.0 - rank

    # k_hit < K: precision-style -- top-2 all own for BOTH images
    m2 = compute_multicaption_allhit(
        {"m": _matrix_scorer(S)}, N, K, k_hit=2)
    assert abs(m2["m_allhit@2"] - 1.0) < 1e-12
    assert abs(m2["m_paircount@2"] - 2.0) < 1e-12

    # k_hit = K: img1's own c12 sits at rank 4 -> img1 fails all-hit
    m3 = compute_multicaption_allhit(
        {"m": _matrix_scorer(S)}, N, K, k_hit=K)
    assert abs(m3["m_allhit@3"] - 0.5) < 1e-12
    assert abs(m3["m_paircount@3"] - (3.0 + 2.0) / 2) < 1e-12

    # k_hit > K: all-hit is 0 by construction; any-hit@5 == any-hit@3 here
    # (img0: own all inside top-3 => inside top-5; img1: c10,c11 in top-5)
    m5 = compute_multicaption_allhit(
        {"m": _matrix_scorer(S)}, N, K, k_hit=5)
    assert m5["m_allhit@5"] == 0.0
    assert abs(m5["m_anyhit@5"] - 1.0) < 1e-12


def test_functional_cosine_scorer():
    g = torch.Generator().manual_seed(1)
    N, K, D = 6, 3, 64
    # orthogonal image anchors; own captions = anchor + small perturbation
    base = torch.eye(N, D)
    img = base + 0.01 * torch.randn(N, D, generator=g)
    text = img.unsqueeze(1) + 0.05 * torch.randn(N, K, D, generator=g)

    img_n = torch.nn.functional.normalize(img, dim=-1)
    cap_n = torch.nn.functional.normalize(text.reshape(N * K, D), dim=-1)
    scorer = lambda sl: img_n[sl] @ cap_n.T                # noqa: E731

    metrics = compute_multicaption_allhit({"cos": scorer}, N, K, k_hit=K)
    assert metrics["cos_allhit@3"] == 1.0
    assert metrics["cos_anyhit@3"] == 1.0
    assert metrics["cos_paircount@3"] == 3.0

    # swap one caption of image 0 with image 1's -> image 0 loses all-hit
    text[0, 0], text[1, 0] = text[1, 0].clone(), text[0, 0].clone()
    cap_n = torch.nn.functional.normalize(text.reshape(N * K, D), dim=-1)
    scorer = lambda sl: img_n[sl] @ cap_n.T                # noqa: E731
    metrics = compute_multicaption_allhit({"cos": scorer}, N, K, k_hit=K)
    assert metrics["cos_allhit@3"] == (N - 2) / N          # imgs 0 AND 1 broken
    assert metrics["cos_anyhit@3"] == 1.0                  # any-hit still fine


def test_k_hit_validation():
    S = torch.zeros(2, 4)
    try:
        compute_multicaption_allhit(
            {"m": _matrix_scorer(S)}, n_images=2, k_per_image=2, k_hit=0)
        raise AssertionError("k_hit=0 must raise")
    except ValueError:
        pass
    try:
        compute_multicaption_allhit(
            {"m": _matrix_scorer(S)}, n_images=2, k_per_image=2, k_hit=5)
        raise AssertionError("k_hit > N*K must raise")
    except ValueError:
        pass


if __name__ == "__main__":
    test_exact_values_handcrafted_matrix()
    test_chunking_invariance()
    test_k_hit_semantics()
    test_functional_cosine_scorer()
    test_k_hit_validation()
    print("All all-hit metric tests passed.")
