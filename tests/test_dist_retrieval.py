"""
Tests for the distribution-aware multi-caption retrieval families
(utils/retrieval.py: compute_multicaption_recall_dist):

  overlap -- Gaussian log-overlap score matrix (must equal the training-time
             L_match compatibility score, reusing gaussian_overlap_scores);
  ellip   -- negative Mahalanobis depth of each caption mean inside the image
             confidence ellipsoid (Woodbury form; checked against a direct
             dense-inverse reference).

Both share the N vs N*K any-hit protocol with the cosine family. Pure CPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from losses.gaussian_overlap import gaussian_overlap_scores
from utils.retrieval import (
    _ellipsoid_score_matrix,
    _overlap_score_matrix,
    compute_multicaption_recall_dist,
)


def _toy_batch(N=6, K=3, D=16, r=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    img_mu = torch.randn(N, D, generator=g)
    img_lv = torch.log(torch.rand(N, D, generator=g) * 0.2 + 0.05)
    img_U = 0.1 * torch.randn(N, D, r, generator=g)
    # each image's K captions scattered tightly around its own mean
    text_mus = img_mu.unsqueeze(1) + 0.05 * torch.randn(N, K, D, generator=g)
    text_lvs = torch.log(torch.rand(N, K, D, generator=g) * 0.1 + 0.05)
    return img_mu, img_lv, img_U, text_mus, text_lvs


def test_overlap_matrix_matches_reference():
    img_mu, img_lv, img_U, text_mus, text_lvs = _toy_batch()
    N, K, D = text_mus.shape
    cap_mu = text_mus.reshape(N * K, D)
    cap_d = torch.exp(text_lvs.reshape(N * K, D))
    ref = gaussian_overlap_scores(
        img_mu, torch.exp(img_lv), img_U, cap_mu, cap_d)
    got = _overlap_score_matrix(
        img_mu, torch.exp(img_lv), img_U, cap_mu, cap_d,
        pair_budget=1024)  # force small chunks -> exercises the double loop
    assert torch.allclose(ref, got, atol=1e-5)


def test_ellipsoid_matrix_matches_dense_inverse():
    img_mu, img_lv, img_U, text_mus, _ = _toy_batch()
    N, K, D = text_mus.shape
    cap_mu = text_mus.reshape(N * K, D)
    d = torch.exp(img_lv)
    got = _ellipsoid_score_matrix(img_mu, d, img_U, cap_mu, img_chunk=2)
    for i in range(N):
        Sig = torch.diag(d[i]) + img_U[i] @ img_U[i].T
            # +I jitter only for the dense reference inverse
        Sig = Sig + 1e-6 * torch.eye(D)
        Sig_inv = torch.linalg.inv(Sig)
        for j in range(N * K):
            dev = cap_mu[j] - img_mu[i]
            ref_m = dev @ Sig_inv @ dev
            assert torch.allclose(got[i, j], -ref_m, rtol=1e-4, atol=1e-2), (i, j)


def test_dist_recall_perfect_pairing():
    img_mu, img_lv, img_U, text_mus, text_lvs = _toy_batch()
    out = compute_multicaption_recall_dist(
        img_mu, img_lv, img_U, text_mus, text_lvs, [1, 5])
    for fam in ("overlap", "ellip"):
        for k in (1, 5):
            assert out[f"mc_{fam}_recall_i2t@{k}"] == 1.0
            assert out[f"mc_{fam}_recall_t2i@{k}"] == 1.0
            assert out[f"mc_{fam}_recall@{k}"] == 1.0


def test_dist_recall_broken_pairing_drops():
    img_mu, img_lv, img_U, text_mus, text_lvs = _toy_batch()
    # rotate the image-caption correspondence: image i now owns caption set
    # of image (i+1) % N -> no true positives remain
    text_mus_wrong = text_mus.roll(shifts=1, dims=0)
    text_lvs_wrong = text_lvs.roll(shifts=1, dims=0)
    out = compute_multicaption_recall_dist(
        img_mu, img_lv, img_U, text_mus_wrong, text_lvs_wrong, [1])
    for fam in ("overlap", "ellip"):
        assert out[f"mc_{fam}_recall_i2t@1"] < 0.5   # near-random, definitely not 1
        assert out[f"mc_{fam}_recall_t2i@1"] < 0.5


def test_dist_recall_diagonal_only_path():
    # img_U=None (cov_rank=0 model) must work for both families
    img_mu, img_lv, _, text_mus, text_lvs = _toy_batch()
    out = compute_multicaption_recall_dist(
        img_mu, img_lv, None, text_mus, text_lvs, [1, 5])
    for fam in ("overlap", "ellip"):
        assert out[f"mc_{fam}_recall@1"] == 1.0  # perfect pairing still holds
    # and the diagonal ellipsoid score equals the closed form (full pairwise)
    d = torch.exp(img_lv)
    cap = text_mus.reshape(-1, img_mu.shape[1])                    # (Q, D)
    ref = -(((img_mu[:, None, :] - cap[None, :, :]) ** 2) / d[:, None, :]).sum(-1)
    got = _ellipsoid_score_matrix(img_mu, d, None, cap)
    # the matmul expansion A + M2 - 2X cancels ~4 digits vs the direct
    # difference form at |mu|^2/d ~ 1e2 scale; ranking-equivalent precision
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-3)


if __name__ == "__main__":
    test_overlap_matrix_matches_reference()
    test_ellipsoid_matrix_matches_dense_inverse()
    test_dist_recall_perfect_pairing()
    test_dist_recall_broken_pairing_drops()
    test_dist_recall_diagonal_only_path()
    print("All distribution-aware retrieval tests passed.")
