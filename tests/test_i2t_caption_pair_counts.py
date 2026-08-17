"""
Tests for compute_i2t_caption_pair_counts (I2T per-caption pair-count metric).

Verifies, for image->text retrieval over the full per-caption gallery:
  1. a perfect-alignment case yields the max hit count (all K of an image's
     captions land in its top-K);
  2. on random features the vectorized implementation matches an independent
     naive double-loop reference, for BOTH the cosine and the MCDisp_Align scorers and
     across multiple query chunks (chunk_size < N exercises the global-index
     logic).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from utils.retrieval import compute_i2t_caption_pair_counts


def _naive(img_mu, img_logvar, text_mus, text_logvars, k_values, tau):
    """Independent double-loop reference implementation."""
    N, K, D = text_mus.shape
    g = text_mus.reshape(N * K, D)
    g_lv = text_logvars.reshape(N * K, D)
    g_n = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    i_n = img_mu / img_mu.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    i_scale = torch.sqrt(1.0 + img_logvar.exp().mean(-1))
    g_scale = torch.sqrt(1.0 + g_lv.exp().mean(-1))

    out = {}
    for name, is_mcdisp in (("cos", False), ("mcdisp_align", True)):
        for k in k_values:
            tot = 0
            for i in range(N):
                sim = i_n[i] @ g_n.T
                if is_mcdisp:
                    sim = sim / (tau * i_scale[i] * g_scale)
                order = torch.argsort(sim, descending=True)
                topk = set(order[:k].tolist())
                pos = set(range(i * K, i * K + K))
                tot += len(topk & pos)
            out[f"{name}_pair_count@{k}"] = tot / N
    return out


def test_perfect_alignment():
    """Each image's K captions are the unique top-K -> mean hit count == K."""
    N, K, D = 4, 5, 21
    text_mus = torch.zeros(N, K, D)
    weights = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    img_mu = torch.zeros(N, D)
    for i in range(N):
        for k in range(K):
            text_mus[i, k, i * K + k] = 1.0          # each caption = a unique basis vector
        img_mu[i, i * K:i * K + K] = weights         # image = weighted sum of its 5 captions
    img_logvar = torch.zeros(N, D)
    text_logvars = torch.zeros(N, K, D)

    out = compute_i2t_caption_pair_counts(
        img_mu, img_logvar, text_mus, text_logvars, k_values=[5, 10], tau=0.07)

    # All K captions of every image have positive sim and outrank all other
    # images' (zero-sim) captions -> mean hit count is exactly K=5 at K=5 and K=10.
    for k in (5, 10):
        assert abs(out[f"cos_pair_count@{k}"] - 5.0) < 1e-6, (k, out)
        assert abs(out[f"mcdisp_align_pair_count@{k}"] - 5.0) < 1e-6, (k, out)
    print("test_perfect_alignment OK:", out)


def test_matches_naive_reference():
    """Vectorized impl (multi-chunk) must equal the naive reference, both scorers."""
    torch.manual_seed(123)
    N, K, D = 12, 5, 16
    img_mu = torch.randn(N, D)
    img_logvar = torch.randn(N, D) * 0.5
    text_mus = torch.randn(N, K, D)
    text_logvars = torch.randn(N, K, D) * 0.5
    k_values = [5, 10]
    tau = 0.07

    got = compute_i2t_caption_pair_counts(
        img_mu, img_logvar, text_mus, text_logvars, k_values, tau=tau, chunk_size=5)
    exp = _naive(img_mu, img_logvar, text_mus, text_logvars, k_values, tau)

    for key in exp:
        assert abs(got[key] - exp[key]) < 1e-9, (key, got[key], exp[key], got, exp)
    print("test_matches_naive_reference OK:", got)


if __name__ == "__main__":
    test_perfect_alignment()
    test_matches_naive_reference()
    print("All i2t_caption_pair_counts tests passed.")
