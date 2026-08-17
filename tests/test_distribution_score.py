"""Tests for the image-text log-likelihood matrix scorer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.distribution_score import image_text_loglik_matrix


def _diag_loglik(text, mean, var):
    """Closed-form log N(text; mean, diag(var)) up to const, NO per-dim norm, WITH logdet."""
    diff = text - mean
    return -0.5 * (diff ** 2 / var).sum(-1) - 0.5 * torch.log(var).sum(-1)


def test_diagonal_matches_closed_form():
    torch.manual_seed(0)
    N, M, D = 3, 5, 8
    img_mean = torch.randn(N, D)
    img_var = torch.rand(N, D) * 0.05 + 0.01
    text_mean = torch.randn(M, D)
    S = image_text_loglik_matrix(img_mean, img_var, None, text_mean,
                                 per_dim_normalize=False, use_logdet=True, chunk_size=256)
    for n in range(N):
        ref = _diag_loglik(text_mean, img_mean[n], img_var[n])
        assert torch.allclose(S[n], ref, atol=1e-4), f"row {n}"


def test_lowrank_matches_bruteforce():
    torch.manual_seed(1)
    N, M, D, r = 2, 4, 6, 3
    img_mean = torch.randn(N, D)
    img_var = torch.rand(N, D) * 0.05 + 0.01
    img_U = torch.randn(N, D, r) * 0.1
    text_mean = torch.randn(M, D)
    S = image_text_loglik_matrix(img_mean, img_var, img_U, text_mean,
                                 per_dim_normalize=False, use_logdet=True, chunk_size=256)
    for n in range(N):
        Sigma = torch.diag(img_var[n]) + img_U[n] @ img_U[n].T
        diff = text_mean - img_mean[n]                 # (M, D)
        sol = torch.linalg.solve(Sigma, diff.T).T      # (M, D)
        mahal = (diff * sol).sum(-1)                   # (M,)
        ref = -0.5 * mahal - 0.5 * torch.linalg.slogdet(Sigma)[1]
        assert torch.allclose(S[n], ref, atol=1e-3), f"row {n}"


def test_chunking_matches_full():
    torch.manual_seed(2)
    N, M, D, r = 10, 7, 8, 3
    img_mean = torch.randn(N, D)
    img_var = torch.rand(N, D) * 0.05 + 0.01
    img_U = torch.randn(N, D, r) * 0.1
    text_mean = torch.randn(M, D)
    S1 = image_text_loglik_matrix(img_mean, img_var, img_U, text_mean, chunk_size=3)
    S2 = image_text_loglik_matrix(img_mean, img_var, img_U, text_mean, chunk_size=256)
    assert torch.allclose(S1, S2, atol=1e-5)


def test_positive_pair_scores_highest():
    torch.manual_seed(3)
    D = 16
    img_mean = torch.randn(1, D)
    img_var = torch.ones(1, D) * 0.01
    texts = torch.randn(6, D)
    texts[0] = img_mean[0]  # text 0 is exactly the image mean
    S = image_text_loglik_matrix(img_mean, img_var, None, texts,
                                 per_dim_normalize=True, use_logdet=True)
    assert int(S[0].argmax()) == 0


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
