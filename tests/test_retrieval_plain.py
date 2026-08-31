"""Tests for compute_multicaption_recall_plain (shared N vs N*K protocol)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.retrieval import compute_multicaption_recall_plain, compute_recall_chunked


def test_crafted_ranking():
    """N=3, K=2, orthonormal basis. Captions: img0->e0 ×2, img1->e1 ×2,
    img2->e2 and (deliberately wrong) normalized e1+0.5*e2 (closest to
    image 1, runner-up image 2 -- strict order, no topk ties). So i2t
    any-hit = 3/3 (each image keeps one faithful caption), t2i = 5/6
    (caption (2,1) retrieves img1), and the true image ranks 2nd for it."""
    img = torch.zeros(3, 4)
    img[0, 0] = img[1, 1] = img[2, 2] = 1.0
    cap = torch.zeros(3, 2, 4)
    cap[0, 0, 0] = cap[0, 1, 0] = 1.0
    cap[1, 0, 1] = cap[1, 1, 1] = 1.0
    cap[2, 0, 2] = 1.0
    cap[2, 1, 1] = 1.0   # wrong: closest to image 1 ...
    cap[2, 1, 2] = 0.5   # ... with image 2 a strict runner-up (no ties)

    out = compute_multicaption_recall_plain(img, cap, [1, 2])
    assert abs(out["mc_recall_i2t@1"] - 1.0) < 1e-9
    assert abs(out["mc_recall_t2i@1"] - 5 / 6) < 1e-9
    assert abs(out["mc_recall@1"] - (1.0 + 5 / 6) / 2) < 1e-9
    assert abs(out["mc_recall_t2i@2"] - 1.0) < 1e-9   # true image ranks 2nd


def test_k1_equivalence_with_chunked():
    """K=1 must equal the classic diagonal-pairing recall."""
    torch.manual_seed(0)
    img = torch.randn(7, 6)
    cap = torch.randn(7, 1, 6)
    out = compute_multicaption_recall_plain(img, cap, [1, 5])
    ref_i2t = compute_recall_chunked(img, cap.squeeze(1), [1, 5])
    ref_t2i = compute_recall_chunked(cap.squeeze(1), img, [1, 5])
    for k in (1, 5):
        assert abs(out[f"mc_recall_i2t@{k}"] - ref_i2t[k]) < 1e-9
        assert abs(out[f"mc_recall_t2i@{k}"] - ref_t2i[k]) < 1e-9


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
        except Exception as e:  # noqa
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
