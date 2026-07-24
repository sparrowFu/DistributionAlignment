"""
Tests for utils/dataset_factory.build_train_dataset — the ``--dataset``
dispatch shared by the three fine-tuning scripts (clip_baseline, dist_align,
prolip), so ``--dataset coco|flickr`` selects the *training data* (not just the
checkpoint filename).

Mirrors utils.eval_common.build_eval_dataloader, which already does the same
for evaluation data: "coco" -> ImageCaptionDataset (MSCOCO train parquet),
"flickr" -> Flickr30KDataset (train split, excluding the held-out test split
that get_flickr30k_test_loader evaluates on).

Runnable without pytest:
    python tests/test_dataset_factory.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from utils.dataset_factory import build_train_dataset, VALID_DATASETS
from utils.dataset_registry import DATASETS, get_dataset_spec


def test_valid_datasets_tags():
    assert VALID_DATASETS == ("coco", "flickr")


def test_registry_is_source_of_truth():
    """Every tag has a well-formed spec; choices/factories all derive from it."""
    for tag in VALID_DATASETS:
        spec = get_dataset_spec(tag)
        assert spec.num_captions >= 1
        assert spec.train_kind in ("image_caption", "flickr_train")
        assert spec.eval_kind in ("image_caption", "flickr_test")
    # coco honors --captions-path/--images-dir; flickr uses fixed config paths
    assert DATASETS["coco"].accepts_path_overrides is True
    assert DATASETS["flickr"].accepts_path_overrides is False


def test_registry_unknown_raises():
    try:
        get_dataset_spec("imagenet")
    except ValueError as e:
        assert "imagenet" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown dataset tag")


def test_invalid_dataset_raises():
    try:
        build_train_dataset("imagenet")
    except ValueError as e:
        assert "imagenet" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown dataset tag")


def test_flickr_missing_root_raises():
    """A clear RuntimeError (not a silent coco fallback) when flickr30k is absent."""
    fake_root = Path("/nonexistent/flickr30k/xyz-12345")
    assert not fake_root.exists(), "test precondition: fake root must not exist"
    orig = config.FLICKR30K_ROOT
    config.FLICKR30K_ROOT = fake_root
    try:
        try:
            build_train_dataset("flickr")
        except RuntimeError as e:
            assert "flickr" in str(e).lower() or "Flickr30K" in str(e)
        else:
            raise AssertionError("expected RuntimeError when Flickr30K root is missing")
    finally:
        config.FLICKR30K_ROOT = orig


def test_coco_returns_dataset_when_available():
    if not config.CAPTIONS_PATH.exists():
        print("skip: MSCOCO captions parquet not present")
        return
    from data.caption_dataset import ImageCaptionDataset
    ds = build_train_dataset("coco")
    assert isinstance(ds, ImageCaptionDataset)
    assert len(ds) > 0


def test_flickr_returns_dataset_when_available():
    if not config.FLICKR30K_ROOT.exists():
        print("skip: Flickr30K not present")
        return
    from data.flickr30k_dataset import Flickr30KDataset
    ds = build_train_dataset("flickr")
    assert isinstance(ds, Flickr30KDataset)
    assert len(ds) > 0


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
