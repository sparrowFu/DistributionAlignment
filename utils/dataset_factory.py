"""
GaussianImageDistribution - Shared training-dataset selection.

The single source of truth for turning a ``--dataset`` tag into the *training*
``Dataset``, so the three fine-tuning scripts (``train_clip_baseline``,
``train_mcdisp_align``, ``train_prolip``) all switch training data by
``--dataset`` the same way the evaluation scripts switch eval data via
:func:`utils.eval_common.build_eval_dataloader`.

  - ``coco``   -> :class:`data.caption_dataset.ImageCaptionDataset`
                  (MSCOCO train parquet + images, via config defaults)
  - ``flickr`` -> :class:`data.flickr30k_dataset.Flickr30KDataset` with
                  ``split="train"`` (the first 90% of images). This excludes the
                  held-out Flickr30K *test* split that
                  :func:`data.flickr30k_dataset.get_flickr30k_test_loader`
                  evaluates on, so training and evaluation data never overlap.

Both datasets yield ``{"image", "captions", ...}`` samples and work with
:func:`data.caption_dataset.filter_none_collate`, so callers can build their
DataLoaders and train/val ``random_split`` exactly as before.

``--captions-path`` / ``--images-dir`` are honored for ``coco`` only (matching
the eval scripts, where those flags are documented as coco-only); ``flickr``
always uses the ``config.FLICKR30K_*`` paths.
"""

from typing import Optional

from torch.utils.data import Dataset

import config
from utils.dataset_registry import VALID_DATASETS, get_dataset_spec  # noqa: F401 (re-export)


def build_train_dataset(
    dataset: str,
    num_captions: Optional[int] = None,
    captions_path: Optional[str] = None,
    images_dir: Optional[str] = None,
) -> Dataset:
    """Build the full training ``Dataset`` for the given ``--dataset`` tag.

    Dispatches on the dataset's ``train_kind`` (see :mod:`utils.dataset_registry`),
    so the set of supported datasets is defined entirely by the registry. The
    returned dataset is the full training pool *before* the caller's train/val
    ``random_split`` — i.e. the same role the old hardcoded
    ``ImageCaptionDataset(config.CAPTIONS_PATH, ...)`` played.

    Args:
        dataset: dataset tag (see :data:`utils.dataset_registry.VALID_DATASETS`).
        num_captions: captions per image; ``None`` uses the dataset's registered
            default.
        captions_path: parquet path (coco only; overrides ``config.CAPTIONS_PATH``).
        images_dir: images directory (coco only; overrides ``config.IMAGES_DIR``).

    Raises:
        ValueError: unknown dataset tag or unsupported train kind.
        RuntimeError: ``flickr`` requested but Flickr30K is not on disk.
    """
    spec = get_dataset_spec(dataset)
    nc = num_captions or spec.num_captions

    if spec.train_kind == "image_caption":
        from data.caption_dataset import ImageCaptionDataset

        # Path overrides apply only to datasets that opt in (coco).
        cp = captions_path if spec.accepts_path_overrides else None
        idr = images_dir if spec.accepts_path_overrides else None
        return ImageCaptionDataset(
            captions_path=cp or config.CAPTIONS_PATH,
            images_dir=idr or config.IMAGES_DIR,
            num_captions=nc,
        )

    if spec.train_kind == "flickr_train":
        from data.flickr30k_dataset import Flickr30KDataset

        if not config.FLICKR30K_ROOT.exists():
            raise RuntimeError(
                f"Flickr30K dataset not available at {config.FLICKR30K_ROOT}. "
                f"Cannot train with --dataset flickr.")
        # split="train" (first 90%) excludes the eval test split (last 5%) -> no
        # train/test leakage. The caller carves a validation subset via random_split.
        return Flickr30KDataset(split="train", num_captions=nc)

    raise ValueError(
        f"No train loader registered for kind {spec.train_kind!r} "
        f"(dataset {dataset!r}). Add a branch in build_train_dataset.")
