"""Build the training Dataset for a ``--dataset`` tag. Returns the full training pool (before the caller's train/val split) yielding ``{"image", "captions"}`` samples. ``--captions-path`` / ``--images-dir`` apply to ``coco`` only; ``flickr`` uses fixed config paths."""

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

    Dispatches on the dataset's ``train_kind``,
    so the set of supported datasets is defined entirely by the registry. The
    returned dataset is the full training pool *before* the caller's train/val
    ``random_split`` — i.e. the same role the old hardcoded
    ``ImageCaptionDataset(config.CAPTIONS_PATH, ...)`` played.

    Args:
        dataset: dataset tag.
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
