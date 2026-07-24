"""
GaussianImageDistribution - Dataset registry.

Single source of truth for the ``--dataset`` tag. Train/eval data selection,
checkpoint naming, and argparse choices all read from here, so adding a dataset
is a one-place change:

  1. add its path constants to ``config.py`` (real on-disk paths);
  2. add one entry to ``DATASETS`` below (and, only if it has a brand-new file
     format, a loader ``Dataset`` class + a matching ``train_kind``/``eval_kind``
     branch in ``utils.dataset_factory`` / ``utils.eval_common``).

Real on-disk paths (verified):
  coco   -> TrainDatasets/mscoco_captions/{captions/*.parquet, images}
  flickr -> TrainDatasets/flickr30k/{captions.txt, flickr30k_images}

The registry itself imports only ``config`` (no data modules / torch), so
importing it — e.g. for argparse ``choices`` — stays cheap. Loader classes are
imported lazily by the factory functions, keyed on ``train_kind`` / ``eval_kind``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """How to build train/eval data for one dataset tag.

    Attributes:
        num_captions: default captions per image for this dataset.
        train_kind: factory key consumed by
            :func:`utils.dataset_factory.build_train_dataset`
            (``"image_caption"`` | ``"flickr_train"``).
        eval_kind: factory key consumed by
            :func:`utils.eval_common.build_eval_dataloader`
            (``"image_caption"`` | ``"flickr_test"``).
        accepts_path_overrides: whether ``--captions-path`` / ``--images-dir``
            apply (coco yes; flickr uses fixed ``config.FLICKR30K_*`` paths).
    """
    num_captions: int
    train_kind: str
    eval_kind: str
    accepts_path_overrides: bool = False


# Importing config here is fine (config imports no data modules -> no cycle).
import config


DATASETS = {
    "coco": DatasetSpec(
        num_captions=config.NUM_CAPTIONS,
        train_kind="image_caption",   # ImageCaptionDataset(MSCOCO parquet + images)
        eval_kind="image_caption",    # same, with optional num_samples subset
        accepts_path_overrides=True,
    ),
    "flickr": DatasetSpec(
        num_captions=config.FLICKR30K_NUM_CAPTIONS,
        train_kind="flickr_train",    # Flickr30KDataset(split="train") -> excludes test
        eval_kind="flickr_test",      # Flickr30KDataset(split="test") via get_flickr30k_test_loader
        accepts_path_overrides=False,
    ),
}

# The canonical, ordered list of supported dataset tags. argparse ``choices``
# and main.py derive from this so a new DATASETS entry is picked up everywhere.
VALID_DATASETS = tuple(DATASETS.keys())


def get_dataset_spec(dataset: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for ``dataset``, or raise ``ValueError``."""
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset tag {dataset!r}. Expected one of {VALID_DATASETS}.")
    return DATASETS[dataset]
