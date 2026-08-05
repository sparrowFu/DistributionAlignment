"""
GaussianImageDistribution - Shared retrieval-evaluation helpers.

Centralizes the two things every retrieval-eval script derives from the
``--dataset`` tag, so checkpoint auto-selection and eval-data selection stay in
one place:

  - resolve_checkpoint(model_name, dataset): the checkpoint trained on that
    dataset, following the {model}_{dataset}_{best|last}.pt naming convention.
  - build_eval_dataloader(dataset, ...): the evaluation DataLoader for that
    dataset ("coco" -> MSCOCO via ImageCaptionDataset, "flickr" -> Flickr30K
    test split).

Both "coco" (MSCOCO) and "flickr" (flickr30k) produce {"image", "captions"}
batches with diagonal image<->caption pairing, so callers can reuse the same
feature-extraction and recall code regardless of dataset.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

import config
from utils.dataset_registry import VALID_DATASETS, get_dataset_spec  # noqa: F401 (re-export)


def resolve_checkpoint(model_name: str, dataset: str, which: str = "best") -> Path:
    """Checkpoint path for a model trained on ``dataset``.

    Convention: ``{model_name}_{dataset}_{which}.pt`` -- e.g.
    ``resolve_checkpoint("prolip", "coco")`` ->
    ``checkpoints/prolip_coco_best.pt``.

    Args:
        model_name: model prefix ("mcdisp_align", "clip_baseline", "prolip").
        dataset: dataset tag (see :data:`utils.dataset_registry.VALID_DATASETS`).
        which: "best" (default) or "last".
    """
    if dataset not in VALID_DATASETS:
        raise ValueError(
            f"Unknown dataset tag {dataset!r}. Expected one of {VALID_DATASETS}.")
    if which not in ("best", "last"):
        raise ValueError(f"`which` must be 'best' or 'last', got {which!r}.")
    return config.CHECKPOINT_DIR / f"{model_name}_{dataset}_{which}.pt"


def build_eval_dataloader(
    dataset: str,
    batch_size: int,
    num_workers: int,
    num_samples: Optional[int] = None,
    num_captions: Optional[int] = None,
    captions_path: Optional[str] = None,
    images_dir: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, int]:
    """Build the evaluation DataLoader for the given dataset tag.

    Dispatches on the dataset's ``eval_kind`` (see :mod:`utils.dataset_registry`),
    so the set of supported datasets is defined entirely by the registry:

    - ``image_caption`` -> MSCOCO via ``ImageCaptionDataset`` (config captions/
      images paths, or the optional ``captions_path``/``images_dir`` overrides),
      with an optional deterministic random ``num_samples`` subset.
    - ``flickr_test`` -> Flickr30K test split via ``get_flickr30k_test_loader``
      (full test set; no subsetting).

    Returns ``(dataloader, num_eval_samples)`` where ``num_eval_samples`` is the
    number of images in the (possibly subsetted) dataset actually iterated.
    """
    spec = get_dataset_spec(dataset)

    if spec.eval_kind == "image_caption":
        from data.caption_dataset import ImageCaptionDataset, filter_none_collate

        ds = ImageCaptionDataset(
            captions_path=captions_path or config.CAPTIONS_PATH,
            images_dir=images_dir or config.IMAGES_DIR,
            num_captions=num_captions or spec.num_captions,
        )

        if num_samples and num_samples < len(ds):
            g = torch.Generator().manual_seed(
                config.SEED if seed is None else seed)
            indices = torch.randperm(len(ds), generator=g)[:num_samples].tolist()
            ds = Subset(ds, indices)

        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=filter_none_collate,
        )
        return dl, len(ds)

    if spec.eval_kind == "flickr_test":
        from data.flickr30k_dataset import get_flickr30k_test_loader

        flickr = get_flickr30k_test_loader(
            batch_size=batch_size,
            num_workers=num_workers,
            num_captions=num_captions or spec.num_captions,
        )
        if flickr is None:
            raise RuntimeError(
                f"Flickr30K dataset not available at {config.FLICKR30K_ROOT}. "
                f"Cannot run evaluation with --dataset flickr.")
        return flickr["dataloader"], len(flickr["dataset"])

    raise ValueError(
        f"No eval loader registered for kind {spec.eval_kind!r} "
        f"(dataset {dataset!r}). Add a branch in build_eval_dataloader.")
