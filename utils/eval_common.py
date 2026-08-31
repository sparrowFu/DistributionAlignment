"""Shared retrieval-evaluation helpers: resolve a model's checkpoint path and build the evaluation DataLoader for a ``--dataset`` tag. Both supported datasets yield {"image", "captions"} batches with diagonal image<->caption pairing."""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split

import config
from utils.dataset_registry import VALID_DATASETS, get_dataset_spec  # noqa: F401 (re-export)


def heldout_val_indices(
    n: int, val_split: float = 0.1, seed: Optional[int] = None
) -> Tuple[list, list]:
    """Indices (into the full dataset) of the held-out val slice that
    training's random_split excludes -- the only pool eligible for COCO eval.

    Mirrors the training scripts' split exactly (``val_size = int(n *
    val_split)`` then ``random_split([n - val_size, val_size])`` under
    ``manual_seed(seed)``): random_split permutes indices identically for any
    dataset of the same length, so splitting a ``range`` yields the same
    indices as splitting the actual dataset.

    Returns ``(train_idx, val_idx)``, each sorted ascending.
    """
    g = torch.Generator().manual_seed(config.SEED if seed is None else seed)
    val_size = int(n * val_split)
    train_size = n - val_size
    train_subset, val_subset = random_split(
        range(n), [train_size, val_size], generator=g)
    return sorted(train_subset.indices), sorted(val_subset.indices)


def resolve_checkpoint(model_name: str, dataset: str, which: str = "best") -> Path:
    """Checkpoint path for a model trained on ``dataset``.

    Convention: ``{model_name}_{dataset}_{which}.pt`` -- e.g.
    ``resolve_checkpoint("prolip", "coco")`` ->
    ``checkpoints/prolip_coco_best.pt``.

    Args:
        model_name: model prefix ("mcdisp_align", "clip_baseline", "prolip").
        dataset: dataset tag.
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

    Dispatches on the dataset's ``eval_kind``,
    so the set of supported datasets is defined entirely by the registry:

    - ``image_caption`` -> MSCOCO via ``ImageCaptionDataset`` (config captions/
      images paths, or the optional ``captions_path``/``images_dir`` overrides),
      restricted to the held-out val slice that training EXCLUDED (R01):
      locally only the TRAIN parquet exists, and every training script
      (clip_baseline / prolip / mcdisp_align) splits it with seed=
      ``config.SEED`` and ``val_split=0.1`` via ``random_split`` -- so this
      branch mirrors that split (see ``heldout_val_indices``) and keeps only
      the ~10% val slice, the one pool no model trained on. An optional
      deterministic random ``num_samples`` subset is then drawn from that
      held-out pool. COUPLED to the training scripts: any change to their
      seed/val_split must be mirrored here or the pool is no longer
      held-out. If a Karpathy test split becomes available, replace this
      branch with it.
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

        # R01: 本地 COCO 只有训练 parquet。三个训练脚本统一以
        # seed=SEED、val_split=0.1 的 random_split 排除 10% 作为验证——
        # 评测只用这个所有模型都没训练过的池，替代原先“训练池随机抽
        # 5000”的泄漏协议。若改用 Karpathy test split，请替换此分支。
        _, val_idx = heldout_val_indices(len(ds), seed=seed)
        ds = Subset(ds, val_idx)

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
