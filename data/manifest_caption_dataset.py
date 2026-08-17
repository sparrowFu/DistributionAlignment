"""Manifest-backed image-caption Dataset for the MCDisp_Align ablation study.

Implements the data preconditions of the ablation experiment plan
(``docs/mcdisp_align_ablation_experiment_plan.md`` §3.1–§3.2):

  * Image-exclusive splits. A manifest is a JSON list of entries
    ``{"image": <file name>, "captions": [str, ...], "n_valid": int}`` where each
    entry is one image, so an image can only appear in one split manifest.
  * No repeat-padding. Entries with fewer valid captions than requested are
    FILTERED (training) rather than padded by repeating caption 0, so every
    caption a loss sees is a real, distinct description. ``n_valid`` records the
    true caption count for auditing.
  * Random-without-replacement caption sampling (``sample_mode="random"``):
    each ``__getitem__`` draws ``num_captions`` distinct captions at random, so
    K=1 / K=3 training regimes re-sample per epoch as the plan requires
    ("每张图每轮随机选择一条/三条有效 caption").
  * Deterministic evaluation (``sample_mode="first"``): the first K captions in
    manifest order, identical across runs/configs.

Yields the same sample dict shape as
:class:`data.caption_dataset.ImageCaptionDataset`, so ``filter_none_collate``
and the training loop work unchanged.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from utils.logger import get_logger


logger = get_logger("manifest_caption_dataset")


class ManifestCaptionDataset(Dataset):
    """Dataset over one image-exclusive manifest split.

    Args:
        manifest_path: path to the manifest JSON produced by the audit phase.
        images_dir: directory containing the image files.
        num_captions: captions per sample. Entries with fewer valid captions
            are filtered out -- captions are NEVER padded by repetition
            (plan §3.2), so the dataset only serves real distinct captions.
        sample_mode: ``"first"`` (deterministic first-K) or ``"random"``
            (random K without replacement per access).
    """

    def __init__(
        self,
        manifest_path: Path,
        images_dir: Path,
        num_captions: int = 5,
        sample_mode: str = "first",
    ):
        if sample_mode not in ("first", "random"):
            raise ValueError(f"sample_mode must be 'first' or 'random', got {sample_mode!r}")
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        entries = blob["entries"] if isinstance(blob, dict) else blob

        self.manifest_path = manifest_path
        self.images_dir = Path(images_dir)
        self.num_captions = num_captions
        self.sample_mode = sample_mode

        kept = []
        for e in entries:
            caps = [c for c in e["captions"] if isinstance(c, str) and c.strip()]
            if len(caps) >= num_captions:
                kept.append({"image": e["image"], "captions": caps, "n_valid": len(caps)})
        self.entries = kept
        dropped = len(entries) - len(kept)

        logger.info(
            f"ManifestCaptionDataset({manifest_path.name}): {len(self.entries)} entries "
            f"(dropped {dropped} with < {num_captions} valid captions), "
            f"K={num_captions}, sample_mode={sample_mode}"
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Optional[Dict[str, object]]:
        entry = self.entries[idx]
        image_path = self.images_dir / entry["image"]
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return None
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:  # noqa: BLE001 - mirror ImageCaptionDataset behavior
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None

        caps = entry["captions"]
        if self.sample_mode == "random" and len(caps) > self.num_captions:
            # Random WITHOUT replacement ("随机无放回抽样").
            idxs = random.sample(range(len(caps)), self.num_captions)
            captions = [caps[i] for i in idxs]
        else:
            captions = caps[: self.num_captions]

        return {
            "image": image,
            "image_path": str(image_path),
            "image_name": entry["image"],
            "captions": captions,
        }
