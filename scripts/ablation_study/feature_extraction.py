"""Checkpoint -> features on a manifest split, for the ablation metric modules.

Loads an ``MCDispAlignModel`` checkpoint and encodes a manifest split
(deterministic first-K captions), returning everything the H1/H2/H3 and scorer
modules need:

    img_mu, img_logvar: (N, D)
    img_U:              (N, D, r) or None (diagonal-only checkpoints)
    text_mus, text_logvars: (N, K, D) per-caption parameters

All tensors are on CPU (float32); metric modules move chunks to the device.
"""

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

import config
from data.manifest_caption_dataset import ManifestCaptionDataset
from data.caption_dataset import filter_none_collate
from models.mcdisp_align_model import MCDispAlignModel
from utils.logger import get_logger


logger = get_logger("ablation_features")


@torch.no_grad()
def extract_features(
    checkpoint: Path,
    manifest_path: Path,
    num_captions: int = 5,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_images: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Encode a manifest split with a trained checkpoint.

    Args:
        checkpoint: model checkpoint path (``MCDispAlignModel.load`` format).
        manifest_path: manifest JSON (test or dev split).
        num_captions: captions per image (deterministic first-K, no padding --
            short-caption entries are filtered by the dataset).
        max_images: optional cap on the number of images evaluated.

    Returns:
        Dict with img_mu/img_logvar (N, D), img_U (N, D, r) or None,
        text_mus/text_logvars (N, K, D).
    """
    model = MCDispAlignModel()
    model.load(str(checkpoint))
    model = model.to(device).eval()

    ds = ManifestCaptionDataset(
        manifest_path, config.IMAGES_DIR,
        num_captions=num_captions, sample_mode="first",
    )
    if max_images is not None and max_images < len(ds):
        from torch.utils.data import Subset
        g = torch.Generator().manual_seed(config.SEED)
        idx = torch.randperm(len(ds), generator=g)[:max_images].tolist()
        ds = Subset(ds, idx)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS if num_workers is None else num_workers,
        collate_fn=filter_none_collate,
    )

    out = {k: [] for k in ("img_mu", "img_logvar", "img_U", "text_mus", "text_logvars")}
    n_done = 0
    for batch in loader:
        if batch is None:
            continue
        pixel_values = model.process_images(batch["image"]).to(device)
        B, K = len(batch["image"]), len(batch["captions"][0])
        caps = [c for cl in batch["captions"] for c in cl]
        ti = model.process_text(caps)
        input_ids = ti["input_ids"].view(B, K, -1).to(device)
        attention_mask = ti["attention_mask"].view(B, K, -1).to(device)

        o = model(pixel_values, input_ids, attention_mask)
        out["img_mu"].append(o["img_mu"].cpu())
        out["img_logvar"].append(o["img_logvar"].cpu())
        out["text_mus"].append(o["text_mus"].cpu())
        out["text_logvars"].append(o["text_logvars"].cpu())
        if o["img_U"] is not None:
            out["img_U"].append(o["img_U"].cpu())
        n_done += B

    feats = {
        "img_mu": torch.cat(out["img_mu"], dim=0),
        "img_logvar": torch.cat(out["img_logvar"], dim=0),
        "text_mus": torch.cat(out["text_mus"], dim=0),
        "text_logvars": torch.cat(out["text_logvars"], dim=0),
        "img_U": torch.cat(out["img_U"], dim=0) if out["img_U"] else None,
    }
    logger.info(
        f"Extracted {feats['img_mu'].shape[0]} images x {feats['text_mus'].shape[1]} captions "
        f"from {checkpoint.name} (img_U={'none' if feats['img_U'] is None else tuple(feats['img_U'].shape[-1:])})"
    )
    return feats
