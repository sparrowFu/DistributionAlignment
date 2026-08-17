"""
GPU Image Preprocessing

CLIP-equivalent image preprocessing (resize + center crop + normalize) that runs
on the GPU instead of the CPU main thread.

This mirrors HuggingFace's ``CLIPImageProcessor`` step by step so the output is
numerically equivalent to
``processor(images=imgs, return_tensors="pt")["pixel_values"]``:

1. Resize the shortest edge to 224 with BICUBIC + antialias, using the *same*
   target dimensions CLIP computes (``round(orig * 224 / min(h, w))``).
2. Center-crop to 224x224 with CLIP's floor-offset convention.
3. Scale [0, 255] -> [0, 1] and normalize with CLIP's mean/std.

Doing the resize/crop/normalize on ``device`` (e.g. cuda) removes the CPU
preprocessing bottleneck from the training loop. JPEG decoding still happens on
the CPU (in DataLoader workers, where the PIL images returned by the datasets are
decoded).
"""

from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# CLIP ViT-L/14 preprocessing constants.
CLIP_IMAGE_MEAN: Sequence[float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD: Sequence[float] = (0.26862954, 0.26130258, 0.27577711)
CLIP_IMAGE_SIZE: int = 224


def preprocess_images_on_gpu(
    images: List,
    device: torch.device,
    size: int = CLIP_IMAGE_SIZE,
    mean: Sequence[float] = CLIP_IMAGE_MEAN,
    std: Sequence[float] = CLIP_IMAGE_STD,
) -> torch.Tensor:
    """
    Preprocess a list of PIL images into a CLIP-compatible tensor on ``device``.

    Equivalent to ``CLIPImageProcessor``: shortest-edge BICUBIC resize (with
    antialias, matching CLIP's target dims) -> center crop -> scale [0, 255] ->
    [0, 1] -> normalize.

    Args:
        images: List of RGB PIL images (already decoded; decoding happens in the
                DataLoader workers).
        device: Device on which to run resize/crop/normalize (typically cuda).
        size: Target shortest-edge size and square center-crop size (CLIP = 224).
        mean: Per-channel normalization mean (CLIP defaults).
        std: Per-channel normalization std (CLIP defaults).

    Returns:
        Tensor of shape (B, 3, size, size) on ``device`` (requires_grad=False).
    """
    mean_t = torch.tensor(mean, device=device).view(3, 1, 1)
    std_t = torch.tensor(std, device=device).view(3, 1, 1)

    out: List[torch.Tensor] = []
    for image in images:
        h, w = image.height, image.width

        # PIL -> uint8 tensor (3, H, W); images are already decoded by workers.
        arr = np.array(image)                              # (H, W, 3) uint8
        t = torch.from_numpy(arr).permute(2, 0, 1)         # (3, H, W) uint8
        t = t.to(device, non_blocking=True).to(torch.float32)

        # Resize shortest edge to `size`, matching CLIPImageProcessor's exact
        # target dimensions (round(orig * size / min(h, w))).
        scale = size / float(min(h, w))
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        t = F.interpolate(
            t.unsqueeze(0), size=(new_h, new_w),
            mode="bicubic", antialias=True,
        ).squeeze(0)
        # PIL resizes in uint8 space and clamps to [0, 255]; bicubic on float can
        # overshoot, so clamp to match CLIP's preprocessing exactly.
        t = t.clamp(0, 255)

        # Center crop to size x size with CLIP's floor-offset convention.
        top = (new_h - size) // 2
        left = (new_w - size) // 2
        t = t[:, top:top + size, left:left + size]

        # Scale to [0, 1] and normalize with CLIP mean/std.
        t = (t / 255.0 - mean_t) / std_t

        out.append(t)

    return torch.stack(out, dim=0)
