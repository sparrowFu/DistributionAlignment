"""
PCME++ data pipeline (B5)

Reimplements the official PCME++ dataset behavior (pcmepp/datasets.py in the
original repo, not vendored) on top of the project's ImageCaptionDataset:

- Caption-level sampling: one (image, caption) item per caption (5 per
  image), shuffled each epoch with drop_last -- the official COCO regime.
- Official train image augmentation: ONE crop ratio r ~ U(0.6, 1.0) applied
  to both dims, random-position crop, plain resize to 224x224 (aspect-ratio
  squashing), p=0.5 horizontal flip, /255 + ImageNet normalization.
  (PIL bilinear stands in for the official cv2.INTER_LINEAR.)
- Official eval image transform: resize to 224x224 + normalization only.
- Official caption token perturbation (process_caption_vit): per token,
  20% chance to mask (10%, 'M' token) / replace with a random id (2%,
  vocab = len(BertTokenizer) = 30522) / delete (8%); disabled at eval.
"""

import random
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pcmepp.modules import clip

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        tensor[c].sub_(m).div_(s)
    return tensor


def pcmepp_train_transform(image: Image.Image) -> torch.Tensor:
    """Official train preprocessing (datasets.py::_process_image, ViT path)."""
    image = image.convert("RGB")
    w, h = image.size

    # ONE ratio per sample for both dimensions (datasets.py:160-163)
    crop_ratio = np.random.random() * 0.4 + 0.6
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)
    if w - crop_w > 0:
        x_start = np.random.randint(w - crop_w)
    else:
        x_start = 0
    if h - crop_h > 0:
        y_start = np.random.randint(h - crop_h)
    else:
        y_start = 0
    image = image.crop((x_start, y_start, x_start + crop_w, y_start + crop_h))

    image = image.resize((224, 224), Image.BILINEAR)
    if np.random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255.0)
    return _normalize(tensor)


def pcmepp_eval_transform(image: Image.Image) -> torch.Tensor:
    """Official eval preprocessing: resize only + normalization.

    Aliased as ``pcmepp_eval_image_transform`` for reuse by models/pcmepp_model.py.
    """
    image = image.convert("RGB").resize((224, 224), Image.BILINEAR)
    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255.0)
    return _normalize(tensor)


# Alias used by models/pcmepp_model.py (single source of truth)
pcmepp_eval_image_transform = pcmepp_eval_transform


def process_caption_vit(tokens: List[int], train: bool, vocab_size: int = 30522,
                        size_augment: float = 0.2, masking: float = 0.5,
                        erasing: float = 0.1) -> List[int]:
    """Verbatim port of the official process_caption_vit token perturbation.

    ``tokens`` is the [sot, ..., eot] list from clip.tokenize2. ``vocab_size``
    is 30522: the official train.py passes len(BertTokenizer('bert-base-
    uncased').vocab); hardcoded here to avoid the network dependency.
    """
    sot = tokens[0]
    eot = tokens[-1]
    tokens = tokens[1:-1]

    output_tokens = []
    for token in tokens:
        if size_augment:
            prob = random.random()
            if prob < size_augment and train:
                prob /= size_augment
                if prob < masking:
                    # mask (masking fraction of the perturbed tokens)
                    output_tokens.append(clip._tokenizer.encoder['M'])
                elif prob < (masking + erasing):
                    # random token (erasing fraction)
                    output_tokens.append(random.choice(list(range(vocab_size))))
                else:
                    # delete
                    pass
            else:
                output_tokens.append(token)
        else:
            output_tokens.append(token)

    return [sot] + output_tokens + [eot]


def pcmepp_collate(batch) -> Optional[Dict[str, object]]:
    """Collate (image_tensor, token_list, base_index) items.

    Returns {"images": (B,3,224,224), "input_ids": (B,77) int64 zero-padded,
    "lengths": list[int], "img_ids": list[int]} or None when every item
    failed to load (project convention).
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    images, token_lists, img_ids = zip(*batch)
    images = torch.stack(images, dim=0)

    input_ids = torch.zeros(len(token_lists), 77, dtype=torch.long)
    lengths = []
    eot = clip._tokenizer.encoder["<|endoftext|>"]
    for i, tokens in enumerate(token_lists):
        # [DA] defensive cap at 77 (official tokenize2 can exceed it)
        tokens = tokens[:77]
        if tokens[-1] != eot:
            tokens[-1] = eot
        input_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        lengths.append(len(tokens))

    return {
        "images": images,
        "input_ids": input_ids,
        "lengths": lengths,
        "img_ids": list(img_ids),
    }


def pcmepp_worker_init_fn(worker_id: int) -> None:
    """Re-seed the global python/numpy RNGs inside DataLoader workers (the
    vendored augmentation uses them; official has no worker seeding)."""
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class PCMEPPCaptionView(Dataset):
    """Caption-level view over a base image-level dataset (or Subset).

    Item i -> base item i // K with caption (i % K). len = K * len(base).
    train=True applies the official train transform + caption perturbation;
    train=False uses caption[0] with eval transforms (deterministic).
    Propagates None items from the base dataset (missing images).
    """

    def __init__(self, base, train: bool, num_captions: int = 5):
        self.base = base
        self.train = train
        self.num_captions = num_captions

    def __len__(self) -> int:
        return len(self.base) * self.num_captions

    def __getitem__(self, index: int):
        base_idx = index // self.num_captions
        cap_idx = index % self.num_captions

        item = self.base[base_idx]
        if item is None:
            return None

        image = item["image"]
        captions = item["captions"]

        if self.train:
            image_tensor = pcmepp_train_transform(image)
            caption = captions[cap_idx]
            tokens = process_caption_vit(
                clip.tokenize2(caption)[0],
                train=True,
                vocab_size=30522,
            )
        else:
            image_tensor = pcmepp_eval_transform(image)
            tokens = clip.tokenize2(captions[0])[0]

        return image_tensor, tokens, base_idx
