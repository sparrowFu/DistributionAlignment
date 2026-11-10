"""
PCME++ loss/optimizer glue (B5)

The criterion itself is the vendored official implementation
(pcmepp/criterions/pcmepp.py). This module ports the training-loop glue from
the official pcmepp/engine.py (PCMEPPModel): the matched-matrix construction
(duplicate image ids), label smoothing, the single-GPU loss call, the
layerwise-LR-decay parameter grouping (verbatim, including the hardcoded 12),
the epoch-milestone LR factor and the loss-multiplied warmup.
"""

import re
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn


def build_matched_matrix(img_ids, device) -> Tensor:
    """Verbatim port of engine.py forward_step: identity matching plus
    marking captions of the SAME image as positives when the caption-level
    sampler puts several of them in one batch."""
    img_ids = list(img_ids)
    matched = torch.eye(len(img_ids)).to(device)
    if len(img_ids) != len(set(img_ids)):
        indices = {}
        seen = set()
        for idx, _id in enumerate(img_ids):
            if _id in seen:
                for _prev_idx in indices[_id]:
                    matched[idx][_prev_idx] = 1
                    matched[_prev_idx][idx] = 1
            seen.add(_id)
            indices.setdefault(_id, []).append(idx)
    return matched


def label_smooth_matched(matched: Tensor, smoothing: float = 1e-4) -> Tensor:
    """engine.py forward_step label-smoothing branch (applied when MSDA is off)."""
    return (1 - smoothing) * matched + smoothing * (
        torch.ones(matched.shape, device=matched.device) - matched
    )


def pcmepp_total_loss(img_emb: Dict[str, Tensor], cap_emb: Dict[str, Tensor],
                      criterion: nn.Module, matched: Tensor) -> Tuple[Tensor, Dict]:
    """Single-GPU loss call (official non-distributed path).

    The criterion internally doubles the BCE term (the i2t and t2i losses are
    the same under the elementwise-BCE formulation) and adds
    vib_beta * VIB + smoothness_alpha * (i2t + t2i pseudo-positive smoothing).
    """
    return criterion(img_emb, cap_emb, matched=matched)


def _llrd_param_groups(params: List[Tuple[str, Tensor]], base_lr: float,
                       layerwise_lr_decay: float, n_unc_layers: int,
                       regexp_pattern: str = r'resblocks\.(\d+)\.'
                       ) -> List[Dict]:
    """Verbatim port of engine.py::__get_params_with_layerwise_lr_decay.

    NOTE the hardcoded 12: for the 24-block ViT-L/14 vision tower, blocks
    12..23 get decay factors 0.7^(12-idx) > 1 (the tail block reaches
    ~0.7^-11 ~= 35.5). This is exactly what the official code computes for
    ViT-L/14 -- kept faithfully.
    """
    regex = re.compile(regexp_pattern)

    llrd_parameters = {}
    non_decayed_parameters = []
    for name, p in params:
        cur_block_cnt = regex.findall(name)
        if cur_block_cnt:
            cur_block_cnt = int(cur_block_cnt[0])
            # check for the probabilistic backbone
            if 'uncertainty' in name or 'prob_resblocks' in name:
                llrd_parameters.setdefault(f'prob_backbone.{cur_block_cnt}', []).append(p)
            else:
                llrd_parameters.setdefault(f'backbone.{cur_block_cnt}', []).append(p)
        else:
            non_decayed_parameters.append(p)
    grouped_parameters = []
    grouped_parameters.append(
        {'params': non_decayed_parameters, 'lr': base_lr})

    for name, params_ in llrd_parameters.items():
        block_cnt = int(name.split('.')[1])
        if 'prob_backbone' in name:
            decay_factor = layerwise_lr_decay ** (n_unc_layers - block_cnt)
        else:
            decay_factor = layerwise_lr_decay ** (12 - block_cnt)
        grouped_parameters.append(
            {'params': params_, 'lr': base_lr * decay_factor})
    return grouped_parameters


def build_param_groups(named_params: List[Tuple[str, Tensor]], lr: float,
                       layerwise_lr_decay: float, img_backbone_lr_decay: float,
                       txt_backbone_lr_decay: float, n_unc_layers: int,
                       total_params: int) -> List[Dict]:
    """Verbatim port of engine.py::configure_optimizers (the layerwise path,
    without the no_*_unc_decay variants -- the official config leaves them
    unset). Submodule names must be img_enc / txt_enc / criterion.

    Groups: txt_enc non-backbone and img_enc non-backbone at base lr;
    criterion scalars at base lr with weight_decay 0; img backbone at
    lr * img_backbone_lr_decay with per-block LLRD (reversed list order);
    txt backbone at lr * txt_backbone_lr_decay likewise.
    """
    params = list(named_params)
    grouped_parameters = [
        {'params': [
            p for name, p in params if 'txt_enc' in name and 'txt_enc.backbone' not in name],
         'lr': lr},
        {'params': [
            p for name, p in params if 'img_enc' in name and 'img_enc.backbone' not in name],
         'lr': lr},
        # do not apply weight decay for scalar values
        {'params': [
            p for name, p in params if 'txt_enc' not in name and 'img_enc' not in name],
         'lr': lr, 'weight_decay': 0},
    ]

    img_backbone_params = [
        (name, p) for name, p in params if 'img_enc.backbone' in name][::-1]
    grouped_parameters.extend(_llrd_param_groups(
        img_backbone_params, lr * img_backbone_lr_decay, layerwise_lr_decay,
        n_unc_layers))

    txt_backbone_params = [
        (name, p) for name, p in params if 'txt_enc.backbone' in name][::-1]
    grouped_parameters.extend(_llrd_param_groups(
        txt_backbone_params, lr * txt_backbone_lr_decay, layerwise_lr_decay,
        n_unc_layers))

    assert total_params == len([p for subparams in grouped_parameters
                                for p in subparams['params']]), \
        "parameter-group coverage mismatch (engine.py:477 invariant)"
    return grouped_parameters


def milestone_lr_factor(epoch: int, milestone: int = 15) -> float:
    """Official SequentialLR([ConstantLR(1, 15), ConstantLR(0.1, 10)]) scaled
    per epoch: 1.0 for epochs 0..14, 0.1 for 15..24. (Beyond 25 epochs the
    second ConstantLR expires and the official scheduler would revert to 1.0;
    training stops at 25 so this does not arise.)"""
    return 1.0 if epoch < milestone else 0.1


def warmup_alpha(epoch: int, batch_idx: int, num_batches: int,
                 embedding_warmup_epochs: int = 2, linear_warmup: bool = False,
                 embedding_warmup_after_freeze: bool = True) -> float:
    """Official warmup, which multiplies the LOSS (engine.py:233-245).

    With the official config (linear_warmup false, embedding_warmup_after_
    freeze true, embedding_warmup_epochs 2) only the intra-epoch ramp at
    epoch == 2 (the first epoch after the image-backbone freeze) is active.
    """
    alpha = 1.0
    if epoch < embedding_warmup_epochs and linear_warmup:
        alpha = (epoch * num_batches + batch_idx + 1) / \
            (embedding_warmup_epochs * num_batches)
    if epoch == embedding_warmup_epochs and embedding_warmup_after_freeze:
        # VSE infty style embedding warmup after freezing backbones
        alpha = (batch_idx + 1) / num_batches
    return alpha
