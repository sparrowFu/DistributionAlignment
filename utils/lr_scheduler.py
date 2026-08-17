"""
Learning-rate scheduling utilities: cosine annealing with linear warmup.

The schedule is a *pure function* of the epoch index. LR at epoch ``e`` depends
only on ``(e, total_epochs, warmup_epochs, min_lr_ratio)`` — so resume needs no
scheduler state: the epoch is already restored from the checkpoint, and the LR
for any epoch is recomputed deterministically.

The schedule is recomputable from the epoch alone, so a resumed run needs only the epoch and the captured base learning rates — no scheduler state to persist.
"""

import math
from typing import List, Optional


def cosine_warmup_factor(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> float:
    """LR multiplier in ``[min_lr_ratio, 1.0]`` for cosine annealing + warmup.

    * Warmup (``0 <= epoch < warmup_epochs``): linear ramp
      ``factor = (epoch + 1) / warmup_epochs`` (peak at the last warmup epoch).
    * Decay (``epoch >= warmup_epochs``): cosine from ~1.0 down toward
      ``min_lr_ratio`` over the remaining epochs.

    Args:
        epoch: 0-based epoch index.
        total_epochs: total number of training epochs (the schedule horizon).
        warmup_epochs: length of the linear warmup (0 disables warmup).
        min_lr_ratio: cosine floor as a fraction of the base LR (e.g. 0.02 ->
            decay to 2% of base LR).

    Returns:
        Multiplier in ``[min_lr_ratio, 1.0]``; ``1.0`` when ``total_epochs <= 0``.
    """
    if total_epochs <= 0:
        return 1.0
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    decay_epochs = max(1, total_epochs - warmup_epochs)
    progress = (epoch - warmup_epochs) / decay_epochs
    progress = min(1.0, max(0.0, progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_lr_for_epoch(
    optimizer,
    base_lrs: List[float],
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
    scheduler: str,
    logger: Optional[object] = None,
) -> float:
    """Set each param group's LR for ``epoch`` under the configured schedule.

    When ``scheduler == "none"`` the optimizer is left untouched and ``1.0`` is
    returned (constant-LR behavior). Otherwise each param group's LR is set to
    ``base_lrs[i] * factor``.

    Returns the applied factor.
    """
    if scheduler == "none":
        return 1.0
    factor = cosine_warmup_factor(epoch, total_epochs, warmup_epochs, min_lr_ratio)
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base * factor
    if logger is not None:
        lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
        logger.info(
            f"Epoch {epoch + 1}/{total_epochs} lr factor={factor:.4f} -> {lrs}"
        )
    return factor
