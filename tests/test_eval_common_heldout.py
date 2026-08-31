"""R01: COCO eval must use the held-out val slice training excluded.

Pure-logic tests for ``heldout_val_indices`` -- no ImageCaptionDataset or
image files needed. The contract: the helper reproduces exactly the split the
training scripts make (seed=``config.SEED``, ``val_split=0.1``), so the val
slice it returns is the one pool no model trained on.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import TensorDataset, random_split

import config
from utils.eval_common import heldout_val_indices


def test_split_covers_all_and_disjoint():
    """n=1000, seed=42: sizes 900/100, union covers everything, no overlap."""
    train_idx, val_idx = heldout_val_indices(1000, seed=42)
    assert len(val_idx) == 100
    assert len(train_idx) == 900
    assert set(train_idx) | set(val_idx) == set(range(1000))
    assert not (set(train_idx) & set(val_idx))
    # sorted ascending, valid index range
    assert val_idx == sorted(val_idx)
    assert all(0 <= i < 1000 for i in val_idx)


def test_equivalence_with_dataset_based_random_split():
    """Training splits a Dataset object; the helper splits a range.
    random_split permutes indices identically regardless of dataset content
    for the same n/seed, so the helper's indices must equal the training
    split's -- this is exactly the coupling the held-out protocol relies on.
    """
    g = torch.Generator().manual_seed(42)
    train_sub, val_sub = random_split(
        TensorDataset(torch.zeros(1000)), [900, 100], generator=g)
    train_idx, val_idx = heldout_val_indices(1000, seed=42)
    assert sorted(train_sub.indices) == train_idx
    assert sorted(val_sub.indices) == val_idx


def test_deterministic_across_calls_and_default_seed():
    """Same inputs -> same indices; default seed is config.SEED."""
    assert heldout_val_indices(1000, seed=42) == heldout_val_indices(
        1000, seed=42)
    assert heldout_val_indices(1000) == heldout_val_indices(
        1000, seed=config.SEED)


def test_val_size_matches_training_floor_math():
    """val_size uses the same floor math as the trainers: int(n * 0.1)."""
    assert len(heldout_val_indices(118283, seed=42)[1]) == 11828
    assert len(heldout_val_indices(999, seed=42)[1]) == 99
