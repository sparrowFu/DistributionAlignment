"""Fixed-B similar-sample batch sampler (plan: 固定批量相似样本组批续训方案 §5-6).

Pure INDEX logic -- no features, no torch tensors, no labels beyond the
image-identity structure of the training list. Three principles (plan §0)
are structural here: batch size B never changes, every image appears exactly
once per epoch (tail-rule inherited, not re-chosen), and images always travel
with their own five captions (batches are lists of TRAINING-SET indices; the
dataset/collate pair keeps image+captions bound).

Inputs:
  pools        list of per-pool index lists (random partition, W = mult*B)
  tables       per-pool candidate tables:
               {anchor_local: {"i2t": [neighbor_local...], "t2i": [...]}}
  P            planned anchor-neighbour pairs per batch (curriculum)
  B            fixed batch size
  rng          torch.Generator (independent sampler RNG, plan §8)
  pair_parity  global pair counter seed for I2T/T2I alternation

Outputs per pool: batches as lists of global indices + accounting stats
(planned/qualified/fallback pairs), sufficient for the §10 JSONL.
"""

from typing import Dict, List, Optional, Sequence

import torch


def make_pools(n_train: int, pool_size: int, rng: torch.Generator) -> List[List[int]]:
    """Random partition of all training indices into pools of ``pool_size``
    (last pool possibly smaller; plan §5.2)."""
    perm = torch.randperm(n_train, generator=rng).tolist()
    return [perm[i:i + pool_size] for i in range(0, len(perm), pool_size)]


def _pop_random(items: List[int], rng: torch.Generator) -> int:
    j = int(torch.randint(len(items), (1,), generator=rng).item())
    return items.pop(j)


def plan_pool_batches(pool: Sequence[int],
                      table: Dict[int, Dict[str, List[int]]],
                      P: int, B: int, rng: torch.Generator,
                      pair_counter_start: int = 0):
    """Plan batches inside ONE search pool (plan §6.2 pseudocode).

    Returns (batches, stats). Pairs alternate direction by a GLOBAL pair
    counter so I2T/T2I stay balanced across the whole epoch, not per pool.
    Random fallback is recorded, never silently widening the filters.
    """
    remaining = list(pool)
    batches: List[List[int]] = []
    stats = {"planned_pairs": 0, "qualified_pairs": 0, "fallback_pairs": 0}
    pair_counter = pair_counter_start

    while len(remaining) >= B:
        batch: List[int] = []
        for _ in range(P):
            if len(remaining) < 2:
                break
            anchor = _pop_random(remaining, rng)
            direction = "i2t" if pair_counter % 2 == 0 else "t2i"
            pair_counter += 1
            stats["planned_pairs"] += 1
            # candidate table is keyed by LOCAL pool position
            cands_local = table.get(pool.index(anchor), {}).get(direction, [])
            inter = [pool[c] for c in cands_local if pool[c] in set(remaining)]
            if inter:
                j = int(torch.randint(len(inter), (1,), generator=rng).item())
                neighbor = inter[j]
                remaining.remove(neighbor)
                stats["qualified_pairs"] += 1
            else:
                neighbor = _pop_random(remaining, rng)
                stats["fallback_pairs"] += 1
            batch.extend((anchor, neighbor))
        while len(batch) < B and remaining:
            batch.append(_pop_random(remaining, rng))
        order = torch.randperm(len(batch), generator=rng).tolist()
        batches.append([batch[i] for i in order])
    if remaining:                       # tail pool smaller than B: inherited rule
        batches.append(list(remaining))
    return batches, stats


def plan_epoch_batches(pools: List[List[int]],
                       tables: List[Dict[int, Dict[str, List[int]]]],
                       P: int, B: int, rng: torch.Generator,
                       shuffle_batches: bool = True):
    """All pools -> full epoch of batches + aggregate stats (plan §6.2 end).

    ``tables`` may be all-empty dicts (arm R: pure random batching) -- the
    same code path then reduces to random partitioning, guaranteeing R and G
    share exposure, batch count and tail rules (plan §8).
    """
    all_batches: List[List[int]] = []
    stats = {"planned_pairs": 0, "qualified_pairs": 0, "fallback_pairs": 0}
    pair_counter = 0
    for pool, table in zip(pools, tables):
        bs, st = plan_pool_batches(pool, table, P, B, rng,
                                   pair_counter_start=pair_counter)
        pair_counter += st["planned_pairs"]
        all_batches.extend(bs)
        for k in ("planned_pairs", "qualified_pairs", "fallback_pairs"):
            stats[k] += st[k]
    if shuffle_batches:
        order = torch.randperm(len(all_batches), generator=rng).tolist()
        all_batches = [all_batches[i] for i in order]
    # exposure invariants (§6.2): every image exactly once, no duplicates
    flat = [i for b in all_batches for i in b]
    assert len(flat) == len(set(flat)), "duplicate image across batches"
    assert sorted(flat) == sorted([i for p in pools for i in p]), "exposure mismatch"
    stats["n_batches"] = len(all_batches)
    stats["index_hash"] = hash(tuple(tuple(sorted(b)) for b in all_batches))
    return all_batches, stats


def batch_indices_to_batch_sampler(batches: List[List[int]],
                                   dataset_order: Optional[List[int]] = None):
    """Map training-set image indices to dataset positions for a DataLoader
    batch_sampler. ``dataset_order[i]`` = dataset position of image i
    (identity when the manifest order equals dataset order)."""
    if dataset_order is None:
        return [list(b) for b in batches]
    return [[dataset_order[i] for i in b] for b in batches]
