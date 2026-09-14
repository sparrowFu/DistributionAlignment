"""Training-set feature snapshots + confusability candidate tables
(plan: 固定批量相似样本组批续训方案 §5).

Snapshots are gradient-free eval-mode extractions used ONLY to build batch
indices (plan §5.1): the merged set parameters come from the model's own
moment-matching forward, never from an ad-hoc "average of normalized
captions". The native set logit z reuses the loss's exact similarity
semantics via an adapter that is consistency-tested against
``_sim_matrix`` in the test suite (plan §11: no private-formula rewrites).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class TrainSnapshot:
    """Plan §5.1 fields + normalized caption fingerprints (§5.5 rule 2)."""
    img_mu: torch.Tensor            # (N, D) raw
    img_logvar: torch.Tensor        # (N, D)
    text_mus: torch.Tensor          # (N, 5, D) raw per-caption means
    text_mu: torch.Tensor           # (N, D) model-merged set mean
    text_logvar: torch.Tensor       # (N, D) model-merged set logvar
    caption_fingerprints: List[frozenset]  # per image: normalized caption set
    image_ids: List[int]            # training-set indices, loader order

    def __len__(self):
        return self.img_mu.shape[0]


@torch.no_grad()
def snapshot_train_features(model, dataloader, device) -> TrainSnapshot:
    """Eval-mode, no-grad pass over the TRAIN split only (plan §5.1)."""
    model.eval()
    acc = {k: [] for k in ("img_mu", "img_logvar", "text_mus", "text_mu", "text_logvar")}
    fps: List[frozenset] = []
    ids: List[int] = []
    for batch in tqdm(dataloader, desc="snapshot"):
        if batch is None:
            continue
        pil_images, caption_lists = batch["image"], batch["captions"]
        B, K = len(pil_images), len(caption_lists[0])
        flat = [c for cl in caption_lists for c in cl]
        ti = model.process_text(flat)
        out = model(model.process_images(pil_images).to(device),
                    ti["input_ids"].to(device).view(B, K, -1),
                    ti["attention_mask"].to(device).view(B, K, -1))
        acc["img_mu"].append(out["img_mu"].float().cpu())
        acc["img_logvar"].append(out["img_logvar"].float().cpu())
        acc["text_mus"].append(out["text_mus"].float().cpu())
        acc["text_mu"].append(out["text_mu"].float().cpu())
        acc["text_logvar"].append(out["text_logvar"].float().cpu())
        for cl in caption_lists:      # §5.5: case-fold + whitespace collapse
            fps.append(frozenset(" ".join(c.lower().split()) for c in cl))
        ids.extend(range(len(ids), len(ids) + B))   # positional training index
    return TrainSnapshot(
        img_mu=torch.cat(acc["img_mu"]), img_logvar=torch.cat(acc["img_logvar"]),
        text_mus=torch.cat(acc["text_mus"]), text_mu=torch.cat(acc["text_mu"]),
        text_logvar=torch.cat(acc["text_logvar"]),
        caption_fingerprints=fps, image_ids=ids)


def native_set_logits(snap_subset: TrainSnapshot, *, tau: float,
                      use_uncertainty_sim: bool) -> torch.Tensor:
    """z[i, j] with the ORIGINAL training score semantics (plan §5.4).

    Adapter around the same formula as ``losses.*._sim_matrix``: cosine of
    L2-normalized means, optionally discounted by sqrt(1 + mean(sigma^2)) on
    BOTH sides with the MERGED set logvar on the text side. Test-verified
    against the loss implementation.
    """
    img_n = F.normalize(snap_subset.img_mu, dim=-1)
    txt_n = F.normalize(snap_subset.text_mu, dim=-1)
    base = img_n @ txt_n.T
    if not use_uncertainty_sim:
        return base / tau
    img_scale = torch.sqrt(1.0 + torch.exp(snap_subset.img_logvar).mean(dim=-1))
    txt_scale = torch.sqrt(1.0 + torch.exp(snap_subset.text_logvar).mean(dim=-1))
    return base / (tau * img_scale.unsqueeze(1) * txt_scale.unsqueeze(0))


def per_caption_confusability(snap_subset: TrainSnapshot):
    """h_I2T[i, j] = max_k cos(mu_v_i, mu_t_jk); h_T2I[i, j] = max_k
    cos(mu_v_j, mu_t_ik)  (plan §5.3)."""
    img_n = F.normalize(snap_subset.img_mu, dim=-1)                  # (W, D)
    N, K, D = snap_subset.text_mus.shape
    cap_n = F.normalize(snap_subset.text_mus.reshape(N * K, D), dim=-1)
    sims = img_n @ cap_n.T                                           # (W, W*K)
    h_i2t = sims.view(N, N, K).max(dim=-1).values                    # [i, j]
    h_t2i = h_i2t.T                                                  # max_k cos(mu_v_j, mu_t_ik)
    return h_i2t, h_t2i


def build_pool_candidate_table(snap_subset: TrainSnapshot, *,
                               tau: float, use_uncertainty_sim: bool,
                               rank_start: int = 5, rank_end: int = 64,
                               gap_min: float = 0.0, gap_max: float = 2.0):
    """Candidate table keyed by LOCAL pool position (plan §5.5).

    Filters, per anchor and direction:
      1. exclude self and duplicate-image relations (identical normalized
         caption SETS, §5.5 rule 2);
      2. rank by h descending, keep ranks [rank_start, rank_end];
      3. keep 0 <= g <= 2 on the native set logit in the matching direction
         (g = z_ii - z_ij for I2T, z_ii - z_ji for T2I, §5.4).
    """
    W = len(snap_subset)
    z = native_set_logits(snap_subset, tau=tau, use_uncertainty_sim=use_uncertainty_sim)
    h_i2t, h_t2i = per_caption_confusability(snap_subset)
    z_diag = torch.diagonal(z)
    dup = torch.zeros(W, W, dtype=torch.bool)
    for a in range(W):
        fa = snap_subset.caption_fingerprints[a]
        for b in range(a + 1, W):
            if snap_subset.caption_fingerprints[b] == fa:
                dup[a, b] = dup[b, a] = True

    table: Dict[int, Dict[str, List[int]]] = {}
    for a in range(W):
        entry: Dict[str, List[int]] = {}
        for direction, h, gap in (
                ("i2t", h_i2t, z_diag[a] - z[a, :]),      # g(i,j) = z_ii - z_ij
                ("t2i", h_t2i, z_diag[a] - z[:, a])):     # g(i,j) = z_ii - z_ji
            scores = h[a].clone()
            scores[a] = -2.0
            scores[dup[a]] = -2.0
            order = torch.argsort(scores, descending=True, stable=True)
            window = order[rank_start:min(rank_end, W - 1)]
            keep = [int(j) for j in window.tolist()
                    if gap_min <= float(gap[j]) <= gap_max]
            if keep:
                entry[direction] = keep
        if entry:
            table[a] = entry
    return table


def subset_snapshot(snap: TrainSnapshot, pool_local_positions: List[int]) -> TrainSnapshot:
    """Subset by POSITION within the snapshot (pool indices are snapshot
    positions); fingerprints follow so duplicate filtering stays per-pool."""
    idx = torch.as_tensor(pool_local_positions, dtype=torch.long)
    return TrainSnapshot(
        img_mu=snap.img_mu[idx], img_logvar=snap.img_logvar[idx],
        text_mus=snap.text_mus[idx], text_mu=snap.text_mu[idx],
        text_logvar=snap.text_logvar[idx],
        caption_fingerprints=[snap.caption_fingerprints[i] for i in pool_local_positions],
        image_ids=[snap.image_ids[i] for i in pool_local_positions])
