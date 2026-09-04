"""Shared image-text Recall@K computation. Supports both retrieval directions: I2T (query=image, gallery=text) and T2I (query=text, gallery=image)."""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def compute_recall_chunked(
    query: torch.Tensor,
    gallery: torch.Tensor,
    k_values: List[int],
    chunk_size: int = 1000,
    normalize: bool = True,
) -> Dict[int, float]:
    """
    Recall@K where query[i]'s positive match is gallery[i] (diagonal pairing).

    Direction-agnostic: pass images as ``query`` for I2T, texts as ``query``
    for T2I.

    Args:
        query: Query features (N, D)
        gallery: Gallery features (N, D); the i-th gallery item matches query i
        k_values: List of K values for Recall@K
        chunk_size: Chunk size to avoid OOM on large similarity matrices
        normalize: L2-normalize features before computing cosine similarity

    Returns:
        Dict mapping K -> recall@K (raw K ints, e.g. {1: 0.82, 5: 0.97, 10: 0.99})
    """
    n = query.shape[0]
    if normalize:
        query = F.normalize(query, dim=-1)
        gallery = F.normalize(gallery, dim=-1)

    hits = {k: 0 for k in k_values}

    for start in tqdm(range(0, n, chunk_size), desc="Recall chunks", leave=False):
        end = min(start + chunk_size, n)
        sim_chunk = torch.matmul(query[start:end], gallery.T)  # (chunk, N)
        ranked = torch.argsort(sim_chunk, dim=1, descending=True)
        gt = torch.arange(start, end, device=query.device).unsqueeze(1)
        for k in k_values:
            hits[k] += (ranked[:, :k] == gt).any(dim=1).sum().item()

    return {k: hits[k] / n for k in k_values}


@torch.no_grad()
def compute_recall_bidirectional(
    img_features: torch.Tensor,
    text_features: torch.Tensor,
    k_values: List[int],
    chunk_size: int = 1000,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Bidirectional (I2T + T2I) Recall@K between aligned image/text feature sets.

    Returns keys: ``recall_i2t@{k}``, ``recall_t2i@{k}``, ``recall@{k}`` (mean).
    """
    i2t = compute_recall_chunked(img_features, text_features, k_values, chunk_size, normalize)
    t2i = compute_recall_chunked(text_features, img_features, k_values, chunk_size, normalize)

    out: Dict[str, float] = {}
    for k in k_values:
        out[f"recall_i2t@{k}"] = i2t[k]
        out[f"recall_t2i@{k}"] = t2i[k]
        out[f"recall@{k}"] = (i2t[k] + t2i[k]) / 2
    return out


@torch.no_grad()
def compute_recall_mcdisp_align_chunked(
    img_mu: torch.Tensor,
    img_logvar: torch.Tensor,
    text_mu: torch.Tensor,
    text_logvar: torch.Tensor,
    k_values: List[int],
    tau: float = 0.07,
    chunk_size: int = 1000,
) -> Dict[str, float]:
    """
    Bidirectional Recall@K under the MCDisp_Align uncertainty-discounted cosine score,
    i.e. the same score used by the L_set contrastive loss (train and eval agree):

        sim(x, y) = (mu_x . mu_y) / (tau * sqrt(1 + mean(sigma_x^2)) * sqrt(1 + mean(sigma_y^2)))

    Means are L2-normalized internally; mean(sigma^2) averages over D.

    Returns keys: ``mcdisp_align_recall_i2t@{k}``, ``mcdisp_align_recall_t2i@{k}``,
    ``mcdisp_align_recall@{k}`` (mean of the two directions).
    """
    img_mu_n = F.normalize(img_mu, dim=-1)
    text_mu_n = F.normalize(text_mu, dim=-1)
    img_scale = torch.sqrt(1.0 + torch.exp(img_logvar).mean(dim=-1))    # (N,)
    text_scale = torch.sqrt(1.0 + torch.exp(text_logvar).mean(dim=-1))  # (N,)
    n = img_mu.shape[0]

    def _direction(query_mu, gallery_mu, query_scale, gallery_scale, desc):
        hits = {k: 0 for k in k_values}
        for start in tqdm(range(0, n, chunk_size), desc=desc, leave=False):
            end = min(start + chunk_size, n)
            sim = torch.matmul(query_mu[start:end], gallery_mu.T)
            scale = query_scale[start:end].unsqueeze(1) * gallery_scale.unsqueeze(0)
            sim = sim / (tau * scale)
            ranked = torch.argsort(sim, dim=1, descending=True)
            gt = torch.arange(start, end, device=query_mu.device).unsqueeze(1)
            for k in k_values:
                hits[k] += (ranked[:, :k] == gt).any(dim=1).sum().item()
        return {k: hits[k] / n for k in k_values}

    i2t = _direction(img_mu_n, text_mu_n, img_scale, text_scale, "MCDisp_Align I2T chunks")
    t2i = _direction(text_mu_n, img_mu_n, text_scale, img_scale, "MCDisp_Align T2I chunks")

    out: Dict[str, float] = {}
    for k in k_values:
        out[f"mcdisp_align_recall_i2t@{k}"] = i2t[k]
        out[f"mcdisp_align_recall_t2i@{k}"] = t2i[k]
        out[f"mcdisp_align_recall@{k}"] = (i2t[k] + t2i[k]) / 2
    return out


@torch.no_grad()
def compute_i2t_caption_pair_counts(
    img_mu: torch.Tensor,
    img_logvar: torch.Tensor,
    text_mus: torch.Tensor,
    text_logvars: torch.Tensor,
    k_values: List[int],
    tau: float = 0.07,
    chunk_size: int = 1000,
) -> Dict[str, float]:
    """
    Image->text retrieval against a per-caption gallery: for each image, count
    how many of its K captions land in the top-K retrieved, averaged over images.

    Unlike :func:`compute_recall_mcdisp_align_chunked` (1:1 against the single merged
    text mean), the gallery here is ALL N*K per-caption means, and image i has K
    positives -- its own captions at the flattened (image-major) indices
    ``[i*K, i*K+K)``. The reported value is the mean per-image hit count
    (range ``0..K``), under both the cosine and the MCDisp_Align uncertainty-discounted
    scores.

    Args:
        img_mu:        (N, D) image means
        img_logvar:    (N, D) image log-variances (MCDisp_Align scorer only)
        text_mus:      (N, K, D) per-caption means
        text_logvars:  (N, K, D) per-caption log-variances (MCDisp_Align scorer only)
        k_values:      top-K cutoffs for the hit count (e.g. [5, 10])
        tau:           MCDisp_Align score temperature
        chunk_size:    query (image) chunk size to bound the sim-matrix memory

    Returns:
        ``{cos_pair_count@{k}, mcdisp_align_pair_count@{k}}`` for each k in ``k_values``.
    """
    N, K, _ = text_mus.shape
    gallery_mu = text_mus.reshape(N * K, -1)             # (N*K, D)
    gallery_logvar = text_logvars.reshape(N * K, -1)     # (N*K, D)

    img_mu_n = F.normalize(img_mu, dim=-1)               # (N, D)
    gallery_mu_n = F.normalize(gallery_mu, dim=-1)       # (N*K, D)
    img_scale = torch.sqrt(1.0 + torch.exp(img_logvar).mean(dim=-1))          # (N,)
    gallery_scale = torch.sqrt(1.0 + torch.exp(gallery_logvar).mean(dim=-1))  # (N*K,)

    def _count(scorer, desc):
        hits = {k: 0 for k in k_values}
        for start in tqdm(range(0, N, chunk_size), desc=desc, leave=False):
            end = min(start + chunk_size, N)
            sim = scorer(img_mu_n[start:end], gallery_mu_n,
                         img_scale[start:end], gallery_scale)   # (chunk, N*K)
            ranked = torch.argsort(sim, dim=1, descending=True)
            rows = torch.arange(start, end, device=sim.device).unsqueeze(1)   # (chunk, 1) global img idx
            pos_lo = rows * K
            pos_hi = pos_lo + K
            for k in k_values:
                in_range = (ranked[:, :k] >= pos_lo) & (ranked[:, :k] < pos_hi)
                hits[k] += in_range.sum().item()
        return {k: hits[k] / N for k in k_values}

    def _cos(q, g, _qs, _gs):
        return torch.matmul(q, g.T)

    def _mcdisp_align(q, g, qs, gs):
        sim = torch.matmul(q, g.T)
        return sim / (tau * qs.unsqueeze(1) * gs.unsqueeze(0))

    cos = _count(_cos, "I2T pair-count (cos)")
    mcdisp_align = _count(_mcdisp_align, "I2T pair-count (mcdisp_align)")

    out: Dict[str, float] = {}
    for k in k_values:
        out[f"cos_pair_count@{k}"] = cos[k]
        out[f"mcdisp_align_pair_count@{k}"] = mcdisp_align[k]
    return out


@torch.no_grad()
def compute_multicaption_recall(
    img_mu: torch.Tensor,
    img_logvar: torch.Tensor,
    text_mus: torch.Tensor,
    text_logvars: torch.Tensor,
    k_values: List[int],
    tau: float = 0.07,
    chunk_size: int = 1000,
) -> Dict[str, float]:
    """Standard multi-caption bidirectional Recall@K (N images vs N*K captions).

    This is the canonical MS-COCO/Flickr30k retrieval protocol (evaluating on the merged text mean would collapse the one-to-many structure at eval time and is not comparable to published baselines):

      I2T: query = N image means, gallery = N*K per-caption means. Image i has K
           positives -- its own captions at flattened indices [i*K, i*K+K). A hit
           is ANY of them landing in the top-K (any-hit).
      T2I: query = N*K per-caption means, gallery = N image means. Each caption's
           ONLY positive is its own image (single-positive per query).

    Reported under BOTH the MCDisp_Align uncertainty-discounted score (primary) and plain
    cosine (for baseline comparison). Means are L2-normalized internally;
    mean(sigma^2) averages over D. Uses torch.topk (not a full argsort) since the
    N*K gallery is large and only the top max(k_values) are needed.

    Returns keys (per k in k_values)::

        mc_recall_i2t@{k}, mc_recall_t2i@{k}, mc_recall@{k}          (MCDisp_Align score)
        mc_cos_recall_i2t@{k}, mc_cos_recall_t2i@{k}, mc_cos_recall@{k} (cosine)
    """
    N, K, _ = text_mus.shape
    cap_mu = text_mus.reshape(N * K, -1)              # (N*K, D)
    cap_lv = text_logvars.reshape(N * K, -1)          # (N*K, D)

    img_mu_n = F.normalize(img_mu, dim=-1)            # (N, D)
    cap_mu_n = F.normalize(cap_mu, dim=-1)            # (N*K, D)
    img_scale = torch.sqrt(1.0 + torch.exp(img_logvar).mean(dim=-1))   # (N,)
    cap_scale = torch.sqrt(1.0 + torch.exp(cap_lv).mean(dim=-1))       # (N*K,)

    maxk = max(k_values)

    def _cos(q, g, _qs, _gs):
        return torch.matmul(q, g.T)

    def _mcdisp_align(q, g, qs, gs):
        return torch.matmul(q, g.T) / (tau * qs.unsqueeze(1) * gs.unsqueeze(0))

    def _i2t(scorer, desc):
        """N image queries vs N*K caption gallery; any-of-K-own-caption hit."""
        hits = {k: 0 for k in k_values}
        for start in tqdm(range(0, N, chunk_size), desc=desc, leave=False):
            end = min(start + chunk_size, N)
            sim = scorer(img_mu_n[start:end], cap_mu_n,
                         img_scale[start:end], cap_scale)           # (chunk, N*K)
            mk = min(maxk, sim.size(1))
            top = torch.topk(sim, mk, dim=1).indices               # (chunk, mk)
            rows = torch.arange(start, end, device=sim.device).unsqueeze(1)
            in_range = (top >= rows * K) & (top < rows * K + K)     # (chunk, mk)
            for k in k_values:
                hits[k] += in_range[:, :k].any(dim=1).sum().item()  # :k clamps to mk
        return {k: hits[k] / N for k in k_values}

    def _t2i(scorer, desc):
        """N*K caption queries vs N image gallery; single-positive per query."""
        Q = N * K
        hits = {k: 0 for k in k_values}
        for start in tqdm(range(0, Q, chunk_size), desc=desc, leave=False):
            end = min(start + chunk_size, Q)
            sim = scorer(cap_mu_n[start:end], img_mu_n,
                         cap_scale[start:end], img_scale)           # (chunk_q, N)
            mk = min(maxk, sim.size(1))
            top = torch.topk(sim, mk, dim=1).indices               # (chunk_q, mk)
            q_idx = torch.arange(start, end, device=sim.device)
            gt_img = (q_idx // K).unsqueeze(1)                      # (chunk_q, 1)
            match = top == gt_img                                   # (chunk_q, mk)
            for k in k_values:
                hits[k] += match[:, :k].any(dim=1).sum().item()
        return {k: hits[k] / Q for k in k_values}

    i2t_cos, t2i_cos = _i2t(_cos, "MC I2T (cos)"), _t2i(_cos, "MC T2I (cos)")
    i2t_mcdisp_align, t2i_mcdisp_align = _i2t(_mcdisp_align, "MC I2T (mcdisp_align)"), _t2i(_mcdisp_align, "MC T2I (mcdisp_align)")

    out: Dict[str, float] = {}
    for k in k_values:
        out[f"mc_recall_i2t@{k}"] = i2t_mcdisp_align[k]
        out[f"mc_recall_t2i@{k}"] = t2i_mcdisp_align[k]
        out[f"mc_recall@{k}"] = (i2t_mcdisp_align[k] + t2i_mcdisp_align[k]) / 2
        out[f"mc_cos_recall_i2t@{k}"] = i2t_cos[k]
        out[f"mc_cos_recall_t2i@{k}"] = t2i_cos[k]
        out[f"mc_cos_recall@{k}"] = (i2t_cos[k] + t2i_cos[k]) / 2
    return out


@torch.no_grad()
def compute_multicaption_allhit(
    scorers: dict,
    n_images: int,
    k_per_image: int,
    k_hit: Optional[int] = None,
    chunk_rows: int = 512,
) -> Dict[str, float]:
    """All-hit@K I2T metric: for each image query, do the top-k_hit retrieved
    captions consist EXACTLY of its own k_per_image captions?

    Strict "set exact match" companion of the any-hit I2T recall
    (:func:`compute_multicaption_recall`): an image counts as an all-hit iff
    every one of its K own captions outranks every foreign caption, i.e.
    min_k s(i, c_ik) > max_{j!=i, k} s(i, c_jk). Reported per score family:

      {fam}_allhit@{k}    fraction of images whose top-k captions are ALL
                          their own (the headline number; k == K is the
                          "retrieve the image's whole caption set" case)
      {fam}_anyhit@{k}    the standard any-of-K recall on the same matrices
                          (reference: what the strictness costs)
      {fam}_paircount@{k} mean number of own captions inside the top-k
                          (range 0..min(k, K); the graded version)

    Args:
        scorers: {family_name: fn(row_slice) -> (n_rows, N*K) score matrix,
            higher = better match}. The callable receives a ``slice`` over the
            N image rows and must return the scores of those rows against the
            FULL N*K caption gallery; it manages its own memory (chunk the
            gallery internally when needed). This keeps the metric
            family-agnostic: cosine / CSD / Gaussian overlap / ellipsoid all
            plug in as closures over their extra parameters.
        n_images: N.
        k_per_image: K (captions per image in the gallery).
        k_hit: top-k cutoff for the hit test. Defaults to K (the whole
            caption set). Must satisfy 1 <= k_hit <= N*K. all-hit is 0 by
            construction when k_hit > K (more slots than own captions);
            anyhit/paircount remain meaningful.
        chunk_rows: image-row chunk size to bound the score-matrix memory.

    Returns:
        Flat dict, ``{fam}_{allhit|anyhit|paircount}@{k_hit}`` per family.
    """
    K = k_per_image
    k = K if k_hit is None else k_hit
    if not (1 <= k <= n_images * K):
        raise ValueError(f"k_hit={k_hit} out of range [1, N*K={n_images * K}]")

    out: Dict[str, float] = {}
    for fam, scorer in scorers.items():
        all_hits = 0
        any_hits = 0
        pair_sum = 0
        for s in range(0, n_images, chunk_rows):
            e = min(s + chunk_rows, n_images)
            S = scorer(slice(s, e))                              # (n, N*K)
            top = torch.topk(S, min(k, S.size(1)), dim=1).indices   # (n, k)
            r = torch.arange(s, e, device=S.device).unsqueeze(1)
            in_range = (top >= r * K) & (top < r * K + K)         # (n, k)
            all_hits += in_range.all(dim=1).sum().item()
            any_hits += in_range.any(dim=1).sum().item()
            pair_sum += in_range.sum().item()
        out[f"{fam}_allhit@{k}"] = all_hits / n_images
        out[f"{fam}_anyhit@{k}"] = any_hits / n_images
        out[f"{fam}_paircount@{k}"] = pair_sum / n_images
    return out


@torch.no_grad()
def compute_multicaption_coverrank(
    scorers: dict,
    n_images: int,
    k_per_image: int,
    k_max: int = 100,
    chunk_rows: int = 512,
) -> Dict[str, float]:
    """Cover-rank multi-caption retrieval metric: on average, how many
    captions must the model return before an image's ENTIRE own caption set
    is covered?

    For image i with K own captions, the cover rank is

        R_i = min{ K' : top-K' retrieved captions include ALL K own captions }
            = 1-based rank of its LAST-ranked own caption,

    i.e. the smallest retrieval depth at which the whole caption set has been
    returned. The headline number is the mean of R_i over images (lower is
    better; R_i == K means the top-K IS the own set -- exactly the all-hit@K
    event, so ``{fam}_covered@K`` here equals ``{fam}_allhit@K`` from
    :func:`compute_multicaption_allhit`).

    Reported per score family:

      {fam}_coverrank_mean@{k_max}          mean R_i over images covered
                                            within k_max (the headline)
      {fam}_coverrank_median@{k_max}        median R_i over covered images
                                            (robust companion)
      {fam}_coverrank_censored_mean@{k_max} mean R_i counting images NOT
                                            covered within k_max as k_max
                                            (conservative; defined for every
                                            image, so it is comparable across
                                            models with different cover rates)
      {fam}_covered@{k_max}                 fraction of images whose whole
                                            caption set is retrieved within
                                            top-k_max

    Args:
        scorers: same interface as :func:`compute_multicaption_allhit`
            ({family: fn(row_slice) -> (n_rows, N*K) score matrix}).
        n_images: N.
        k_per_image: K (own captions per image).
        k_max: retrieval-depth cap. Must satisfy K <= k_max <= N*K. Images
            not covered within k_max are excluded from mean/median and
            counted as k_max in the censored mean.
        chunk_rows: image-row chunk size to bound the score-matrix memory.

    Returns:
        Flat dict as listed above (NaN mean/median when no image is covered).
    """
    K = k_per_image
    if not (K <= k_max <= n_images * K):
        raise ValueError(
            f"k_max={k_max} out of range [K={K}, N*K={n_images * K}]")

    out: Dict[str, float] = {}
    for fam, scorer in scorers.items():
        covered_ranks: List[torch.Tensor] = []
        censored_ranks: List[torch.Tensor] = []
        covered = 0
        for s in range(0, n_images, chunk_rows):
            e = min(s + chunk_rows, n_images)
            S = scorer(slice(s, e))                              # (n, N*K)
            m = min(k_max, S.size(1))
            top = torch.topk(S, m, dim=1).indices                # (n, m)
            r = torch.arange(s, e, device=S.device).unsqueeze(1)
            in_range = (top >= r * K) & (top < r * K + K)        # (n, m)
            # cover rank = 1-based position of the LAST own-caption hit
            # (bool * position zeroes non-hits, so amax picks the last one)
            pos = torch.arange(1, m + 1, device=S.device)
            last = (in_range * pos).amax(dim=1)                  # (n,)
            cov = in_range.sum(dim=1) == K
            covered += cov.sum().item()
            covered_ranks.append(last[cov])
            censored_ranks.append(
                torch.where(cov, last, torch.full_like(last, m)))

        R = torch.cat(covered_ranks).double() if covered_ranks else torch.tensor([])
        Rc = torch.cat(censored_ranks).double()
        out[f"{fam}_coverrank_mean@{k_max}"] = R.mean().item() if R.numel() else float("nan")
        out[f"{fam}_coverrank_median@{k_max}"] = R.median().item() if R.numel() else float("nan")
        out[f"{fam}_coverrank_censored_mean@{k_max}"] = Rc.mean().item()
        out[f"{fam}_covered@{k_max}"] = covered / n_images
    return out
