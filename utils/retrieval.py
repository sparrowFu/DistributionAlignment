"""Shared image-text Recall@K computation. Supports both retrieval directions: I2T (query=image, gallery=text) and T2I (query=text, gallery=image)."""

from typing import Dict, List, Optional, Tuple

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

    Used by the CLIP baseline evaluations (single-feature models without
    distribution parameters).

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

    THE evaluation protocol for MCDisp_Align: checkpoint selection during
    training (``select_by="recall"`` / ``"mr"``) and test-time evaluation both
    use it. This is the canonical MS-COCO/Flickr30k retrieval protocol
    (evaluating on the merged text mean would collapse the one-to-many
    structure at eval time and is not comparable to published baselines):

      I2T: query = N image means, gallery = N*K per-caption means. Image i has K
           positives -- its own captions at flattened indices [i*K, i*K+K). A hit
           is ANY of them landing in the top-K (any-hit).
      T2I: query = N*K per-caption means, gallery = N image means. Each caption's
           ONLY positive is its own image (single-positive per query).

    Two score families are computed:

      cosine : the PLAIN cosine of the means (no variance enters the
               retrieval score). This is the primary/headline metric.
      CSD    : the ProLIP-style Contraction-Subspace distance used by this
               repo's ProLIP baselines (utils/retrieval_metrics.py):
               score = cos(mu_q, mu_g) - 0.5 * sum_d sigma_g,d^2, i.e. the
               GALLERY side is discounted by its uncertainty. NOTE the two
               families use different uncertainty semantics: ProLIP's sigma
               means "unreliability" (small, learned implicitly), whereas our
               image variance means "semantic range" (dispersion-supervised,
               ~caption-spread scale) -- see the CSD probe discussion before
               using it as a headline number.

    ``tau`` is accepted for API compatibility and unused. Means are
    L2-normalized internally. Uses torch.topk (not a full argsort) since the
    N*K gallery is large and only the top max(k_values) are needed.

    Returns keys (per k in k_values)::

        mc_recall_i2t@{k}, mc_recall_t2i@{k}, mc_recall@{k} (mean of the two)
        mc_csd_recall_i2t@{k}, mc_csd_recall_t2i@{k}, mc_csd_recall@{k}
    """
    N, K, _ = text_mus.shape
    cap_mu = text_mus.reshape(N * K, -1)              # (N*K, D)
    cap_lv = text_logvars.reshape(N * K, -1)          # (N*K, D)

    img_mu_n = F.normalize(img_mu, dim=-1)            # (N, D)
    cap_mu_n = F.normalize(cap_mu, dim=-1)            # (N*K, D)
    img_unc = torch.exp(img_logvar).sum(dim=-1)       # (N,) sum sigma_v^2
    cap_unc = torch.exp(cap_lv).sum(dim=-1)           # (N*K,) sum sigma_k^2

    maxk = max(k_values)

    def _i2t(use_csd: bool):
        """N image queries vs N*K caption gallery; any-of-K-own-caption hit."""
        hits = {k: 0 for k in k_values}
        for start in tqdm(range(0, N, chunk_size),
                          desc="MC I2T (csd)" if use_csd else "MC I2T", leave=False):
            end = min(start + chunk_size, N)
            sim = img_mu_n[start:end] @ cap_mu_n.T                    # (chunk, N*K)
            if use_csd:
                sim = sim - 0.5 * cap_unc.unsqueeze(0)                # gallery-side discount
            mk = min(maxk, sim.size(1))
            top = torch.topk(sim, mk, dim=1).indices                   # (chunk, mk)
            rows = torch.arange(start, end, device=sim.device).unsqueeze(1)
            in_range = (top >= rows * K) & (top < rows * K + K)        # (chunk, mk)
            for k in k_values:
                hits[k] += in_range[:, :k].any(dim=1).sum().item()     # :k clamps to mk
        return {k: hits[k] / N for k in k_values}

    def _t2i(use_csd: bool):
        """N*K caption queries vs N image gallery; single-positive per query."""
        Q = N * K
        hits = {k: 0 for k in k_values}
        for start in tqdm(range(0, Q, chunk_size),
                          desc="MC T2I (csd)" if use_csd else "MC T2I", leave=False):
            end = min(start + chunk_size, Q)
            sim = cap_mu_n[start:end] @ img_mu_n.T                    # (chunk_q, N)
            if use_csd:
                sim = sim - 0.5 * img_unc.unsqueeze(0)                # gallery-side discount
            mk = min(maxk, sim.size(1))
            top = torch.topk(sim, mk, dim=1).indices                   # (chunk_q, mk)
            q_idx = torch.arange(start, end, device=sim.device)
            gt_img = (q_idx // K).unsqueeze(1)                         # (chunk_q, 1)
            match = top == gt_img                                      # (chunk_q, mk)
            for k in k_values:
                hits[k] += match[:, :k].any(dim=1).sum().item()
        return {k: hits[k] / Q for k in k_values}

    i2t, t2i = _i2t(False), _t2i(False)
    csd_i2t, csd_t2i = _i2t(True), _t2i(True)

    out: Dict[str, float] = {}
    for k in k_values:
        out[f"mc_recall_i2t@{k}"] = i2t[k]
        out[f"mc_recall_t2i@{k}"] = t2i[k]
        out[f"mc_recall@{k}"] = (i2t[k] + t2i[k]) / 2
        out[f"mc_csd_recall_i2t@{k}"] = csd_i2t[k]
        out[f"mc_csd_recall_t2i@{k}"] = csd_t2i[k]
        out[f"mc_csd_recall@{k}"] = (csd_i2t[k] + csd_t2i[k]) / 2
    return out


@torch.no_grad()
def compute_multicaption_recall_plain(
    img_features: torch.Tensor,
    text_features: torch.Tensor,
    k_values: List[int],
    chunk_size: int = 1000,
) -> Dict[str, float]:
    """Plain-cosine multi-caption bidirectional Recall@K (N images vs N*K
    captions) for models WITHOUT distribution parameters (CLIP baseline,
    ProLIP means) -- the unified protocol shared by ALL methods:

      I2T: query = N image features, gallery = N*K caption features.
           Image i has K positives (its own captions, flattened indices
           [i*K, i*K+K)); a hit is ANY of them in the top-K (any-hit).
      T2I: query = N*K caption features, gallery = N image features.
           Each caption's ONLY positive is its own image.

    Same semantics as the cosine family of compute_multicaption_recall;
    features are L2-normalized internally. Returns keys (per k):
    mc_recall_i2t@{k}, mc_recall_t2i@{k}, mc_recall@{k} (mean of the two).
    """
    N, K, _ = text_features.shape
    img_n = F.normalize(img_features, dim=-1)                      # (N, D)
    cap_n = F.normalize(text_features.reshape(N * K, -1), dim=-1)  # (N*K, D)
    maxk = max(k_values)

    hits_i2t = {k: 0 for k in k_values}
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        sim = img_n[start:end] @ cap_n.T                          # (chunk, N*K)
        mk = min(maxk, sim.size(1))
        top = torch.topk(sim, mk, dim=1).indices
        rows = torch.arange(start, end, device=sim.device).unsqueeze(1)
        in_range = (top >= rows * K) & (top < rows * K + K)
        for k in k_values:
            hits_i2t[k] += in_range[:, :k].any(dim=1).sum().item()

    Q = N * K
    hits_t2i = {k: 0 for k in k_values}
    for start in range(0, Q, chunk_size):
        end = min(start + chunk_size, Q)
        sim = cap_n[start:end] @ img_n.T                          # (chunk, N)
        mk = min(maxk, sim.size(1))
        top = torch.topk(sim, mk, dim=1).indices
        q_idx = torch.arange(start, end, device=sim.device)
        gt_img = (q_idx // K).unsqueeze(1)
        match = top == gt_img
        for k in k_values:
            hits_t2i[k] += match[:, :k].any(dim=1).sum().item()

    out: Dict[str, float] = {}
    for k in k_values:
        i2t = hits_i2t[k] / N
        t2i = hits_t2i[k] / Q
        out[f"mc_recall_i2t@{k}"] = i2t
        out[f"mc_recall_t2i@{k}"] = t2i
        out[f"mc_recall@{k}"] = (i2t + t2i) / 2
    return out


# ---------------------------------------------------------------------------
# Distribution-aware multi-caption retrieval (MCDisp_Align-native scores)
# ---------------------------------------------------------------------------
def _topk_hits_both_dirs(
    S: torch.Tensor, k_values: List[int], K: int
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Both-direction any-hit recalls from a full (N, N*K) score matrix.

    I2T: image i (row) has K positives at flattened columns [i*K, i*K+K).
    T2I: caption j (column) has single positive image j//K (rank rows).
    """
    N, Q = S.shape
    maxk = max(k_values)

    top_rows = torch.topk(S, min(maxk, Q), dim=1).indices          # (N, mk)
    rows = torch.arange(N, device=S.device).unsqueeze(1)
    in_range = (top_rows >= rows * K) & (top_rows < rows * K + K)

    top_cols = torch.topk(S.T, min(maxk, N), dim=1).indices        # (Q, mk)
    q_idx = torch.arange(Q, device=S.device)
    gt_img = (q_idx // K).unsqueeze(1)
    match = top_cols == gt_img

    i2t = {k: in_range[:, :k].any(dim=1).sum().item() / N for k in k_values}
    t2i = {k: match[:, :k].any(dim=1).sum().item() / Q for k in k_values}
    return i2t, t2i


def _overlap_score_matrix(
    img_mu: torch.Tensor,          # (N, D)
    img_diag: torch.Tensor,        # (N, D)   d_v > 0
    img_U: Optional[torch.Tensor], # (N, D, r) or None
    cap_mu: torch.Tensor,          # (Q, D)
    cap_d: torch.Tensor,           # (Q, D)   d_t > 0
    pair_budget: int,
) -> torch.Tensor:
    """Full (N, Q) Gaussian log-overlap matrix, double-chunked to pair_budget.

    Reuses losses/gaussian_overlap.gaussian_overlap_scores (Woodbury form,
    per-dim normalized) so the retrieval score is exactly the training-time
    L_match compatibility score psi_{i,j,k}.
    """
    from losses.gaussian_overlap import gaussian_overlap_scores

    N, Q, D = img_mu.shape[0], cap_mu.shape[0], cap_mu.shape[1]
    cap_chunk = min(Q, max(1, 8192))
    row_chunk = max(1, int(pair_budget // (cap_chunk * D)))

    S = torch.empty((N, Q), device=img_mu.device, dtype=img_mu.dtype)
    for c0 in range(0, Q, cap_chunk):
        c1 = min(c0 + cap_chunk, Q)
        for r0 in range(0, N, row_chunk):
            r1 = min(r0 + row_chunk, N)
            U = img_U[r0:r1] if img_U is not None else None
            S[r0:r1, c0:c1] = gaussian_overlap_scores(
                img_mu[r0:r1], img_diag[r0:r1], U,
                cap_mu[c0:c1], cap_d[c0:c1])
    return S


def _ellipsoid_score_matrix(
    img_mu: torch.Tensor,          # (N, D)
    img_diag: torch.Tensor,        # (N, D)
    img_U: Optional[torch.Tensor], # (N, D, r) or None
    cap_mu: torch.Tensor,          # (Q, D)
    img_chunk: int = 64,
) -> torch.Tensor:
    """(N, Q) matrix of NEGATIVE Mahalanobis depths: S = -m_{i,j} with
    m = (mu_j - mu_i)^T (Diag(d_i) + U_i U_i^T)^-1 (mu_j - mu_i).

    Same Woodbury form as the L_cov hinge (higher score = caption mean sits
    deeper inside the image confidence ellipsoid). Chunked matmul/triangular
    solves; no (chunk, Q, D) intermediate is materialized.
    """
    N, D = img_mu.shape
    Q = cap_mu.shape[0]
    inv_d = 1.0 / img_diag                                        # (N, D)
    cap_sq = cap_mu ** 2                                          # (Q, D)
    A = (img_mu ** 2 * inv_d).sum(-1)                             # (N,)

    S = torch.empty((N, Q), device=img_mu.device, dtype=img_mu.dtype)
    eye = torch.eye(img_U.shape[-1], device=img_mu.device, dtype=img_mu.dtype) \
        if img_U is not None else None

    for s in range(0, N, img_chunk):
        e = min(s + img_chunk, N)
        invd_c = inv_d[s:e]                                       # (C, D)
        # diagonal part: |mu_i - mu_j|^2_{diag(1/d_i)} expanded into matmuls
        X = (img_mu[s:e] * invd_c) @ cap_mu.T                     # (C, Q)
        M2 = invd_c @ cap_sq.T                                     # (C, Q)
        m = A[s:e].unsqueeze(1) + M2 - 2.0 * X
        if img_U is not None:
            Uc = img_U[s:e]                                       # (C, D, r)
            W = Uc * invd_c.unsqueeze(-1)                         # (C, D, r)
            b = torch.einsum("cdr,cd->cr", Uc, img_mu[s:e] * invd_c)
            capW = torch.einsum("qd,cdr->cqr", cap_mu, W)         # (C, Q, r)
            v = b.unsqueeze(1) - capW                             # (C, Q, r)
            Mc = torch.einsum("cdr,cd,cds->crs", Uc, invd_c, Uc) + eye
            Lc = torch.linalg.cholesky(Mc)                        # (C, r, r)
            # y = L^-1 v per (image, caption); M is SPD by the +I ridge
            y = torch.linalg.solve_triangular(
                Lc.unsqueeze(1), v.unsqueeze(-1), upper=False)    # (C, Q, r, 1)
            m = m - (y ** 2).sum(dim=-2).squeeze(-1)
        S[s:e] = -m
    return S


@torch.no_grad()
def compute_multicaption_recall_dist(
    img_mu: torch.Tensor,           # (N, D)
    img_logvar: torch.Tensor,       # (N, D)
    img_U: Optional[torch.Tensor],  # (N, D, r) or None (diagonal-only model)
    text_mus: torch.Tensor,         # (N, K, D)
    text_logvars: torch.Tensor,     # (N, K, D)
    k_values: List[int],
    pair_budget: int = 64_000_000,
    img_chunk: int = 64,
) -> Dict[str, float]:
    """Distribution-aware multi-caption bidirectional Recall@K -- the two
    MCDisp_Align-native score families in which the learned distribution
    parameters DO enter the retrieval score (unlike the plain-cosine family):

      overlap : Gaussian log-overlap between the image Gaussian
          N(mu_v, Diag(d_v) + U_v U_v^T) and each caption Gaussian
          N(mu_k, Diag(d_k)) -- the exact L_match compatibility score
          psi_{i,j,k}. Means AND (co)variances of both modalities score.
      ellip   : negative Mahalanobis depth of each caption mean inside the
          image confidence ellipsoid (the L_cov geometry as a ranker).
          Caption variance does not enter; the full image covariance does.

    Protocol identical to the cosine family (N vs N*K, any-hit I2T,
    per-caption T2I), so the three families are directly comparable. Both
    directions read the SAME (N, N*K) score matrix.

    Returns keys (per k): mc_overlap_recall_i2t@{k}, mc_overlap_recall_t2i@{k},
    mc_overlap_recall@{k}, mc_ellip_recall_i2t@{k}, mc_ellip_recall_t2i@{k},
    mc_ellip_recall@{k}.
    """
    N, K, D = text_mus.shape
    img_diag = torch.exp(img_logvar)                              # (N, D)
    cap_mu = text_mus.reshape(N * K, D)                           # (Q, D)
    cap_d = torch.exp(text_logvars.reshape(N * K, D))             # (Q, D)

    S_ov = _overlap_score_matrix(img_mu, img_diag, img_U, cap_mu, cap_d,
                                 pair_budget)
    S_el = _ellipsoid_score_matrix(img_mu, img_diag, img_U, cap_mu,
                                   img_chunk=img_chunk)

    out: Dict[str, float] = {}
    for S, fam in ((S_ov, "overlap"), (S_el, "ellip")):
        i2t, t2i = _topk_hits_both_dirs(S, k_values, K)
        for k in k_values:
            out[f"mc_{fam}_recall_i2t@{k}"] = i2t[k]
            out[f"mc_{fam}_recall_t2i@{k}"] = t2i[k]
            out[f"mc_{fam}_recall@{k}"] = (i2t[k] + t2i[k]) / 2
    return out
