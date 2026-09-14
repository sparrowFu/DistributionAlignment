"""Coverage-penalty retrieval scoring for MCDisp-Align (inference only).

Implements the plan in paper/MCDisp-Align_覆盖评分推理实施方案.md §2:

    q(i, t) = (mu_t - mu_v)^T (Sigma_i + eps I)^-1 (mu_t - mu_v) / D
    P(i, t) = max(0, q(i, t) - m_pos)
    S(i, t) = cos(mu_v, mu_t) - lambda * P(i, t)

Coordinate contract (plan §4.1): the cosine term uses L2-NORMALIZED means;
the distance term uses RAW MLP-output means with the RAW img_logvar / img_U.
Sigma = diag(exp(logvar)) + U U^T; the epsilon stabilizer is added once on
the diagonal (A = diag(exp(logvar) + eps)) and the Woodbury identity with a
Cholesky factorization of G = I_r + U^T A^-1 U replaces any explicit D x D
inverse (plan §6.1). U=None is the legitimate r=0 case (diagonal only).

No function here may receive ground-truth pairing information (plan §6).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

EPS_DEFAULT = 1e-6
SCORER_NAME = "cosine_minus_coverage_penalty_v1"
DEFAULT_PENALTY_GRID = (0.0, 0.01, 0.03, 0.1, 0.3)
# Plan §7.4.6: tie-breaking uses a PRE-FIXED seeded permutation of images and
# captions, generated before any pairing label is read, so the natural
# "five consecutive captions per image" layout cannot bias ties. Every lambda,
# the cosine baseline and both directions share the same order.
TIE_ORDER_SEED = 20260914


def make_tie_order(n_images: int, n_captions: int, seed: int = TIE_ORDER_SEED):
    """Independent random candidate orders (plan §7.4.6). Returns
    (img_perm, cap_perm): position j in the scoring space holds original
    id perm[j]. Content-free: depends only on the counts and the seed."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randperm(n_images, generator=g),
            torch.randperm(n_captions, generator=g))


@dataclass
class NumericStats:
    """Anomaly bookkeeping required by plan §6.2 / §9."""
    negative_clipped: int = 0          # rounding-level negatives cut to 0 after FP64 recheck
    negative_fatal: int = 0            # still meaningfully negative in FP64
    nonfinite: int = 0                 # non-finite q before any repair
    fp64_rechecks: int = 0
    cholesky_failures: int = 0

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


def validate_features(img_mu: torch.Tensor, img_logvar: torch.Tensor,
                      img_U: Optional[torch.Tensor],
                      caption_mu: Optional[torch.Tensor] = None) -> None:
    """Finite/dimension checks; raise on anything the plan forbids silently
    absorbing (§5.1, §6.2)."""
    if img_mu.dim() != 2 or img_logvar.shape != img_mu.shape:
        raise ValueError(f"img_mu/img_logvar shape mismatch: {img_mu.shape} vs {img_logvar.shape}")
    for name, t in (("img_mu", img_mu), ("img_logvar", img_logvar),
                    ("img_U", img_U), ("caption_mu", caption_mu)):
        if t is not None and not torch.isfinite(t).all():
            raise ValueError(f"non-finite values in {name}")
    v = torch.exp(img_logvar)
    if not torch.isfinite(v).all() or (v <= 0).any():
        raise ValueError("exp(img_logvar) not finite positive")
    if img_U is not None:
        if img_U.dim() != 3 or img_U.shape[:2] != img_mu.shape:
            raise ValueError(f"img_U shape {img_U.shape} incompatible with img_mu {img_mu.shape}")
        if img_U.shape[2] == 0:
            raise ValueError("img_U has r=0 columns; pass None instead")


@torch.no_grad()
def prepare_image_covariance(img_logvar: torch.Tensor,
                             img_U: Optional[torch.Tensor] = None,
                             *, eps: float = EPS_DEFAULT) -> Dict:
    """Per-image-chunk Woodbury state: A = diag(exp(logvar)+eps).

    Returns {"inv_var": (B,D), "U_scaled": A^-1 U or None, "chol": cholesky(G)
    or None}. The decomposition is computed once per image chunk and reused
    for every caption chunk (plan §6.1)."""
    inv_var = 1.0 / (torch.exp(img_logvar) + eps)
    state: Dict = {"inv_var": inv_var, "U_scaled": None, "chol": None}
    if img_U is not None:
        U_scaled = img_U * inv_var.unsqueeze(-1)
        G = U_scaled.transpose(-1, -2) @ img_U
        G = G + torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
        state["U_scaled"] = U_scaled
        state["chol"] = torch.linalg.cholesky(G)
    return state


def _quad_per_dim(delta: torch.Tensor, inv_var: torch.Tensor,
                  U_scaled: Optional[torch.Tensor], chol: Optional[torch.Tensor],
                  *, fp64_recheck: bool = False, stats: Optional[NumericStats] = None
                  ) -> torch.Tensor:
    """Squared Mahalanobis / D for broadcastable (..., D) deltas (plan §6.1)."""
    D = delta.shape[-1]

    def _compute(dt, iv, us, ch):
        quad = (dt * dt * iv).sum(-1)
        if us is not None:
            b = (us * dt.unsqueeze(-1)).sum(-2)                       # (..., r)
            # G is the tiny r x r gram matrix; invert it via its Cholesky
            # and apply with an explicit einsum. (A matmul unsqueeze trick
            # here silently mis-batches per-row states: (N,r)@(N,r,r) grows
            # an extra singleton and cross-multiplies DIFFERENT images'
            # covariances.)
            Ginv = torch.cholesky_inverse(ch)
            Ginv = Ginv.expand(*b.shape[:-1], Ginv.shape[-2], Ginv.shape[-1])
            y = torch.einsum("...r,...rs->...s", b, Ginv)             # (..., r)
            quad = quad - (b * y).sum(-1)
        return quad / D

    quad = _compute(delta, inv_var, U_scaled, chol)

    bad = ~torch.isfinite(quad)
    neg = (quad < 0) & ~bad
    if bad.any():
        if stats is not None:
            stats.nonfinite += int(bad.sum())
        raise FloatingPointError(f"non-finite Mahalanobis values at {int(bad.sum())} entries")
    if neg.any():
        if stats is not None:
            stats.fp64_rechecks += 1
        recheck = _compute(delta.double(), inv_var.double(),
                           None if U_scaled is None else U_scaled.double(),
                           None if chol is None else chol.double())
        rn = recheck < 0
        tol = 1e-8 * max(1.0, float(recheck.abs().max()) if recheck.numel() else 1.0)
        truly_neg = rn & (recheck < -tol)
        if truly_neg.any():
            if stats is not None:
                stats.negative_fatal += int(truly_neg.sum())
            raise FloatingPointError(
                f"Mahalanobis distance meaningfully negative at {int(truly_neg.sum())} "
                "entries even in FP64; covariance is not positive definite")
        if stats is not None:
            stats.negative_clipped += int((quad < 0).sum())
        quad = quad.clamp_min(0.0)
    return quad


@torch.no_grad()
def coverage_score_parts(img_mu: torch.Tensor, caption_mu: torch.Tensor,
                         covariance_state: Dict, *,
                         coverage_margin: float,
                         stats: Optional[NumericStats] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(C, q, P) for one [image_chunk, caption_chunk] block.

    img_mu: (B, D) RAW image means; caption_mu: (M, D) RAW caption means.
    C = cosine of the L2-normalized means (coordinate contract §4.1);
    q = per-dim squared Mahalanobis in RAW coordinates; P = relu(q - m).
    """
    if covariance_state["inv_var"].device != img_mu.device \
            or caption_mu.device != img_mu.device:
        raise ValueError("all inputs must share one device")
    delta = caption_mu.unsqueeze(0) - img_mu.unsqueeze(1)            # (B, M, D)
    us = None if covariance_state["U_scaled"] is None \
        else covariance_state["U_scaled"].unsqueeze(1)
    chol = covariance_state["chol"]
    if chol is not None and us is not None and chol.dim() == us.dim() - 1:
        chol = chol.unsqueeze(1)                                      # (B, 1, r, r)
    q = _quad_per_dim(delta, covariance_state["inv_var"].unsqueeze(1),
                      us, chol, stats=stats)
    C = F.normalize(img_mu, dim=-1) @ F.normalize(caption_mu, dim=-1).T
    P = torch.clamp(q - coverage_margin, min=0.0)
    return C, q, P


@torch.no_grad()
def combine_coverage_score(cosine_score: torch.Tensor, penalty: torch.Tensor,
                           *, penalty_weight: float) -> torch.Tensor:
    """S = C - lambda * P. lambda == 0 returns C untouched (plan §3: avoid
    0 * Inf/NaN and skip the distance path entirely at the caller)."""
    if penalty_weight == 0:
        return cosine_score
    if penalty_weight < 0:
        raise ValueError("penalty_weight must be non-negative")
    return cosine_score - penalty_weight * penalty


def topk_stable(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k indices along dim=1 under the plan §6.3 tie rule:
    score descending, ties broken by ascending COLUMN position. Columns must
    therefore be ordered by the fixed global candidate index (chunk slices
    preserve that order)."""
    order = torch.argsort(-scores, dim=1, stable=True)
    idx = order[:, :k]
    return torch.gather(scores, 1, idx), idx


def merge_topk(running_scores: torch.Tensor, running_idx: torch.Tensor,
               block_scores: torch.Tensor, block_idx: torch.Tensor, k: int
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge a new candidate block into the running per-query top-k under the
    same (-score, global index ascending) rule (lexsort via two stable passes)."""
    s = torch.cat([running_scores, block_scores], dim=1)
    i = torch.cat([running_idx, block_idx], dim=1)
    by_idx = torch.argsort(i, dim=1, stable=True)
    s, i = torch.gather(s, 1, by_idx), torch.gather(i, 1, by_idx)
    by_score = torch.argsort(-s, dim=1, stable=True)
    s, i = torch.gather(s, 1, by_score), torch.gather(i, 1, by_score)
    return s[:, :k], i[:, :k]


@dataclass
class LambdaRun:
    """Per-lambda results of one block sweep."""
    penalty_weight: float
    img_topk_idx: torch.Tensor          # (N, k) caption indices per image
    cap_topk_idx: torch.Tensor          # (M, k) image indices per caption


def sweep_all_lambdas(img_mu, img_logvar, img_U, caption_mu, *,
                      penalty_grid, coverage_margin, topk=10,
                      image_chunk=32, caption_chunk=512,
                      eps=EPS_DEFAULT) -> Tuple[Dict[float, LambdaRun], NumericStats]:
    """One pass over all N x M blocks; C/q/P computed once per block and
    reused for every lambda (plan §6.3). No labels enter this function."""
    N, D = img_mu.shape
    M = caption_mu.shape[0]
    stats = NumericStats()
    dev = img_mu.device

    init = lambda rows: (torch.full((rows, topk), -float("inf"), device=dev),
                         torch.full((rows, topk), -1, dtype=torch.long, device=dev))
    runs = {lam: (init(N), init(M)) for lam in penalty_grid}

    for s in tqdm(range(0, N, image_chunk), desc="Scoring (image chunks)"):
        e = min(s + image_chunk, N)
        cov = prepare_image_covariance(img_logvar[s:e], img_U[s:e] if img_U is not None else None,
                                       eps=eps)
        for cs in range(0, M, caption_chunk):
            ce = min(cs + caption_chunk, M)
            C, _q, P = coverage_score_parts(
                img_mu[s:e], caption_mu[cs:ce], cov,
                coverage_margin=coverage_margin, stats=stats)
            for lam in penalty_grid:
                S = combine_coverage_score(C, P, penalty_weight=lam)
                # row direction: image queries (rows s:e), candidates = captions
                v, i = topk_stable(S, topk)
                (rs, ri), (cap_s, cap_i) = runs[lam]
                ns, ni = merge_topk(rs[s:e], ri[s:e], v, i + cs, topk)
                rs[s:e], ri[s:e] = ns, ni
                # column direction: SAME S, caption queries (rows cs:ce),
                # candidates = images
                vt, it = topk_stable(S.T, topk)
                ns2, ni2 = merge_topk(cap_s[cs:ce], cap_i[cs:ce], vt, it + s, topk)
                cap_s[cs:ce], cap_i[cs:ce] = ns2, ni2

    out = {lam: LambdaRun(lam, v[0][1], v[1][1]) for lam, v in runs.items()}
    return out, stats


def recalls_from_topk(img_topk_idx: torch.Tensor, cap_topk_idx: torch.Tensor,
                      caption_image_ids: torch.Tensor, ks=(1, 5, 10),
                      row_image_ids: Optional[torch.Tensor] = None,
                      row_caption_ids: Optional[torch.Tensor] = None
                      ) -> Dict[str, float]:
    """Six retrieval metrics from the SAME score matrix via its two sort axes
    (plan §4.2, §7.3). I2T: N queries, any-hit over the query image's own
    captions; T2I: M caption queries, hit iff its own image is in the top-k.

    Own-captions are identified via caption_image_ids (never via the
    positional "K consecutive captions" layout, which the pre-fixed tie
    order deliberately destroys). row_image_ids maps each score row to its
    image id (identity by default)."""
    N, _ = img_topk_idx.shape
    if row_image_ids is None:
        row_image_ids = torch.arange(N, device=img_topk_idx.device)
    if row_caption_ids is None:
        row_caption_ids = torch.arange(cap_topk_idx.shape[0],
                                        device=cap_topk_idx.device)
    # cap_topk_idx rows are tie-order rows: the caption behind row m is
    # row_caption_ids[m], so its own image is caption_image_ids[row_caption_ids[m]].
    hit_matrix = cap_topk_idx == caption_image_ids[row_caption_ids].unsqueeze(1)
    out: Dict[str, float] = {}
    for k in ks:
        sub = img_topk_idx[:, :k]
        # own captions of the image behind each row
        in_own = (caption_image_ids[sub] == row_image_ids.unsqueeze(1)).any(dim=1)
        out[f"mc_i2t@{k}"] = in_own.float().mean().item()
        out[f"mc_t2i@{k}"] = hit_matrix[:, :k].any(dim=1).float().mean().item()
    out["mr"] = sum(out[f"mc_{d}@{k}"] for d in ("i2t", "t2i") for k in ks) / (2 * len(ks))
    return out


def select_best_lambda(dev_results):
    """Plan §7.2: max dev mR; ties -> SMALLER penalty_weight. Input is a list
    of dicts with keys penalty_weight / mr (unrounded values)."""
    return sorted(dev_results, key=lambda d: (-d["mr"], d["penalty_weight"]))[0]["penalty_weight"]


def rank_all_lambdas(img_mu, img_logvar, img_U, caption_mu, *,
                     penalty_grid, coverage_margin, topk=10,
                     image_chunk=32, caption_chunk=512, eps=EPS_DEFAULT,
                     tie_order_seed=TIE_ORDER_SEED):
    """Label-free scoring + ranking under the pre-fixed tie order (plan §7.4).

    Features are permuted into the seeded tie order, swept, and the top-k
    indices are mapped back to ORIGINAL extraction indices. No pairing
    information enters or leaves this function: callers only get candidate
    ids, which they may interpret with labels afterwards.
    """
    N, _ = img_mu.shape
    M = caption_mu.shape[0]
    img_perm, cap_perm = make_tie_order(N, M, tie_order_seed)
    runs, stats = sweep_all_lambdas(
        img_mu[img_perm], img_logvar[img_perm],
        None if img_U is None else img_U[img_perm],
        caption_mu[cap_perm],
        penalty_grid=penalty_grid, coverage_margin=coverage_margin, topk=topk,
        image_chunk=image_chunk, caption_chunk=caption_chunk, eps=eps)
    out = {lam: LambdaRun(lam, cap_perm[r.img_topk_idx.cpu()],
                          img_perm[r.cap_topk_idx.cpu()])
           for lam, r in runs.items()}
    # rows of img_topk_idx are tie-order rows: original image id img_perm[i];
    # rows of cap_topk_idx likewise hold original caption id cap_perm[m].
    return out, stats, img_perm, cap_perm
