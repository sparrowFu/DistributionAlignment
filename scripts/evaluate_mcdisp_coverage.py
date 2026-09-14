#!/usr/bin/env python
"""Coverage-penalty retrieval evaluation for MCDisp-Align.

Follows the repo's evaluate_* architecture: same CLI surface, the shared
eval dataloader (identical test protocol to the other models), standard
metric-group output (families printed via print_recall_groups and appended
via append_eval_results).

Families per run (plan §7.3 naming):
    mc_cos_recall        cosine baseline on the N vs N*K multi-caption protocol
    mc_coverage_recall   S = cos(mu_v, mu_t) - lambda * max(0, q - m_pos),
                         one family PER requested --penalty-weight (the label
                         carries lambda); lambda = 0 IS the cosine baseline
                         and is not duplicated.

Scoring math lives in utils/mcdisp_coverage_score.py (raw-coordinate
Mahalanobis via Woodbury, cosine on normalized means; no training labels).
lambda selection (plan §7.2): sweep --penalty-weight on a DEV evaluation
(--dev-seed <training seed> evaluates on the reconstructed random_split
validation subset instead of the test protocol) and keep the best-mR value;
ties -> smaller lambda.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.eval_common import build_eval_dataloader
from utils.eval_results import append_eval_results, groups_to_flat, print_recall_groups
from utils.logger import get_logger, log_exception
from utils.retrieval import compute_multicaption_recall
from utils.seed import set_seed
from utils.mcdisp_coverage_score import (
    DEFAULT_PENALTY_GRID, EPS_DEFAULT, TIE_ORDER_SEED, prepare_image_covariance,
    rank_all_lambdas, select_best_lambda, recalls_from_topk, _quad_per_dim,
)

# Exclude faulty CPU cores before DataLoader workers and torch threads.
from utils.cpu_affinity import apply_cpu_affinity
apply_cpu_affinity()

logger = get_logger("eval_mcdisp_coverage", config.LOG_DIR / "eval_mcdisp_coverage.log")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Explicit checkpoint path; otherwise resolved from "
                        "the actual layout (see resolve_checkpoint_path)")
    p.add_argument("--checkpoint-name", type=str, default="mcdisp_align",
                   choices=["mcdisp_align", "mcdisp_align_kl"],
                   help="Name used for auto checkpoint resolution")
    p.add_argument("--ckpt-dir", type=str, default=None,
                   help="Directory to resolve {checkpoint-name}_{dataset}"
                        "_best.pt from (e.g. checkpoints/seed42)")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed used with the default layout: tries "
                        "checkpoints/seed{N}/ then unfrozen_seed{N}/")
    p.add_argument("--dataset", type=str, default="coco",
                   choices=["coco", "flickr"],
                   help="Dataset to evaluate on (coco=MSCOCO, flickr=flickr30k)")
    p.add_argument("--captions-path", type=str, default=None)
    p.add_argument("--images-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    p.add_argument("--num-samples", type=int, default=5000,
                   help="coco subset size; flickr uses its full test split")
    p.add_argument("--recall-at-k", type=int, nargs="+", default=config.RECALL_AT_K)
    p.add_argument("--penalty-weight", type=float, nargs="+",
                   default=list(DEFAULT_PENALTY_GRID),
                   help="lambda grid for the coverage score (plan §3)")
    p.add_argument("--coverage-margin", type=float, default=config.MCDISP_ALIGN_M_POS,
                   help="m_pos margin (training default)")
    p.add_argument("--eps", type=float, default=EPS_DEFAULT)
    p.add_argument("--dev-seed", type=int, default=None,
                   help="If set, evaluate on the training run's random_split "
                        "validation subset rebuilt with this seed (dev "
                        "protocol for lambda selection) instead of the test "
                        "protocol")
    p.add_argument("--gate-diagnostic", action="store_true",
                   help="Run the plan's front-section applicability diagnostic "
                        "(dev; uses labels only to pick candidate pairs) and exit")
    p.add_argument("--gate-top-neg", type=int, default=20,
                   help="Max wrong candidates per query in the gate diagnostic")
    p.add_argument("--tie-order-seed", type=int, default=TIE_ORDER_SEED,
                   help="Pre-fixed seed for the label-free candidate tie "
                        "order (plan §7.4.6)")
    p.add_argument("--export-rankings", type=str, default=None,
                   help="Export label-free top-k rankings to this JSON path "
                        "(plan §7.4.4: possible without any pairing labels)")
    p.add_argument("--image-chunk-size", type=int, default=32)
    p.add_argument("--caption-chunk-size", type=int, default=512)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def extract_features(model, dataloader, device, num_samples=None):
    """Raw MLP outputs: img mu/logvar/U + per-caption means (plan §5.1).

    Mirrors the mcdisp branch of eval_allhit.extract_features.
    """
    model.eval()
    acc = {"img_mu": [], "img_logvar": [], "img_U": [], "text_mus": []}
    K = None
    from tqdm import tqdm
    for batch in tqdm(dataloader, desc="Extracting"):
        if batch is None:
            continue
        pil_images = batch["image"]
        caption_lists = batch["captions"]
        B = len(pil_images)
        K = len(caption_lists[0])
        all_captions = [c for cl in caption_lists for c in cl]
        pixel_values = model.process_images(pil_images).to(device)
        ti = model.process_text(all_captions)
        out = model(pixel_values,
                    ti["input_ids"].to(device).view(B, K, -1),
                    ti["attention_mask"].to(device).view(B, K, -1))
        acc["img_mu"].append(out["img_mu"].float().cpu())
        acc["img_logvar"].append(out["img_logvar"].float().cpu())
        acc["text_mus"].append(out["text_mus"].float().cpu())
        if out.get("img_U") is not None:
            acc["img_U"].append(out["img_U"].float().cpu())
        if num_samples and sum(t.shape[0] for t in acc["img_mu"]) >= num_samples:
            break

    feats = {"img_mu": torch.cat(acc["img_mu"]),
             "img_logvar": torch.cat(acc["img_logvar"]),
             "text_mus": torch.cat(acc["text_mus"])}
    feats["img_U"] = torch.cat(acc["img_U"]) if acc["img_U"] else None
    N = feats["img_mu"].shape[0]
    feats["text_mus"] = feats["text_mus"].view(N, K, -1)
    logger.info(f"Features: images {feats['img_mu'].shape}, "
                f"captions {tuple(feats['text_mus'].shape)}, "
                f"r={0 if feats['img_U'] is None else feats['img_U'].shape[-1]}")
    return feats


def build_dev_loader(args):
    """DEV protocol: rebuild the training run's random_split validation
    subset with its seed (plan §7.1)."""
    from utils.dataset_factory import build_train_dataset
    from data.caption_dataset import filter_none_collate
    full = build_train_dataset(dataset=args.dataset)
    val_size = int(len(full) * 0.1)
    g = torch.Generator().manual_seed(args.dev_seed)
    _tr, val = torch.utils.data.random_split(full, [len(full) - val_size, val_size],
                                             generator=g)
    return DataLoader(val, batch_size=args.batch_size, shuffle=False,
                      num_workers=config.NUM_WORKERS, collate_fn=filter_none_collate)


def build_recall_groups(recalls_by_lambda, ks):
    """Standard metric groups: one mc_cos_recall family + one
    mc_coverage_recall family per positive lambda."""
    def _mean(r, k):
        return (r[f"mc_i2t@{k}"] + r[f"mc_t2i@{k}"]) / 2

    groups = [{
        "family": "mc_cos_recall",
        "label": "Multi-caption Recall@K (cosine baseline)",
        "per_k": {
            k: {"i2t": recalls_by_lambda[0.0][f"mc_i2t@{k}"],
                "t2i": recalls_by_lambda[0.0][f"mc_t2i@{k}"],
                "mean": _mean(recalls_by_lambda[0.0], k)}
            for k in ks},
    }]
    for lam in sorted(l for l in recalls_by_lambda if l > 0):
        r = recalls_by_lambda[lam]
        groups.append({
            "family": "mc_coverage_recall",
            "label": f"Multi-caption Recall@K (coverage score, lambda={lam})",
            "per_k": {
                k: {"i2t": r[f"mc_i2t@{k}"], "t2i": r[f"mc_t2i@{k}"],
                    "mean": _mean(r, k)}
                for k in ks},
        })
    return groups


def log_diagnostics(feats, coverage_margin):
    """Plan §9 essentials, logged (not a custom schema): correct-pair
    coverage ratio and q summaries for correct vs cosine-confusable
    captions. Labels are used only here, after scoring."""
    import torch.nn.functional as F
    img_mu, cap_mus = feats["img_mu"], feats["text_mus"]
    N, K, D = cap_mus.shape
    cap_mu = cap_mus.reshape(N * K, D)
    cids = torch.arange(N).repeat_interleave(K)
    state = prepare_image_covariance(feats["img_logvar"], feats["img_U"])
    q_true = _quad_per_dim(cap_mu - img_mu[cids], state["inv_var"][cids],
                           None if state["U_scaled"] is None
                           else state["U_scaled"][cids], state["chol"][cids])
    C = F.normalize(img_mu, dim=-1) @ F.normalize(cap_mu, dim=-1).T
    wrong_q = []
    for i in range(N):
        scores = C[i].clone()
        scores[i * K:(i + 1) * K] = -2.0
        cand = torch.argsort(scores, descending=True, stable=True)[:50]
        wrong_q.append(_quad_per_dim(
            cap_mu[cand] - img_mu[i], state["inv_var"][i].unsqueeze(0),
            None if state["U_scaled"] is None
            else state["U_scaled"][i].unsqueeze(0), state["chol"][i].unsqueeze(0)))
    wrong_q = torch.cat(wrong_q)

    def _s(t):
        return (f"mean={t.mean():.3f} median={t.median():.3f} "
                f"p90={t.quantile(0.9):.3f} p99={t.quantile(0.99):.3f}")

    logger.info(f"[diag] correct-pair coverage ratio (q<=m): "
                f"{(q_true <= coverage_margin).float().mean():.4f}")
    logger.info(f"[diag] correct q: {_s(q_true)}")
    logger.info(f"[diag] confusable (cosine top-50 wrong) q: {_s(wrong_q)}")


def resolve_checkpoint_path(args):
    """Resolve the checkpoint from the ACTUAL weights layout on disk.

    Priority: explicit --checkpoint > --ckpt-dir > seed subdirectories
    (checkpoints/seed{N}/, then unfrozen_seed{N}/ when --seed is given) >
    the legacy flat checkpoints/{name}_{dataset}_best.pt. If nothing matches
    directly, a unique match anywhere under config.CHECKPOINT_DIR is used;
    multiple matches raise with the full candidate list.
    """
    name = f"{args.checkpoint_name}_{args.dataset}_best.pt"
    candidates = []
    if args.checkpoint:
        candidates.append(Path(args.checkpoint))
    if args.ckpt_dir:
        candidates.append(Path(args.ckpt_dir) / name)
    if args.seed is not None:
        candidates.append(Path(config.CHECKPOINT_DIR) / f"seed{args.seed}" / name)
        candidates.append(Path(config.CHECKPOINT_DIR) / f"unfrozen_seed{args.seed}" / name)
    candidates.append(Path(config.CHECKPOINT_DIR) / name)
    for c in candidates:
        if c.exists():
            return c
    matches = sorted(Path(config.CHECKPOINT_DIR).glob(f"*/{name}"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"checkpoint {name!r} not found. Tried: "
        f"{[str(c) for c in candidates]}. "
        f"Matches under {config.CHECKPOINT_DIR}: {[str(m) for m in matches]}")




def confusable_gate_stats(feats, coverage_margin, penalty_grid, *,
                          top_neg=20, device="cpu", chunk=512):
    """Plan (front section) applicability diagnostic -- DEV ONLY.

    Locates cosine-misranked queries, takes the best correct candidate p and
    the wrong candidates n ranked above it, and asks whether the penalty
    difference can flip the pair: needs lambda * (P_n - P_p) > C_n - C_p > 0.
    Reports the fraction of pairs with a positive (ReLU) penalty difference,
    the lambda each such pair would require, and the same on UNTRUNCATED q
    (to separate "ReLU discarded the signal" from "geometry has none").
    Labels are used HERE ONLY to pick the candidate pairs (plan §7.4 last
    paragraph); the scoring itself is the same pure functions.
    """
    import torch.nn.functional as F
    img_mu, cap_mus = feats["img_mu"].to(device), feats["text_mus"].to(device)
    N, K, D = cap_mus.shape
    cap_mu = cap_mus.reshape(N * K, D)
    cids = torch.arange(N, device=device).repeat_interleave(K)
    state = prepare_image_covariance(feats["img_logvar"].to(device),
                                     None if feats["img_U"] is None
                                     else feats["img_U"].to(device))
    C = F.normalize(img_mu, dim=-1) @ F.normalize(cap_mu, dim=-1).T   # (N, M)

    def _q_pairs(img_idx, cap_idx):
        delta = cap_mu[cap_idx] - img_mu[img_idx]
        return _quad_per_dim(delta, state["inv_var"][img_idx],
                             None if state["U_scaled"] is None
                             else state["U_scaled"][img_idx], state["chol"][img_idx])

    dC_list, dP_list, dq_list = [], [], []

    def _collect(pairs):  # pairs: (img_ids, cap_neg, cap_pos)
        ii, cn, cp = pairs
        if len(ii) == 0:
            return
        qn, qp = _q_pairs(ii, cn), _q_pairs(ii, cp)
        dC_list.append((C[ii, cn] - C[ii, cp]).cpu())
        dP_list.append((torch.clamp(qn - coverage_margin, min=0)
                        - torch.clamp(qp - coverage_margin, min=0)).cpu())
        dq_list.append((qn - qp).cpu())

    # I2T: per image, negatives ranked above the BEST own caption
    for s0 in range(0, N, chunk):
        e0 = min(s0 + chunk, N)
        block = C[s0:e0]                                        # (B, M)
        own = cids.unsqueeze(0) == torch.arange(s0, e0, device=device).unsqueeze(1)
        best_own = torch.where(own, block, torch.tensor(-2.0, device=device))             .argmax(dim=1)                                      # (B,)
        thresh = block.gather(1, best_own.unsqueeze(1))         # (B, 1)
        above = block > thresh                                  # negatives ahead
        # keep the strongest top_neg negatives per row
        masked = torch.where(above, block, torch.tensor(-2.0, device=device))
        neg_idx = masked.topk(min(top_neg, block.shape[1]), dim=1).indices
        rows, cols = torch.nonzero(above.gather(1, neg_idx), as_tuple=True)
        gi = torch.arange(s0, e0, device=device)[rows]
        _collect((gi, neg_idx[rows, cols], best_own[rows]))
    # T2I: per caption, negative images ranked above the own image
    Ct = C.T                                                  # (M, N)
    for s0 in range(0, Ct.shape[0], chunk):
        e0 = min(s0 + chunk, Ct.shape[0])
        block = Ct[s0:e0]                                      # (B, N)
        own_col = cids[s0:e0].clamp(0, N - 1).unsqueeze(1)     # own image id
        thresh = block.gather(1, own_col)                      # (B, 1)
        above = block > thresh
        masked = torch.where(above, block, torch.tensor(-2.0, device=device))
        neg_idx = masked.topk(min(top_neg, N), dim=1).indices
        rows, cols = torch.nonzero(above.gather(1, neg_idx), as_tuple=True)
        ti = torch.arange(s0, e0, device=device)[rows]
        _collect((neg_idx[rows, cols], cids[ti], cids[ti]))

    dC = torch.cat(dC_list) if dC_list else torch.tensor([])
    dP = torch.cat(dP_list) if dP_list else torch.tensor([])
    dq = torch.cat(dq_list) if dq_list else torch.tensor([])
    n = max(int(dC.numel()), 1)
    pos = dP > 0
    need_lam = (dC[pos] / dP[pos]) if pos.any() else torch.tensor([])
    return {
        "n_confusable_pairs": int(dC.numel()),
        "pairs_with_positive_relu_penalty": int(pos.sum()),
        "frac_positive_relu_penalty": float(pos.float().mean()) if dP.numel() else 0.0,
        "raw_dq_positive_frac": float((dq > 0).float().mean()) if dq.numel() else 0.0,
        "raw_dq_mean": float(dq.mean()) if dq.numel() else 0.0,
        "lambda_required": {"min": float(need_lam.min()) if need_lam.numel() else None,
                            "median": float(need_lam.median()) if need_lam.numel() else None,
                            "p90": float(need_lam.quantile(0.9)) if need_lam.numel() else None},
        "feasible_frac_per_lambda": {str(lam): float((dP > 0).logical_and(
            lam * dP > dC).float().sum() / n) for lam in penalty_grid},
    }

def main():
    args = parse_args()
    set_seed(config.SEED)

    from models.mcdisp_align_model import MCDispAlignModel
    checkpoint_path = resolve_checkpoint_path(args)
    logger.info(f"Loading model from {checkpoint_path}")
    model = MCDispAlignModel()
    model.load(str(checkpoint_path))
    model = model.to(args.device)

    if args.dev_seed is not None:
        dataloader, num_eval_samples = build_dev_loader(args), None
        logger.info(f"DEV protocol: random_split val subset (seed={args.dev_seed})")
    else:
        dataloader, num_eval_samples = build_eval_dataloader(
            args.dataset, batch_size=args.batch_size,
            num_workers=config.NUM_WORKERS, num_samples=args.num_samples,
            captions_path=args.captions_path, images_dir=args.images_dir)
    logger.info(f"Dataset loaded ({args.dataset}): {num_eval_samples or 'dev'} samples")

    feats = extract_features(model, dataloader, args.device, args.num_samples)

    grid = sorted({0.0, *[float(l) for l in args.penalty_weight]})

    if args.gate_diagnostic:
        # Plan front section: run BEFORE any full scoring; if the confusable
        # pairs sit in the zero-penalty zone, stop here and keep cosine.
        gate = confusable_gate_stats(feats, args.coverage_margin, grid,
                                     top_neg=args.gate_top_neg,
                                     device=args.device)
        logger.info("=" * 64)
        logger.info(f"GATE DIAGNOSTIC | margin={args.coverage_margin} "
                    f"| top_neg={args.gate_top_neg}")
        for k, v in gate.items():
            logger.info(f"  {k}: {v}")
        verdict = ("STOP: confusable pairs sit in the zero-penalty zone -- "
                   "coverage scoring cannot change their order (plan front section)"
                   if gate["pairs_with_positive_relu_penalty"] == 0 else
                   "signal present: some pairs have positive penalty difference")
        logger.info(f"  verdict: {verdict}")
        logger.info("=" * 64)
        return
    dev = torch.device(args.device)
    runs, stats, img_perm, cap_perm = rank_all_lambdas(
        feats["img_mu"].to(dev), feats["img_logvar"].to(dev),
        None if feats["img_U"] is None else feats["img_U"].to(dev),
        feats["text_mus"].reshape(-1, feats["text_mus"].shape[-1]).to(dev),
        penalty_grid=grid, coverage_margin=args.coverage_margin, topk=10,
        image_chunk=args.image_chunk_size, caption_chunk=args.caption_chunk_size,
        eps=args.eps, tie_order_seed=args.tie_order_seed)
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # plan §7.4.4: rankings are final and label-free; they can be exported
    # before any pairing information is loaded.
    if args.export_rankings:
        import json as _json
        Path(args.export_rankings).parent.mkdir(parents=True, exist_ok=True)
        Path(args.export_rankings).write_text(_json.dumps({
            "tie_order_seed": args.tie_order_seed,
            "penalty_grid": grid,
            "rankings": {str(lam): {"img_topk": r.img_topk_idx.tolist(),
                                     "cap_topk": r.cap_topk_idx.tolist()}
                         for lam, r in runs.items()},
        }))
        logger.info(f"rankings exported (label-free): {args.export_rankings}")
    if stats.negative_fatal or stats.nonfinite:
        raise FloatingPointError(f"numerical anomalies: {stats.as_dict()}")

    cids = torch.arange(feats["img_mu"].shape[0]).repeat_interleave(
        feats["text_mus"].shape[1])
    recalls = {lam: recalls_from_topk(runs[lam].img_topk_idx, runs[lam].cap_topk_idx,
                                      cids, row_image_ids=img_perm,
                                      row_caption_ids=cap_perm)
               for lam in grid}
    groups = build_recall_groups(recalls, args.recall_at_k)
    print_recall_groups(groups, logger)

    # cross-check: lambda=0 must track the repo's standard cosine protocol.
    # The reference runs torch.topk on the eval device while this scorer uses
    # a CPU stable argsort with index tie-breaking, so near-ties can flip a
    # handful of queries; log the delta and only fail on wiring-level drift.
    ref = compute_multicaption_recall(
        feats["img_mu"].to(args.device),
        torch.zeros_like(feats["img_mu"]).to(args.device),
        feats["text_mus"].to(args.device),
        torch.zeros_like(feats["text_mus"]).to(args.device),
        args.recall_at_k)
    deltas = {f"{d}@{k}": abs(recalls[0.0][f"mc_{d}@{k}"] - ref[f"mc_cos_recall_{d}@{k}"])
              for d in ("i2t", "t2i") for k in args.recall_at_k}
    worst = max(deltas.values())
    logger.info(f"[xcheck] lambda=0 vs mc_cos_recall max|delta|={worst:.2e}")
    assert worst < 0.02, f"cosine cross-check diverged: {deltas}"

    log_diagnostics(feats, args.coverage_margin)
    best = select_best_lambda([{"penalty_weight": lam, "mr": r["mr"]}
                               for lam, r in recalls.items()])
    logger.info(f"Best mR at lambda={best} (ties -> smaller lambda)")

    output_path = args.output_path or str(config.MCDISP_COVERAGE_EVAL_RESULTS_PATH)
    append_eval_results(output_path, {
        "checkpoint": str(checkpoint_path),
        "dataset": args.dataset,
        "split": "dev" if args.dev_seed is not None else "test",
        "num_samples": num_eval_samples,
        "coverage_margin": args.coverage_margin,
        "eps": args.eps,
        "penalty_grid": grid,
        "tie_order_seed": args.tie_order_seed,
        "numeric_stats": stats.as_dict(),
        "metrics": groups_to_flat(groups),
    }, logger)
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Coverage evaluation failed")
        raise
