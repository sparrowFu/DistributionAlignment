#!/usr/bin/env python
"""Grouped-similar-batch continuation training for MCDisp-Align
(plan: 固定批量相似样本组批续训方案, 2026-09-14).

Runs the minimal controlled comparison of plan §8 from one completed-stage
checkpoint C0:

    C0  no further training (epoch-0 validation baseline)
    R   5 epochs, small LR, RANDOM batching (rules out "just trained longer")
    G   5 epochs, same as R + the grouped-similarity batching curriculum

Non-negotiables enforced structurally (plan §0): model/loss untouched, batch
size fixed at B, every image exactly once per epoch, images travel with
their own five captions, the loss-stage clock INHERITS the original run's
final state (stage funcs evaluated at epoch E0+e against total E0), and the
LR schedule advances only on successful optimizer steps (§7.2/§7.3).

Example:
    python scripts/finetune_mcdisp_grouped.py \
        --init-checkpoint checkpoints/seed42/mcdisp_align_kl_coco_last.pt \
        --checkpoint-name mcdisp_align_kl --dataset coco --seed 42
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.logger import get_logger, log_exception
from utils.seed import set_seed
from utils.cpu_affinity import apply_cpu_affinity
apply_cpu_affinity()

from utils.lr_scheduler import StepCosineSchedule
from utils.mcdisp_align_trainer import (
    create_optimizer, evaluate, stage_multipliers, train_epoch,
)
from utils.mcdisp_grouped_sampler import (
    batch_indices_to_batch_sampler, make_pools, plan_epoch_batches,
)
from utils.mcdisp_mining_features import (
    build_pool_candidate_table, snapshot_train_features, subset_snapshot,
)

logger = get_logger("finetune_grouped", config.LOG_DIR / "finetune_grouped.log")

PAIR_COUNTS = lambda B: [0, B // 8, B // 8, B // 4, B // 4]   # plan §6.1


def strict_collate(batch):
    """Plan §6.2: corrupt/None samples must FAIL the run, never silently
    shrink a planned batch (the planned indices depend on exact positions)."""
    if any(item is None for item in batch):
        raise RuntimeError("None sample encountered in continuation data path")
    return {"image": [i["image"] for i in batch],
            "captions": [i["captions"] for i in batch]}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def locked_manifests(args, full_dataset):
    """Plan §4.2: rebuild the ORIGINAL random_split with the C0 run's seed,
    then lock the index lists to files; later runs reuse the files so the
    split is decoupled from any training seed."""
    mdir = Path(args.manifest_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    tjson, djson = mdir / f"{args.dataset}_train.json", mdir / f"{args.dataset}_dev.json"
    if tjson.exists() and djson.exists():
        return json.loads(tjson.read_text()), json.loads(djson.read_text())
    val_size = int(len(full_dataset) * 0.1)
    g = torch.Generator().manual_seed(args.orig_split_seed)
    tr, va = torch.utils.data.random_split(
        full_dataset, [len(full_dataset) - val_size, val_size], generator=g)
    tjson.write_text(json.dumps(list(tr.indices)))
    djson.write_text(json.dumps(list(va.indices)))
    logger.info(f"manifests locked: {tjson}, {djson}")
    return list(tr.indices), list(va.indices)


def build_criterion(loss_name: str):
    """Same construction as the trainer; the C0 runs used config defaults,
    which the script records in the run config for traceability (§4.1)."""
    kwargs = dict(
        lambda_ctr=config.MCDISP_ALIGN_LAMBDA_CTR,
        lambda_cover_pos=config.MCDISP_ALIGN_LAMBDA_COVER_POS,
        lambda_cover_neg=config.MCDISP_ALIGN_LAMBDA_COVER_NEG,
        lambda_cov=config.MCDISP_ALIGN_LAMBDA_COV,
        lambda_reg=config.MCDISP_ALIGN_LAMBDA_REG,
        tau=config.MCDISP_ALIGN_TAU, m_pos=config.MCDISP_ALIGN_M_POS,
        target_var=config.MCDISP_ALIGN_TARGET_VAR, m_neg=config.MCDISP_ALIGN_M_NEG,
        use_uncertainty_sim=config.MCDISP_ALIGN_USE_UNCERTAINTY_SIM,
    )
    if loss_name == "kl":
        from losses.mcdisp_align_losses_kl import MCDispAlignKLLoss
        return MCDispAlignKLLoss(lambda_kl=1.0, **kwargs)
    from losses.mcdisp_align_losses import MCDispAlignLoss
    return MCDispAlignLoss(lambda_mu=config.MCDISP_ALIGN_LAMBDA_MU,
                           lambda_var=config.MCDISP_ALIGN_LAMBDA_VAR, **kwargs)


def run_arm(arm, args, model, criterion, full_dataset, train_idx, dev_idx,
            init_sha, run_cfg):
    """One arm ('random' or 'grouped'): 5 epochs, per-epoch validation,
    cosine_mr selection with the epoch-0 C0 baseline in the candidate set."""
    from models.mcdisp_align_model import MCDispAlignModel  # noqa: F401 (type docs)

    out_dir = Path(args.output_root) / f"{args.dataset}_{arm}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "log.jsonl"

    train_ds = Subset(full_dataset, train_idx)
    dev_ds = Subset(full_dataset, dev_idx)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=strict_collate)

    set_seed(args.seed)
    model.load(str(args.init_checkpoint))          # C0 weights ONLY (fresh experiment)
    model = model.to(args.device)

    optimizer = create_optimizer(model, freeze_clip=True,
                                 clip_lr=config.MCDISP_ALIGN_CLIP_LR,
                                 mlp_lr=config.MCDISP_ALIGN_MLP_LR,
                                 weight_decay=config.MCDISP_ALIGN_WEIGHT_DECAY)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    # §7.1: peak = 0.2x ORIGINAL base LR; optimizer freshly created (momentum reset)
    steps_per_epoch = len(train_idx) // args.batch_size + (1 if len(train_idx) % args.batch_size else 0)
    schedule = StepCosineSchedule(base_lrs, total_steps=args.epochs * steps_per_epoch,
                                  peak_multiplier=args.lr_peak_mult,
                                  warmup_fraction=args.lr_warmup_frac,
                                  min_ratio=args.lr_min_ratio)

    E0 = run_cfg["origin_total_epochs"]
    best = {"epoch": 0, "cosine_mr": -1.0}
    successful_steps = 0
    rng = torch.Generator().manual_seed(args.seed + (0 if arm == "random" else 10_000))

    # epoch 0: C0 baseline (plan §9.2 -- C0 is a selection candidate)
    t0 = time.time()
    c0_metrics = evaluate(model, dev_loader, criterion, args.device,
                          compute_recall=True, recall_k_values=[1, 5, 10],
                          multicaption=True)
    best["cosine_mr"] = c0_metrics["cosine_mr"]
    _record(jsonl_path, arm, 0, {"cosine_mr": c0_metrics["cosine_mr"],
                                 "i2t@1": c0_metrics["mc_cos_recall_i2t@1"],
                                 "t2i@1": c0_metrics["mc_cos_recall_t2i@1"],
                                 "stage": "C0-baseline", "val_s": round(time.time() - t0, 1)})
    logger.info(f"[{arm}] epoch 0 (C0): cosine_mr={best['cosine_mr']:.4f}")

    snap_ds_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=strict_collate)
    pair_plan = PAIR_COUNTS(args.batch_size)

    for e in range(args.epochs):
        ep = e + 1
        t_snap = t_plan = 0.0
        # --- per-epoch plan (§5.2): fresh random pools each epoch
        t0 = time.time()
        pools = make_pools(len(train_idx), args.pool_multiplier * args.batch_size, rng)
        tables = [dict() for _ in pools]
        if arm == "grouped" and pair_plan[e] > 0:
            snap = snapshot_train_features(model, snap_ds_loader, args.device)
            t_snap = time.time() - t0
            t0 = time.time()
            for pi, pool in enumerate(pools):
                sub = subset_snapshot(snap, pool)
                tables[pi] = build_pool_candidate_table(
                    sub, tau=config.MCDISP_ALIGN_TAU,
                    use_uncertainty_sim=config.MCDISP_ALIGN_USE_UNCERTAINTY_SIM,
                    rank_start=args.rank_start, rank_end=args.rank_end,
                    gap_min=args.gap_min, gap_max=args.gap_max)
            del snap
        t_plan = time.time() - t0

        batches, plan_stats = plan_epoch_batches(
            pools, tables, pair_plan[e], args.batch_size, rng)
        train_loader = DataLoader(
            train_ds, batch_sampler=batch_indices_to_batch_sampler(batches),
            num_workers=args.num_workers, collate_fn=strict_collate)

        # --- inherited loss-stage clock (§7.2): epoch E0+e against total E0
        mult = stage_multipliers(E0 + e, E0, False)
        criterion.lambda_ctr = config.MCDISP_ALIGN_LAMBDA_CTR * mult["ctr"]
        if hasattr(criterion, "lambda_mu"):
            criterion.lambda_mu = config.MCDISP_ALIGN_LAMBDA_MU * mult["mu"]
        criterion.lambda_cover_pos = config.MCDISP_ALIGN_LAMBDA_COVER_POS * mult["cover_pos"]
        criterion.lambda_cover_neg = config.MCDISP_ALIGN_LAMBDA_COVER_NEG * mult["cover_neg"]
        criterion.lambda_cov = config.MCDISP_ALIGN_LAMBDA_COV * mult["cov"]
        criterion.lambda_reg = config.MCDISP_ALIGN_LAMBDA_REG * mult["reg"]

        t0 = time.time()
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, args.device, e,
            desc_prefix=f"{arm}/", total_epochs=E0, epoch_offset=E0,
            base_lambda_var=config.MCDISP_ALIGN_LAMBDA_VAR,
            base_lambda_kl=getattr(criterion, "lambda_kl", 1.0),
            lr_apply_fn=lambda s: schedule.apply(optimizer, successful_steps + s))
        successful_steps += int(train_metrics.get("successful_steps", 0))
        t_train = time.time() - t0

        t0 = time.time()
        val_metrics = evaluate(model, dev_loader, criterion, args.device,
                               compute_recall=True, recall_k_values=[1, 5, 10],
                               multicaption=True)
        t_val = time.time() - t0

        improved = val_metrics["cosine_mr"] > best["cosine_mr"]
        if improved:
            best = {"epoch": ep, "cosine_mr": val_metrics["cosine_mr"]}
            model.save(str(out_dir / "best.pt"))
        model.save(str(out_dir / "last.pt"))

        _record(jsonl_path, arm, ep, {
            "stage": mult.get("stage"), "planned_pairs": plan_stats["planned_pairs"],
            "qualified_pairs": plan_stats["qualified_pairs"],
            "fallback_pairs": plan_stats["fallback_pairs"],
            "n_batches": plan_stats["n_batches"], "index_hash": plan_stats["index_hash"],
            "successful_steps": successful_steps,
            "clip_step_ratio": train_metrics["clip_step_ratio"],
            "nonfinite_steps": train_metrics["nonfinite_steps"],
            "train_loss": train_metrics["loss"], "set_nce": train_metrics["set_nce"],
            "i2t@1": val_metrics["mc_cos_recall_i2t@1"],
            "t2i@1": val_metrics["mc_cos_recall_t2i@1"],
            "cosine_mr": val_metrics["cosine_mr"], "best": improved,
            "snapshot_s": round(t_snap, 1), "plan_s": round(t_plan, 1),
            "train_s": round(t_train, 1), "val_s": round(t_val, 1),
        })
        logger.info(f"[{arm}] epoch {ep}/{args.epochs}: cosine_mr="
                    f"{val_metrics['cosine_mr']:.4f} best={best['cosine_mr']:.4f}@{best['epoch']} "
                    f"pairs q/f={plan_stats['qualified_pairs']}/{plan_stats['fallback_pairs']}")

    return {"arm": arm, "best_epoch": best["epoch"], "best_cosine_mr": best["cosine_mr"],
            "c0_cosine_mr": c0_metrics["cosine_mr"], "out_dir": str(out_dir)}


def _record(path: Path, arm, epoch, payload: dict):
    rec = {"arm": arm, "epoch": epoch, **payload}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--init-checkpoint", required=True,
                   help="C0: the ORIGINAL run's LAST checkpoint (full-stage)")
    p.add_argument("--checkpoint-name", required=True,
                   choices=["mcdisp_align", "mcdisp_align_kl"])
    p.add_argument("--dataset", required=True, choices=["coco", "flickr"])
    p.add_argument("--seed", type=int, default=42, help="Continuation seed (RNG only)")
    p.add_argument("--orig-split-seed", type=int, default=42,
                   help="Seed of the C0 run's random_split (rebuilds + locks manifests)")
    p.add_argument("--origin-total-epochs", type=int, default=10,
                   help="E0: total epochs of the ORIGINAL run (stage clock)")
    p.add_argument("--arms", default="random,grouped")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pool-multiplier", type=int, default=32)
    p.add_argument("--lr-peak-mult", type=float, default=0.2)
    p.add_argument("--lr-warmup-frac", type=float, default=0.05)
    p.add_argument("--lr-min-ratio", type=float, default=0.1)
    p.add_argument("--rank-start", type=int, default=5)
    p.add_argument("--rank-end", type=int, default=64)
    p.add_argument("--gap-min", type=float, default=0.0)
    p.add_argument("--gap-max", type=float, default=2.0)
    p.add_argument("--manifest-dir", default="outputs/grouped_continuation/manifests")
    p.add_argument("--output-root", default="outputs/grouped_continuation")
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    loss_name = "kl" if args.checkpoint_name.endswith("_kl") else "standard"
    ck = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    ckpt_loss = ck.get("loss_name", "standard")
    if ckpt_loss != loss_name:
        raise ValueError(f"C0 loss_name {ckpt_loss!r} != expected {loss_name!r} "
                         "(plan §11: mismatch must error, not warn)")
    del ck

    set_seed(args.seed)
    from models.mcdisp_align_model import MCDispAlignModel
    from utils.dataset_factory import build_train_dataset
    model = MCDispAlignModel()
    full_dataset = build_train_dataset(dataset=args.dataset)
    train_idx, dev_idx = locked_manifests(args, full_dataset)
    criterion = build_criterion(loss_name).to(args.device)

    run_cfg = {"experiment": "mcdisp_grouped_continuation_v1",
               "init_checkpoint": str(Path(args.init_checkpoint).resolve()),
               "init_sha256": sha256_file(Path(args.init_checkpoint)),
               "loss_name": loss_name, "dataset": args.dataset,
               "origin_total_epochs": args.origin_total_epochs,
               "continuation_epochs": args.epochs, "batch_size": args.batch_size,
               "pool_multiplier": args.pool_multiplier,
               "pair_counts_by_epoch": PAIR_COUNTS(args.batch_size),
               "lr": {"peak_mult": args.lr_peak_mult,
                      "warmup_frac": args.lr_warmup_frac,
                      "min_ratio": args.lr_min_ratio},
               "filters": {"rank": [args.rank_start, args.rank_end],
                           "gap": [args.gap_min, args.gap_max]},
               "manifests": {"train": len(train_idx), "dev": len(dev_idx)}}
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{args.dataset}_run_config.json").write_text(json.dumps(run_cfg, indent=2))

    results = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        logger.info(f"=== arm {arm} ===")
        results[arm] = run_arm(arm, args, model, criterion, full_dataset,
                               train_idx, dev_idx, run_cfg["init_sha256"], run_cfg)
        logger.info(f"=== arm {arm} done: {results[arm]} ===")
    (root / f"{args.dataset}_summary.json").write_text(json.dumps(results, indent=2))
    logger.info("Continuation finished; selection used validation cosine_mr only "
                "(test evaluation is a separate, post-locking step, plan §9.3).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "grouped continuation failed")
        raise
