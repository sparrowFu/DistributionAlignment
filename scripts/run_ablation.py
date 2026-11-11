"""
Exp5: Ablation Study Script (per 消融实验方案.md)

Quantifies the contribution of the three key constraints of MCDisp-Align by
training the KL-objective model with different configurations and evaluating
bidirectional retrieval.

Each ablation variant is trained by the SAME code as the real model — staged
schedule, grad clipping, recall/loss best-checkpoint selection, early stopping —
differing only in the loss weights / ``cov_rank`` / ``num_captions`` overrides,
the ``--dataset`` (coco or flickr), and the per-variant checkpoint path. So
ablation results are directly comparable to a full training run.

Backbone scope: default is frozen-CLIP heads-only (legacy, ~3-4GB). Pass
``--no-freeze-clip`` to unfreeze the ViT-L/14 backbone for every row (the
Full-model recipe LRs: backbone 1e-6, heads 5e-5; ~25-30GB at batch 32 fp32).
Whichever scope you choose must MATCH the Full model being compared against --
frozen and unfrozen rows are not comparable (rebuttal plan 3B).

Ablation configurations (only the named weight -- or cov_rank for diag_only,
or the sg[.] detach for no_sg -- is changed; everything else is identical
across groups; rebuttal-plan rows A0-A7, see config.ABLATION_CONFIGS):
    - A0 Full MCDisp-Align (set-NCE + KL + cover + cov + reg)
    - A1 w/o L_align (lambda_kl = lambda_cover_pos = lambda_cov = lambda_reg = 0;
      only L_match remains)
    - A2 w/o KL Alignment (lambda_kl = 0)
    - A3 w/o Caption Coverage (lambda_cover_pos = 0)
    - A4 w/o Direction Alignment (lambda_cov = 0, low-rank branch kept)
    - A5 w/o Variance Regularization (lambda_reg = 0)
    - A6 Diagonal-only (cov_rank = 0: no low-rank covariance branch; contrasts
      with no_direction, which keeps the branch and removes only the
      direction supervision -- together they isolate supervision vs capacity)
    - A7 w/o sg (detach_targets = False: caption-derived targets of the three
      alignment terms are NOT stop-gradiented; variance schedule unchanged)

Sensitivity grids (``--config sensitivity``): lambda_kl / lambda_cover / tau.

Reports per dataset: image-to-text and text-to-image Recall@1/5/10 under the
MCDisp_Align score; the summary table shows I->T R@1, T->I R@1 and mR (mean of
the six bidirectional recalls).
"""

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.dataset_registry import VALID_DATASETS
from utils.mcdisp_align_trainer import MCDispAlignTrainConfig, run_mcdisp_align_training
from utils.logger import get_logger, log_exception


logger = get_logger("ablation", config.ABLATION_LOG_PATH)

# Exclude faulty CPU cores (e.g. unstable CPU 2) before DataLoader workers and
# torch threads are created. Inherited by forked worker processes.
from utils.cpu_affinity import apply_cpu_affinity
apply_cpu_affinity()


def parse_args():
    parser = argparse.ArgumentParser(description="Exp5: Ablation Study")
    parser.add_argument("--config", type=str, default="all",
                        choices=list(config.ABLATION_CONFIGS.keys()) + ["all", "sensitivity"],
                        help="Ablation configuration to run")
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=list(VALID_DATASETS),
                        help="Dataset to train/evaluate on (coco=MSCOCO, flickr=flickr30k)")
    parser.add_argument("--epochs", type=int, default=config.MCDISP_ALIGN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.MCDISP_ALIGN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.MCDISP_ALIGN_MLP_LR,
                        help="Legacy single LR for the heads (applies to both groups "
                             "in the default frozen-backbone mode)")
    parser.add_argument("--clip-lr", type=float, default=config.MCDISP_ALIGN_CLIP_LR,
                        help="Backbone LR when --no-freeze-clip is set "
                             "(default = the Full-model recipe: MCDISP_ALIGN_CLIP_LR)")
    parser.add_argument("--mlp-lr", type=float, default=config.MCDISP_ALIGN_MLP_LR,
                        help="Head LR when --no-freeze-clip is set "
                             "(default = MCDISP_ALIGN_MLP_LR)")
    parser.add_argument("--no-freeze-clip", dest="freeze_clip", action="store_false",
                        default=True,
                        help="Unfreeze the CLIP backbone (full fine-tune) for EVERY "
                             "ablation row. Rebuttal-plan comparability: whatever this "
                             "setting is, it must match the Full model's recipe -- if "
                             "the paper's Full checkpoints are unfrozen, run the whole "
                             "matrix unfrozen. Needs a large-VRAM GPU (batch 32 fp32 "
                             "~25-30GB; no AMP / grad checkpointing in this path). "
                             "Default: frozen (heads-only, legacy ablation behavior).")
    parser.add_argument("--eval-samples", type=int, default=500,
                        help="Number of eval samples for the final retrieval report "
                             "(coco subset; flickr uses its full test split)")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed (shared by every ablation group so all "
                             "groups use the same data split and candidate pool)")
    parser.add_argument("--loss", type=str, default=None, choices=["standard", "kl"],
                        help="Training objective override for every ablation group "
                             "(default: each ABLATION_CONFIGS entry's loss_name; the "
                             "ablation plan entries use 'kl'). 'standard' = separate "
                             "L_mu/L_var terms; 'kl' = single KL(p_v||p_t) term "
                             "(lambda_kl). Note: the no_kl group is specific to the "
                             "kl objective; under 'standard' it degrades to a second "
                             "full-model run.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only evaluate existing checkpoints")
    return parser.parse_args()


def _config_from_ablation(config_name, ablation_config, args, best_path, last_path):
    """Map an ABLATION_CONFIGS entry + CLI args to a MCDispAlignTrainConfig."""
    return MCDispAlignTrainConfig(
        dataset=args.dataset,
        tag=config_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        # Frozen mode (legacy): heads-only at a single LR. Unfrozen mode
        # (--no-freeze-clip): backbone at MCDISP_ALIGN_CLIP_LR (1e-6) + heads at
        # MCDISP_ALIGN_MLP_LR (5e-5), i.e. the Full-model recipe -- a single
        # 5e-5 LR on the ViT-L backbone would diverge.
        freeze_clip=args.freeze_clip,
        clip_lr=(args.lr if args.freeze_clip else args.clip_lr),
        mlp_lr=(args.lr if args.freeze_clip else args.mlp_lr),
        loss_name=args.loss or ablation_config.get("loss_name", config.ABLATION_LOSS_NAME),
        cov_rank=ablation_config.get("cov_rank", config.MCDISP_ALIGN_COV_RANK),
        num_captions_override=ablation_config.get("num_captions"),
        lambda_ctr=ablation_config.get("lambda_ctr", config.MCDISP_ALIGN_LAMBDA_CTR),
        lambda_kl=ablation_config.get("lambda_kl", 1.0),
        lambda_mu=ablation_config.get("lambda_mu", config.MCDISP_ALIGN_LAMBDA_MU),
        lambda_var=ablation_config.get("lambda_var", config.MCDISP_ALIGN_LAMBDA_VAR),
        lambda_cover_pos=ablation_config.get("lambda_cover_pos", ablation_config.get("lambda_cover", config.MCDISP_ALIGN_LAMBDA_COVER_POS)),
        lambda_cover_neg=ablation_config.get("lambda_cover_neg", config.MCDISP_ALIGN_LAMBDA_COVER_NEG),
        lambda_cov=ablation_config.get("lambda_cov", config.MCDISP_ALIGN_LAMBDA_COV),
        lambda_reg=ablation_config.get("lambda_reg", config.MCDISP_ALIGN_LAMBDA_REG),
        tau=ablation_config.get("temperature", config.MCDISP_ALIGN_TAU),
        use_uncertainty_sim=ablation_config.get("use_uncertainty_sim", True),
        seed=args.seed,
        device=args.device,
        detach_caption_targets=ablation_config.get("detach_targets", True),
        best_ckpt_path=best_path,
        last_ckpt_path=last_path,
        skip_training=args.skip_training,
        eval_num_samples=args.eval_samples,
    )


def train_ablation(config_name, ablation_config, args, output_dir):
    """Train and evaluate a single ablation configuration via the shared trainer."""
    logger.info("\n" + "=" * 60)
    logger.info(f"Ablation: {config_name} (dataset={args.dataset})")
    logger.info(f"Config: {ablation_config}")
    logger.info("=" * 60)

    ckpt_dir = output_dir / "checkpoints" / config_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg = _config_from_ablation(
        config_name, ablation_config, args,
        best_path=ckpt_dir / "best.pt", last_path=ckpt_dir / "last.pt",
    )
    res = run_mcdisp_align_training(cfg, logger)

    return {
        "config": config_name,
        "description": ablation_config.get("description", ""),
        "dataset": args.dataset,
        "best_recall": res.get("best_recall"),
        "best_val_loss": res.get("best_val_loss"),
        "retrieval": res.get("retrieval"),
    }


def run_sensitivity_analysis(args, output_dir):
    """Run lambda_kl, lambda_cover, and tau sensitivity analysis."""
    sensitivity_results = {}

    for lam in config.ABLATION_LAMBDA_KL_VALUES:
        name = f"lambda_kl_{lam}"
        cfg = {**config.ABLATION_CONFIGS["full_model"],
               "lambda_kl": lam, "description": f"lambda_kl={lam}"}
        sensitivity_results[name] = train_ablation(name, cfg, args, output_dir)

    for lam in config.ABLATION_LAMBDA_COVER_VALUES:
        name = f"lambda_cover_{lam}"
        cfg = {**config.ABLATION_CONFIGS["full_model"],
               "lambda_cover_pos": lam, "description": f"lambda_cover={lam}"}
        sensitivity_results[name] = train_ablation(name, cfg, args, output_dir)

    for tau in config.ABLATION_TAU_VALUES:
        name = f"tau_{tau}"
        cfg = {**config.ABLATION_CONFIGS["full_model"],
               "temperature": tau, "description": f"tau={tau}"}
        sensitivity_results[name] = train_ablation(name, cfg, args, output_dir)

    return sensitivity_results


def _retr(metrics, k, direction="mean"):
    """Pull R@k from an ablation result's retrieval block.

    direction: "i2t" / "t2i" pick the one-direction score, anything else the
    bidirectional mean ("mcdisp_align_recall" primary).
    """
    if not metrics:
        return 0.0
    key = "mcdisp_align_recall" if direction == "mean" else f"mcdisp_align_recall_{direction}"
    block = metrics.get(key, metrics.get("mcdisp_align_recall", {}))
    if f"R@{k}" in block:
        return block[f"R@{k}"]
    return metrics.get("cos_recall", {}).get(f"R@{k}", 0.0)


def _mr(metrics):
    """mR: mean of the six bidirectional recalls (I->T/T->I at R@1/5/10)."""
    if not metrics:
        return 0.0
    vals = []
    for key in ("mcdisp_align_recall_i2t", "mcdisp_align_recall_t2i"):
        vals.extend(metrics.get(key, {}).get(f"R@{k}", 0.0) for k in (1, 5, 10))
    return sum(vals) / len(vals) if vals else 0.0


def main():
    args = parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else config.ABLATION_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.config == "all":
        for name, cfg in config.ABLATION_CONFIGS.items():
            all_results[name] = train_ablation(name, cfg, args, output_dir)
    elif args.config == "sensitivity":
        all_results = run_sensitivity_analysis(args, output_dir)
    else:
        cfg = config.ABLATION_CONFIGS[args.config]
        all_results[args.config] = train_ablation(args.config, cfg, args, output_dir)

    # Save results
    output_path = output_dir / f"ablation_results_{args.dataset}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print summary table (main-table columns of the ablation plan)
    print("\n" + "=" * 78)
    print(f"Table: Ablation Study (dataset={args.dataset}, MCDisp_Align score)")
    print("=" * 78)
    print(f"{'Configuration':<42} {'I->T R@1':>9} {'T->I R@1':>9} {'mR':>8}")
    print("-" * 78)
    for name, r in all_results.items():
        ret = r.get("retrieval")
        print(f"{r.get('description', name):<42} "
              f"{_retr(ret, 1, 'i2t'):>9.4f} "
              f"{_retr(ret, 1, 't2i'):>9.4f} "
              f"{_mr(ret):>8.4f}")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Ablation study failed")
        raise
