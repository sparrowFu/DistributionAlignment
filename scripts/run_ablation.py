"""
GaussianImageDistribution - Exp5: Ablation Study Script

Quantifies the contribution of each MSDA loss component by training dist_align
with different configurations and evaluating retrieval.

Each ablation variant is trained by the SAME code as the real model
(:func:`utils.dist_align_trainer.run_dist_align_training`) — staged schedule,
grad clipping, recall/loss best-checkpoint selection, early stopping — differing
only in the loss weights / ``cov_rank`` / ``num_captions`` overrides, the
``--dataset`` (coco or flickr), and the per-variant checkpoint path. So ablation
results are directly comparable to a full training run.

Ablation configurations (see config.ABLATION_CONFIGS):
    - Full MSDA (set-NCE + mu + var + cover + cov + reg)
    - w/o L_var (variance semantic consistency)
    - w/o L_cover (multi-caption coverage)
    - w/o L_cov (covariance direction)
    - w/o L_mu (mean-center alignment)
    - diagonal only (cov_rank=0)
    - w/o uncertainty-discounted similarity (standard cosine)
    - K = 1 / 3 / 5 captions
    - lambda_var sensitivity: 0.1 / 0.5 / 1.0 / 2.0 / 5.0
    - lambda_cover sensitivity: 0.1 / 0.5 / 1.0 / 2.0
    - tau sensitivity: 0.05 / 0.07 / 0.1 / 0.2

Usage:
    python scripts/run_ablation.py --config all --dataset coco
    python scripts/run_ablation.py --config no_var --dataset flickr
    python main.py --task run_ablation --config all --dataset flickr
"""

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.dataset_registry import VALID_DATASETS
from utils.dist_align_trainer import DistAlignTrainConfig, run_dist_align_training
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
    parser.add_argument("--epochs", type=int, default=config.DIST_ALIGN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DIST_ALIGN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.DIST_ALIGN_MLP_LR,
                        help="Learning rate for the (non-frozen) MSDA heads")
    parser.add_argument("--eval-samples", type=int, default=500,
                        help="Number of eval samples for the final retrieval report "
                             "(coco subset; flickr uses its full test split)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only evaluate existing checkpoints")
    return parser.parse_args()


def _config_from_ablation(config_name, ablation_config, args, best_path, last_path):
    """Map an ABLATION_CONFIGS entry + CLI args to a DistAlignTrainConfig."""
    return DistAlignTrainConfig(
        dataset=args.dataset,
        tag=config_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        # Ablation freezes CLIP and trains all heads at a single LR (legacy behavior).
        freeze_clip=True,
        clip_lr=args.lr,
        mlp_lr=args.lr,
        cov_rank=ablation_config.get("cov_rank", config.MSDA_COV_RANK),
        num_captions_override=ablation_config.get("num_captions"),
        lambda_ctr=ablation_config.get("lambda_ctr", config.MSDA_LAMBDA_CTR),
        lambda_mu=ablation_config.get("lambda_mu", config.MSDA_LAMBDA_MU),
        lambda_var=ablation_config.get("lambda_var", config.MSDA_LAMBDA_VAR),
        lambda_cover=ablation_config.get("lambda_cover", config.MSDA_LAMBDA_COVER),
        lambda_cov=ablation_config.get("lambda_cov", config.MSDA_LAMBDA_COV),
        lambda_reg=ablation_config.get("lambda_reg", config.MSDA_LAMBDA_REG),
        tau=ablation_config.get("temperature", config.MSDA_TAU),
        use_uncertainty_sim=ablation_config.get("use_uncertainty_sim", True),
        device=args.device,
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
    res = run_dist_align_training(cfg, logger)

    return {
        "config": config_name,
        "description": ablation_config.get("description", ""),
        "dataset": args.dataset,
        "best_recall": res.get("best_recall"),
        "best_val_loss": res.get("best_val_loss"),
        "retrieval": res.get("retrieval"),
    }


def run_sensitivity_analysis(args, output_dir):
    """Run lambda_var, lambda_cover, and tau sensitivity analysis."""
    sensitivity_results = {}

    for lam in config.ABLATION_LAMBDA_VAR_VALUES:
        name = f"lambda_var_{lam}"
        cfg = {**config.ABLATION_CONFIGS["full_model"],
               "lambda_var": lam, "description": f"lambda_var={lam}"}
        sensitivity_results[name] = train_ablation(name, cfg, args, output_dir)

    for lam in config.ABLATION_LAMBDA_COVER_VALUES:
        name = f"lambda_cover_{lam}"
        cfg = {**config.ABLATION_CONFIGS["full_model"],
               "lambda_cover": lam, "description": f"lambda_cover={lam}"}
        sensitivity_results[name] = train_ablation(name, cfg, args, output_dir)

    for tau in config.ABLATION_TAU_VALUES:
        name = f"tau_{tau}"
        cfg = {**config.ABLATION_CONFIGS["full_model"],
               "temperature": tau, "description": f"tau={tau}"}
        sensitivity_results[name] = train_ablation(name, cfg, args, output_dir)

    return sensitivity_results


def _retr(metrics, k):
    """Pull R@k from an ablation result's retrieval block (msda_recall primary)."""
    if not metrics:
        return 0.0
    msda = metrics.get("msda_recall", {})
    if f"R@{k}" in msda:
        return msda[f"R@{k}"]
    return metrics.get("cos_recall", {}).get(f"R@{k}", 0.0)


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

    # Print summary table
    print("\n" + "=" * 70)
    print(f"Table 4: Ablation Study Results (dataset={args.dataset})")
    print("=" * 70)
    print(f"{'Configuration':<45} {'R@1':>6} {'R@5':>6} {'R@10':>6}")
    print("-" * 70)
    for name, r in all_results.items():
        ret = r.get("retrieval")
        r1 = _retr(ret, 1)
        r5 = _retr(ret, 5)
        r10 = _retr(ret, 10)
        print(f"{r.get('description', name):<45} {r1:>6.4f} {r5:>6.4f} {r10:>6.4f}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Ablation study failed")
        raise
