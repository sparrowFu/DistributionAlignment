"""
Main Entry Point

This script provides a unified entry point for all tasks in the project.
It supports running training and evaluation scripts with a consistent interface.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import config
from utils.dataset_registry import VALID_DATASETS
from utils.logger import get_logger, log_exception
from utils.seed import set_seed
from utils.cpu_affinity import apply_cpu_affinity


logger = get_logger("main", config.MAIN_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available tasks:
  Stage 1 (Alignment Training):
    train_clip_baseline    Train CLIP baseline model (B2)
    eval_clip_baseline     Evaluate CLIP baseline model (B2)
    train_mcdisp_align       Train distribution alignment model (Ours/MCDisp_Align)
    eval_mcdisp_align        Evaluate distribution alignment model (Ours/MCDisp_Align)
    train_prolip           Train ProLIP baseline model (B3)
    eval_prolip            Evaluate ProLIP baseline model (B3)
    eval_prolip_zero_shot  Evaluate ProLIP Zero-Shot baseline (B3)
    eval_clip_zero_shot    Evaluate CLIP Zero-Shot baseline (B1)

  Experiment Tasks:
    run_ablation           Ablation v2 (--command train|eval|report|all
                           --variant full|no_mu|no_var|no_dir_loss|
                           diagonal_only|no_reg|cosine_match)
    eval_ood               Exp4: OOD detection (sigma-based anomaly scoring)
    eval_sigma_analysis    Exp7: sigma semantic analysis
    visualize_gap          Exp8: Modality gap visualization
    eval_flickr30k         Exp6: Flickr30K cross-dataset generalization

Supported model types (--model-type, used by eval_flickr30k): mcdisp_align,
                                     clip_baseline, clip_zero_shot, prolip

MCDisp_Align objective arguments (--lambda-match/mu/var/reg/dir, --tau,
--tau-match, --match-score, --sigma0-sq, --warmup-frac, --cov-rank) are
forwarded to train_mcdisp_align only (ablation variants use their fixed
configs). --lambda-ctr/--lambda-cal are still accepted as deprecated
aliases of --lambda-match/--lambda-reg.

Examples:
  python main.py --task train_mcdisp_align
  python main.py --task train_mcdisp_align --lambda-var 0.5 --cov-rank 0
  python main.py --task eval_mcdisp_align
  python main.py --task run_ablation --command train --variant full
  python main.py --task eval_sigma_analysis
  python main.py --task eval_flickr30k --model-type mcdisp_align
        """
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            # Stage 1: Alignment training
            "train_clip_baseline", "eval_clip_baseline",
            "train_mcdisp_align", "eval_mcdisp_align",
            "train_prolip", "eval_prolip", "eval_prolip_zero_shot",
            "eval_clip_zero_shot",
            # Experiments
            "eval_ood",
            "run_ablation",
            "eval_sigma_analysis",
            "visualize_gap",
            "eval_flickr30k",
        ],
        help="Task to run"
    )

    # Pass-through arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["mcdisp_align", "clip_baseline", "clip_zero_shot",
                 "prolip"],
        help="Model to evaluate for eval_flickr30k"
    )

    parser.add_argument("--captions-path", type=str, default=None,
                        help="Path to captions parquet file")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Path to images directory")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(VALID_DATASETS),
                        help="Dataset: selects the train/eval data source and the "
                             "checkpoint-name tag (coco=MSCOCO, flickr=flickr30k)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--freeze-image", action="store_true")
    parser.add_argument("--freeze-text", action="store_true")

    # MCDisp_Align objective arguments (train_mcdisp_align / run_ablation only)
    parser.add_argument("--lambda-match", type=float, default=None,
                        help="Weight of L_match (distribution-to-set contrastive)")
    parser.add_argument("--lambda-mu", type=float, default=None,
                        help="Weight of L_mu (raw-coordinate center alignment)")
    parser.add_argument("--lambda-var", type=float, default=None,
                        help="Weight of L_var (variance alignment, core)")
    parser.add_argument("--lambda-reg", type=float, default=None,
                        help="Weight of R_prior (weak caption-variance prior)")
    parser.add_argument("--lambda-dir", type=float, default=None,
                        help="Weight of L_dir (direction alignment)")
    parser.add_argument("--lambda-ctr", type=float, default=None,
                        help="[DEPRECATED] Alias of --lambda-match")
    parser.add_argument("--lambda-cal", type=float, default=None,
                        help="[DEPRECATED] Alias of --lambda-reg")
    parser.add_argument("--tau", type=float, default=None,
                        help="Fixed temperature in the cosine match score / retrieval scoring")
    parser.add_argument("--tau-match", type=float, default=None,
                        help="Fixed temperature of the gaussian overlap match logits")
    parser.add_argument("--match-score", type=str, default=None,
                        choices=["gaussian", "cosine"],
                        help="L_match score: gaussian overlap (default) or plain cosine")
    parser.add_argument("--sigma0-sq", type=float, default=None,
                        help="Caption-calibration prior sigma_0^2 for L_cal")
    parser.add_argument("--warmup-frac", type=float, default=None,
                        help="L_var/L_dir warmup fraction of total steps (0 = no ramp)")
    parser.add_argument("--cov-rank", type=int, default=None,
                        help="Low-rank covariance rank r for the image side (0 = diagonal only)")
    parser.add_argument("--mlp-lr", type=float, default=None,
                        help="Learning rate for the MLP distribution heads (train_mcdisp_align only)")
    parser.add_argument("--clip-lr", type=float, default=None,
                        help="Learning rate for CLIP when not frozen (train_mcdisp_align only)")

    # System arguments
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    # Evaluation arguments
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--recall-at-k", type=int, nargs="+", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-path", type=str, default=None)

    # Experiment-specific arguments
    parser.add_argument("--config", "--variant", type=str, default=None,
                        dest="config",
                        help="Ablation variant: full|no_mu|no_var|no_dir_loss|"
                             "diagonal_only|no_reg|cosine_match (alias of --config)")

    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    # Ablation v2 subcommand (run_ablation only; ignored for other tasks)
    parser.add_argument("--command", type=str, default=None,
                        choices=["train", "eval", "report", "all"],
                        help="Ablation v2 subcommand (train/eval/report/all) — "
                             "required for run_ablation")

    return parser.parse_args()


def run_python_script(script_path: Path, args: argparse.Namespace) -> int:
    """Run a Python script with the given arguments."""
    cmd = [sys.executable, str(script_path)]

    # Ablation v2: subcommand must be the first positional argument
    if args.task == "run_ablation" and getattr(args, "command", None):
        cmd.append(args.command)

    if args.model_type:
        cmd.extend(["--model-type", args.model_type])
    if args.captions_path:
        cmd.extend(["--captions-path", args.captions_path])
    if args.images_dir:
        cmd.extend(["--images-dir", args.images_dir])
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.lr is not None:
        cmd.extend(["--lr", str(args.lr)])
    if args.weight_decay is not None:
        cmd.extend(["--weight-decay", str(args.weight_decay)])
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.freeze_image:
        cmd.append("--freeze-image")
    if args.freeze_text:
        cmd.append("--freeze-text")
    # MCDisp_Align objective knobs (train_mcdisp_align only; ablation
    # variants keep their fixed configs, so they are never forwarded there)
    if args.task == "train_mcdisp_align":
        if args.lambda_match is not None:
            cmd.extend(["--lambda-match", str(args.lambda_match)])
        if args.lambda_mu is not None:
            cmd.extend(["--lambda-mu", str(args.lambda_mu)])
        if args.lambda_var is not None:
            cmd.extend(["--lambda-var", str(args.lambda_var)])
        if args.lambda_reg is not None:
            cmd.extend(["--lambda-reg", str(args.lambda_reg)])
        if args.lambda_dir is not None:
            cmd.extend(["--lambda-dir", str(args.lambda_dir)])
        if args.lambda_ctr is not None:
            cmd.extend(["--lambda-ctr", str(args.lambda_ctr)])
        if args.lambda_cal is not None:
            cmd.extend(["--lambda-cal", str(args.lambda_cal)])
        if args.tau is not None:
            cmd.extend(["--tau", str(args.tau)])
        if args.tau_match is not None:
            cmd.extend(["--tau-match", str(args.tau_match)])
        if args.match_score is not None:
            cmd.extend(["--match-score", args.match_score])
        if args.sigma0_sq is not None:
            cmd.extend(["--sigma0-sq", str(args.sigma0_sq)])
        if args.warmup_frac is not None:
            cmd.extend(["--warmup-frac", str(args.warmup_frac)])
        if args.cov_rank is not None:
            cmd.extend(["--cov-rank", str(args.cov_rank)])
        if args.mlp_lr is not None:
            cmd.extend(["--mlp-lr", str(args.mlp_lr)])
        if args.clip_lr is not None:
            cmd.extend(["--clip-lr", str(args.clip_lr)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.num_workers is not None:
        cmd.extend(["--num-workers", str(args.num_workers)])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.recall_at_k:
        cmd.extend(["--recall-at-k"] + [str(k) for k in args.recall_at_k])
    if args.num_samples:
        cmd.extend(["--num-samples", str(args.num_samples)])
    if args.output_path:
        cmd.extend(["--output-path", args.output_path])
    if hasattr(args, 'config') and args.config:
        cmd.extend(["--config", args.config])
    if hasattr(args, 'resume') and args.resume:
        cmd.extend(["--resume", args.resume])

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Script failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        log_exception(logger, e, "Failed to run script")
        return 1


# Task name -> script file mapping.
TASK_SCRIPTS = {
    # Stage 1: Alignment training (Ours + B2-B4)
    "train_clip_baseline":   "train_clip_baseline.py",
    "eval_clip_baseline":    "evaluate_clip_baseline.py",
    "train_mcdisp_align":      "train_mcdisp_align.py",
    "eval_mcdisp_align":       "evaluate_mcdisp_align.py",
    "train_prolip":          "train_prolip.py",
    "eval_prolip":           "evaluate_prolip.py",
    "eval_prolip_zero_shot": "evaluate_prolip_zero_shot.py",
    # Stage 1: Zero-shot
    "eval_clip_zero_shot":   "evaluate_clip_zero_shot.py",
    # Experiments
    "eval_ood":              "eval_ood.py",
    "run_ablation":          "run_ablation.py",
    "eval_sigma_analysis":   "eval_sigma_analysis.py",
    "visualize_gap":         "visualize_modality_gap.py",
    "eval_flickr30k":        "eval_flickr30k.py",
}


def main():
    """Main entry point."""
    args = parse_args()

    if args.task == "run_ablation" and not args.command:
        logger.error("--task run_ablation requires --command "
                     "(train|eval|report|all)")
        return 1

    # Exclude faulty CPU cores (e.g. unstable CPU 2 on this server) before
    # spawning the task subprocess, which inherits this affinity.
    apply_cpu_affinity()

    config.ensure_project_dirs()

    logger.info("=" * 60)
    logger.info("DistributionAlignment")
    logger.info(f"Task: {args.task}")
    logger.info("=" * 60)

    if args.seed:
        set_seed(args.seed)

    scripts_dir = Path(__file__).parent / "scripts"

    script_file = TASK_SCRIPTS.get(args.task)
    if script_file is None:
        logger.error(f"Unknown task: {args.task}")
        return 1

    script_path = scripts_dir / script_file
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return 1

    exit_code = run_python_script(script_path, args)

    if exit_code == 0:
        logger.info(f"Task '{args.task}' completed successfully")
    else:
        logger.error(f"Task '{args.task}' failed with exit code {exit_code}")

    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log_exception(logger, e, "Main entry point failed")
        sys.exit(1)
