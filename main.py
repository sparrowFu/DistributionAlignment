"""
GaussianImageDistribution - Main Entry Point

This script provides a unified entry point for all tasks in the project.
It supports running training and evaluation scripts with a consistent interface.

Usage:
    python main.py --task train_dist_align
    python main.py --task eval_dist_align
    python main.py --task run_ablation --config all
"""

import argparse
import subprocess
import sys
from pathlib import Path

import config
from utils.logger import get_logger, log_exception
from utils.seed import set_seed
from utils.cpu_affinity import apply_cpu_affinity


# Setup logger
logger = get_logger("main", config.MAIN_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GaussianImageDistribution - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available tasks:
  Stage 1 (Alignment Training):
    train_clip_baseline    Train CLIP baseline model (B2)
    eval_clip_baseline     Evaluate CLIP baseline model (B2)
    train_dist_align       Train distribution alignment model (Ours/MSDA)
    eval_dist_align        Evaluate distribution alignment model (Ours/MSDA)
    train_prolip           Train ProLIP baseline model (B3)
    eval_prolip            Evaluate ProLIP baseline model (B3)
    eval_prolip_zero_shot  Evaluate ProLIP Zero-Shot baseline (B3)
    train_grove            Train GroVE baseline model (B4)
    eval_grove             Evaluate GroVE baseline model (B4)
    eval_clip_zero_shot    Evaluate CLIP Zero-Shot baseline (B1)

  Stage 2 (VQA Downstream):
    train_vqa              Train VQA classification head
    eval_llm_vqa           Query LLMs on VQA test set (B7/B8)
    evaluate_llm_vqa       Compute LLM VQA metrics

  Experiment Tasks:
    run_ablation           Exp5: Ablation study (--config all|no_consistency|...)
    eval_calibration       Exp3: Uncertainty calibration (ECE/NLL/Brier/AUROC)
    eval_ood               Exp4: OOD detection (sigma-based anomaly scoring)
    eval_sigma_analysis    Exp7: sigma semantic analysis
    visualize_gap          Exp8: Modality gap visualization
    eval_flickr30k         Exp6: Flickr30K cross-dataset generalization

Supported model types: dist_align, clip_baseline, clip_zero_shot,
                       prolip, grove

Examples:
  python main.py --task train_dist_align
  python main.py --task eval_dist_align
  python main.py --task train_vqa --model-type dist_align
  python main.py --task run_ablation --config all
  python main.py --task eval_sigma_analysis
  python main.py --task eval_flickr30k --model-type dist_align
        """
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            # Stage 1: Alignment training
            "train_clip_baseline", "eval_clip_baseline",
            "train_dist_align", "eval_dist_align",
            "train_prolip", "eval_prolip", "eval_prolip_zero_shot",
            "train_grove", "eval_grove",
            "eval_clip_zero_shot",
            # Stage 2: VQA downstream
            "train_vqa",
            "eval_llm_vqa", "evaluate_llm_vqa",
            # Experiments
            "eval_calibration",
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
        choices=["dist_align", "clip_baseline", "clip_zero_shot",
                 "prolip", "grove"],
        help="Base model type for VQA training"
    )

    parser.add_argument("--captions-path", type=str, default=None,
                        help="Path to captions parquet file")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Path to images directory")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--freeze-image", action="store_true")
    parser.add_argument("--freeze-text", action="store_true")

    # System arguments
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    # Evaluation arguments
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--recall-at-k", type=int, nargs="+", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-path", type=str, default=None)

    # LLM VQA arguments
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--api-config", type=str, default=None)
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--delay", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)

    # Experiment-specific arguments
    parser.add_argument("--config", type=str, default=None,
                        help="Configuration name for ablation study")

    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    return parser.parse_args()


def run_python_script(script_path: Path, args: argparse.Namespace) -> int:
    """Run a Python script with the given arguments."""
    cmd = [sys.executable, str(script_path)]

    if args.model_type:
        cmd.extend(["--model-type", args.model_type])
    if args.captions_path:
        cmd.extend(["--captions-path", args.captions_path])
    if args.images_dir:
        cmd.extend(["--images-dir", args.images_dir])
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
    if args.models:
        cmd.extend(["--models"] + args.models)
    if args.api_config:
        cmd.extend(["--api-config", args.api_config])
    if args.start_idx is not None:
        cmd.extend(["--start-idx", str(args.start_idx)])
    if args.end_idx is not None:
        cmd.extend(["--end-idx", str(args.end_idx)])
    if args.delay is not None:
        cmd.extend(["--delay", str(args.delay)])
    if args.max_retries is not None:
        cmd.extend(["--max-retries", str(args.max_retries)])
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


# Task to script mapping
TASK_SCRIPTS = {
    # Stage 1: Alignment training (Ours + B2-B4)
    "train_clip_baseline": "train_clip_baseline.py",
    "eval_clip_baseline": "evaluate_clip_baseline.py",
    "train_dist_align": "train_dist_align.py",
    "eval_dist_align": "evaluate_dist_align.py",
    "train_prolip": "train_prolip.py",
    "eval_prolip": "evaluate_prolip.py",
    "eval_prolip_zero_shot": "evaluate_prolip_zero_shot.py",
    "train_grove": "train_grove.py",
    "eval_grove": "evaluate_grove.py",
    # Stage 1: Zero-shot
    "eval_clip_zero_shot": "evaluate_clip_zero_shot.py",
    # Stage 2: VQA downstream
    "train_vqa": "train_vqa.py",
    "eval_llm_vqa": "eval_llm_vqa.py",
    "evaluate_llm_vqa": "evaluate_llm_vqa.py",
    # Experiments
    "eval_calibration": "eval_calibration.py",
    "eval_ood": "eval_ood.py",
    "run_ablation": "run_ablation.py",
    "eval_sigma_analysis": "eval_sigma_analysis.py",
    "visualize_gap": "visualize_modality_gap.py",
    "eval_flickr30k": "eval_flickr30k.py",
}


def main():
    """Main entry point."""
    args = parse_args()

    # Exclude faulty CPU cores (e.g. unstable CPU 2 on this server) before
    # spawning the task subprocess, which inherits this affinity.
    apply_cpu_affinity()

    config.ensure_project_dirs()

    logger.info("=" * 60)
    logger.info("GaussianImageDistribution")
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
