"""
GaussianImageDistribution - Main Entry Point

This script provides a unified entry point for all tasks in the project.
It supports running training and evaluation scripts with a consistent interface.

Usage:
    python main.py --task train_clip_baseline
    python main.py --task eval_clip_baseline
"""

import argparse
import subprocess
import sys
from pathlib import Path

import config
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


# Setup logger
logger = get_logger("main", config.MAIN_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GaussianImageDistribution - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available tasks:
  train_clip_baseline    Train CLIP baseline model
  eval_clip_baseline     Evaluate CLIP baseline model
  train_dist_align       Train distribution alignment model
  eval_dist_align        Evaluate distribution alignment model
  train_freeze_align     Train Freeze-Align model
  eval_freeze_align      Evaluate Freeze-Align model
  train_fate             Train FATE model
  eval_fate              Evaluate FATE model
  train_clip_ast         Train CLIP-AST model
  eval_clip_ast          Evaluate CLIP-AST model
  eval_clip_zero_shot    Evaluate CLIP Zero-Shot baseline
  train_vqa              Train VQA classification head
  eval_llm_vqa           Query LLMs on VQA test set (API calls)
  evaluate_llm_vqa       Compute LLM VQA metrics (answer mapping + accuracy)

Examples:
  python main.py --task train_freeze_align
  python main.py --task eval_freeze_align
  python main.py --task train_vqa --model-type freeze_align
  python main.py --task train_vqa --model-type clip_zero_shot
        """
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["train_clip_baseline", "eval_clip_baseline", "train_dist_align", "eval_dist_align", "train_freeze_align", "eval_freeze_align", "train_fate", "eval_fate", "train_clip_ast", "eval_clip_ast", "eval_clip_zero_shot", "train_vqa", "eval_llm_vqa", "evaluate_llm_vqa"],
        help="Task to run"
    )

    # Pass-through arguments for scripts
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["dist_align", "clip_baseline", "freeze_align", "fate", "clip_ast", "clip_zero_shot"],
        help="Base model type for VQA training"
    )

    parser.add_argument(
        "--captions-path",
        type=str,
        default=None,
        help="Path to captions parquet file (uses config default if None)"
    )

    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Path to images directory (uses config default if None)"
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Weight decay")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature for contrastive loss")

    # Model arguments
    parser.add_argument("--freeze-image", action="store_true",
                        help="Freeze image encoder")
    parser.add_argument("--freeze-text", action="store_true",
                        help="Freeze text encoder")

    # System arguments
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use")

    # Evaluation arguments
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (for evaluation)")
    parser.add_argument("--recall-at-k", type=int, nargs="+", default=None,
                        help="Recall@K values to compute")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to evaluate")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output JSON path (for evaluation)")

    # LLM VQA evaluation arguments
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="LLM model shortnames to evaluate (qwen3.5-4b, kimi-k2.5)")
    parser.add_argument("--api-config", type=str, default=None,
                        help="Path to API configuration JSON file")
    parser.add_argument("--start-idx", type=int, default=None,
                        help="Start index in test set")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="End index in test set (-1 for all)")
    parser.add_argument("--delay", type=float, default=None,
                        help="Delay between API calls in seconds")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="Maximum retries for failed API calls")

    return parser.parse_args()


def run_python_script(script_path: Path, args: argparse.Namespace) -> int:
    """
    Run a Python script with the given arguments.

    Args:
        script_path: Path to the script to run
        args: Parsed command line arguments

    Returns:
        Exit code from the script
    """
    # Build command list
    cmd = [sys.executable, str(script_path)]

    # Add model type (for VQA)
    if args.model_type:
        cmd.extend(["--model-type", args.model_type])

    # Add relevant arguments
    if args.captions_path:
        cmd.extend(["--captions-path", args.captions_path])
    if args.images_dir:
        cmd.extend(["--images-dir", args.images_dir])

    # Training/evaluation specific arguments
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

    # Model arguments
    if args.freeze_image:
        cmd.append("--freeze-image")
    if args.freeze_text:
        cmd.append("--freeze-text")

    # System arguments
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.num_workers is not None:
        cmd.extend(["--num-workers", str(args.num_workers)])
    if args.device is not None:
        cmd.extend(["--device", args.device])

    # Evaluation specific arguments
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.recall_at_k:
        cmd.extend(["--recall-at-k"] + [str(k) for k in args.recall_at_k])
    if args.num_samples:
        cmd.extend(["--num-samples", str(args.num_samples)])
    if args.output_path:
        cmd.extend(["--output-path", args.output_path])

    # LLM VQA evaluation arguments
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

    logger.info(f"Running: {' '.join(cmd)}")

    # Run the script
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Script failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        log_exception(logger, e, "Failed to run script")
        return 1


def main():
    """Main entry point."""
    args = parse_args()

    # Ensure output directories exist
    config.ensure_project_dirs()

    # Log configuration
    logger.info("=" * 60)
    logger.info("GaussianImageDistribution")
    logger.info("=" * 60)
    logger.info(f"Task: {args.task}")
    logger.info("=" * 60)

    # Set random seed if provided
    if args.seed:
        set_seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")

    # Determine which script to run
    scripts_dir = Path(__file__).parent / "scripts"

    if args.task == "train_clip_baseline":
        script_path = scripts_dir / "train_clip_baseline.py"
        logger.info("Starting CLIP baseline training...")
    elif args.task == "eval_clip_baseline":
        script_path = scripts_dir / "evaluate_clip_baseline.py"
        logger.info("Starting CLIP baseline evaluation...")
    elif args.task == "train_dist_align":
        script_path = scripts_dir / "train_dist_align.py"
        logger.info("Starting distribution alignment training...")
    elif args.task == "eval_dist_align":
        script_path = scripts_dir / "evaluate_dist_align.py"
        logger.info("Starting distribution alignment evaluation...")
    elif args.task == "train_freeze_align":
        script_path = scripts_dir / "train_freeze_align.py"
        logger.info("Starting Freeze-Align training...")
    elif args.task == "eval_freeze_align":
        script_path = scripts_dir / "evaluate_freeze_align.py"
        logger.info("Starting Freeze-Align evaluation...")
    elif args.task == "train_fate":
        script_path = scripts_dir / "train_fate.py"
        logger.info("Starting FATE training...")
    elif args.task == "eval_fate":
        script_path = scripts_dir / "evaluate_fate.py"
        logger.info("Starting FATE evaluation...")
    elif args.task == "train_clip_ast":
        script_path = scripts_dir / "train_clip_ast.py"
        logger.info("Starting CLIP-AST training...")
    elif args.task == "eval_clip_ast":
        script_path = scripts_dir / "evaluate_clip_ast.py"
        logger.info("Starting CLIP-AST evaluation...")
    elif args.task == "eval_clip_zero_shot":
        script_path = scripts_dir / "evaluate_clip_zero_shot.py"
        logger.info("Starting CLIP Zero-Shot evaluation...")
    elif args.task == "train_vqa":
        script_path = scripts_dir / "train_vqa.py"
        logger.info("Starting VQA fine-tuning training...")
    elif args.task == "eval_llm_vqa":
        script_path = scripts_dir / "eval_llm_vqa.py"
        logger.info("Starting LLM VQA evaluation...")
    elif args.task == "evaluate_llm_vqa":
        script_path = scripts_dir / "evaluate_llm_vqa.py"
        logger.info("Starting LLM VQA metrics computation...")
    else:
        logger.error(f"Unknown task: {args.task}")
        return 1

    # Run the script
    exit_code = run_python_script(script_path, args)

    if exit_code == 0:
        logger.info("=" * 60)
        logger.info(f"Task '{args.task}' completed successfully")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error(f"Task '{args.task}' failed with exit code {exit_code}")
        logger.error("=" * 60)

    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log_exception(logger, e, "Main entry point failed")
        sys.exit(1)
