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

  Stage 2 (VQA Downstream):
    build_vqa_expansions   Build VQA-as-retrieval caption dataset (local gemma/llama or API)
    eval_vqa_retrieval     VQA-as-retrieval eval (gemma caption, 5 models)

  Experiment Tasks:
    run_ablation           Ablation v2 (--command train --variant full|no_var|no_dir|no_cover)
    eval_ood               Exp4: OOD detection (sigma-based anomaly scoring)
    eval_sigma_analysis    Exp7: sigma semantic analysis
    visualize_gap          Exp8: Modality gap visualization
    eval_flickr30k         Exp6: Flickr30K cross-dataset generalization

Supported model types (--model-type): mcdisp_align, clip_baseline, clip_zero_shot,
                                     prolip
eval_vqa_retrieval --model: clip_zero_shot, clip_baseline, mcdisp_align,
                            prolip_zero_shot, prolip, all

Examples:
  python main.py --task train_mcdisp_align
  python main.py --task eval_mcdisp_align
  python main.py --task build_vqa_expansions --split test --limit 0 --no-batch
  python main.py --task eval_vqa_retrieval --model mcdisp_align
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
            # Stage 2: VQA-as-retrieval downstream (gemma caption)
            "build_vqa_expansions",
            "eval_vqa_retrieval",
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
        help="Base model type for VQA training"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["clip_zero_shot", "clip_baseline", "mcdisp_align",
                 "prolip_zero_shot", "prolip", "all"],
        help="Model to evaluate for eval_vqa_retrieval ('all' = 全部 5 个)"
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
                        help="Ablation variant: full|no_var|no_dir|no_cover "
                             "(alias of --config)")

    # Caption-build arguments (build_vqa_expansions)
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "test", "both"],
                        help="VQA split for build_vqa_expansions")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["local", "api"],
                        help="Caption backend for build_vqa_expansions (local/api)")
    parser.add_argument("--model-kind", type=str, default=None,
                        choices=["gemma", "llama"],
                        help="Local model family for build_vqa_expansions (gemma/llama)")
    parser.add_argument("--no-batch", action="store_true",
                        help="build_vqa_expansions: one caption at a time (required for local gemma)")
    parser.add_argument("--limit", type=int, default=None,
                        help="build_vqa_expansions: samples to process (0 = all)")
    parser.add_argument("--no-resume", action="store_true",
                        help="build_vqa_expansions: start from scratch (overwrite existing outputs)")

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
    if args.model:
        cmd.extend(["--model", args.model])
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
    if hasattr(args, 'split') and args.split:
        cmd.extend(["--split", args.split])
    if hasattr(args, 'backend') and args.backend:
        cmd.extend(["--backend", args.backend])
    if hasattr(args, 'model_kind') and args.model_kind:
        cmd.extend(["--model-kind", args.model_kind])
    if hasattr(args, 'no_batch') and args.no_batch:
        cmd.append("--no-batch")
    if hasattr(args, 'limit') and args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if hasattr(args, 'no_resume') and args.no_resume:
        cmd.append("--no-resume")
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
    # Stage 2: VQA-as-retrieval downstream (gemma caption)
    "build_vqa_expansions":  "build_vqa_expansions.py",
    "eval_vqa_retrieval":    "eval_vqa_retrieval.py",
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
