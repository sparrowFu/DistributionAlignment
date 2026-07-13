"""
GaussianImageDistribution - Exp3: Uncertainty Calibration Evaluation Script

Evaluates calibration metrics (ECE, NLL, Brier) for models with
uncertainty estimates. Only compares methods that model σ: B3 ProLIP,
B4 GroVE, and Ours (MSDA).

Usage:
    python scripts/eval_calibration.py --methods dist_align prolip grove
    python main.py --task eval_calibration --methods dist_align
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.vqa_dataset import VQADataset, vqa_collate_fn
from models.vqa_model import VQAModel
from utils.calibration import evaluate_calibration
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


logger = get_logger("eval_calibration", config.CALIBRATION_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Exp3: Uncertainty Calibration Evaluation"
    )

    parser.add_argument(
        "--methods", type=str, nargs="+",
        default=["dist_align"],
        choices=["dist_align", "prolip", "grove"],
        help="Methods to evaluate (only methods with σ)",
    )
    parser.add_argument(
        "--mc-samples", type=int, nargs="+",
        default=[0, 5, 10],
        help="MC sample counts for dist_align (0=deterministic)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-bins", type=int, default=config.CALIBRATION_NUM_BINS)
    parser.add_argument("--output-dir", type=str, default=None)

    return parser.parse_args()


def load_vqa_test_data():
    """Load VQA test dataset."""
    test_dataset = VQADataset(
        questions_path=config.VQA_TEST_QUESTIONS,
        img_filenames_path=config.VQA_TEST_IMG_FILENAMES,
        types_path=config.VQA_TEST_TYPES,
        answers_path=config.VQA_TEST_ANSWERS,
        images_dir=config.VQA_IMAGES_DIR,
    )
    return test_dataset


def evaluate_method(
    method: str,
    test_dataset,
    args,
):
    """Evaluate calibration for a single method."""
    results_list = []

    # Get answer vocab from test dataset
    answer_vocab = test_dataset.get_answer_vocab()

    # Determine checkpoint and MC samples
    if method == "dist_align":
        ckpt_path = str(config.VQA_DIST_ALIGN_CKPT)
        mc_samples_list = args.mc_samples
    elif method == "prolip":
        ckpt_path = str(config.VQA_PROLIP_CKPT)
        mc_samples_list = [0]  # ProLIP doesn't use MC in this context
    elif method == "grove":
        ckpt_path = str(config.VQA_GROVE_CKPT)
        mc_samples_list = [0]
    else:
        raise ValueError(f"Unknown method: {method}")

    for num_mc in mc_samples_list:
        logger.info(f"\n{'='*50}")
        logger.info(f"Method: {method}, MC samples: {num_mc}")
        logger.info(f"{'='*50}")

        dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=vqa_collate_fn,
        )

        # Create VQA model
        vqa_model = VQAModel(
            model_type=method if method != "prolip" else "dist_align",
            num_classes=test_dataset.num_classes,
            hidden_dim=config.VQA_HIDDEN_DIM,
            dropout=config.VQA_DROPOUT,
            answer_vocab=answer_vocab,
            base_ckpt_path=ckpt_path if Path(ckpt_path).exists() else None,
            device=args.device,
            num_mc_samples=num_mc,
        )
        vqa_model = vqa_model.to(args.device)

        # Evaluate
        results = evaluate_calibration(
            model_type=f"{method}_mc{num_mc}",
            vqa_model=vqa_model,
            dataloader=dataloader,
            device=args.device,
            num_mc_samples=num_mc,
            num_bins=args.num_bins,
        )
        results_list.append(results)

    return results_list


def main():
    """Main calibration evaluation."""
    args = parse_args()
    set_seed(config.SEED)

    # Create output directory
    output_dir = Path(args.output_dir) if args.output_dir else config.CALIBRATION_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    logger.info("Loading VQA test dataset...")
    test_dataset = load_vqa_test_data()
    logger.info(f"Test set: {len(test_dataset)} samples")

    # Evaluate each method
    all_results = {}
    for method in args.methods:
        results_list = evaluate_method(method, test_dataset, args)
        all_results[method] = results_list

    # Save results
    output_path = output_dir / "calibration_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("Table 3: Uncertainty Calibration Results")
    print("=" * 80)
    print(f"{'Method':<25} {'ECE↓':>8} {'NLL↓':>8} {'Brier↓':>10} {'Acc':>8}")
    print("-" * 80)
    for method, results_list in all_results.items():
        for r in results_list:
            name = r["model_type"]
            print(f"{name:<25} {r['ece']:>8.4f} {r['nll']:>8.4f} "
                  f"{r['brier_score']:>10.4f} {r['accuracy']:>8.4f}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Calibration evaluation failed")
        raise
