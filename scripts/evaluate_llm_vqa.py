"""
GaussianImageDistribution - LLM VQA Evaluation & Metrics Script

This script implements Strategy C for LLM VQA evaluation:
  1. Load LLM free-form answers from eval_llm_vqa.py output
  2. Map answers to the 430-class answer vocabulary
  3. Compute accuracy metrics (overall + per-type) comparable with the VQA classifier

Mapping strategy (applied in order):
  - Step 1: Exact match (case-insensitive, punctuation stripped)
  - Step 2: Remove articles ("a"/"an"/the") then exact match
  - Step 3: LLM answer contains a vocab word, or a vocab word contains the LLM answer
  - Step 4: Unmatched → treated as incorrect

Usage:
    python scripts/evaluate_llm_vqa.py
    python scripts/evaluate_llm_vqa.py --models qwen3.5-4b
    python main.py --task evaluate_llm_vqa
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.logger import get_logger, log_exception


logger = get_logger("evaluate_llm_vqa", config.EVALUATE_LLM_VQA_LOG_PATH)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM VQA results with answer mapping"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(config.LLM_MODELS.values()),
        default=list(config.LLM_MODELS.values()),
        help="Model shortnames to evaluate",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output metrics JSON path (uses config default if None)",
    )

    return parser.parse_args()


def normalize_answer(text: str) -> str:
    """
    Normalize an answer string for matching.

    Steps:
      1. Strip whitespace
      2. Lowercase
      3. Remove trailing punctuation (. , ! ? ; :)
    """
    text = text.strip().lower()
    text = re.sub(r'[.,!?;:]+$', '', text)
    text = text.strip()
    return text


def remove_articles(text: str) -> str:
    """Remove leading English articles from text."""
    text = text.strip()
    for article in ("a ", "an ", "the "):
        if text.lower().startswith(article):
            text = text[len(article):].strip()
            break
    return text


def build_vocab_set(train_answers_path: Path) -> Tuple[Set[str], List[str]]:
    """
    Build answer vocabulary from training answers.

    Returns:
        vocab_set: Set of normalized answer strings
        vocab_list: Sorted list of unique answer strings
    """
    with open(train_answers_path, "r", encoding="utf-8") as f:
        answers = [line.strip() for line in f]

    vocab_list = sorted(set(answers))
    vocab_set = {a.lower() for a in vocab_list}
    return vocab_set, vocab_list


def map_answer_to_vocab(
    llm_answer: str,
    vocab_set: Set[str],
) -> Tuple[Optional[str], str]:
    """
    Map an LLM free-form answer to one of the 430 vocab answers.

    Mapping priority:
      1. Exact match (normalized)
      2. Remove articles then exact match
      3. Containment: LLM answer contains a vocab word, or vocab word contains LLM answer
         (only for single-token matches to avoid false positives)

    Args:
        llm_answer: Raw LLM response string
        vocab_set: Set of normalized vocab answers

    Returns:
        (mapped_answer, match_type) where match_type is one of:
        "exact", "no_article", "containment", "unmatched"
    """
    # Step 1: Exact match after normalization
    norm = normalize_answer(llm_answer)
    if norm in vocab_set:
        return norm, "exact"

    # Step 2: Remove leading articles
    no_article = normalize_answer(remove_articles(norm))
    if no_article != norm and no_article in vocab_set:
        return no_article, "no_article"

    # Step 3: Containment matching
    # 3a: Check if any vocab word is contained in the LLM answer
    #     (vocab word must be a standalone word, not a substring)
    llm_tokens = norm.split()
    for token in llm_tokens:
        clean_token = re.sub(r'[.,!?;:]', '', token)
        if clean_token in vocab_set:
            return clean_token, "containment"

    # 3b: Check if any vocab word contains the LLM answer
    #     (useful for short abbreviations, but only if LLM answer is single token)
    if len(llm_tokens) == 1 and len(norm) >= 2:
        for vocab_word in vocab_set:
            if norm in vocab_word and len(norm) / len(vocab_word) >= 0.6:
                return vocab_word, "containment"

    return None, "unmatched"


def load_ground_truth(
    answers_path: Path,
    types_path: Path,
) -> Tuple[List[str], List[int]]:
    """Load ground truth answers and question types."""
    with open(answers_path, "r", encoding="utf-8") as f:
        gt_answers = [line.strip() for line in f]
    with open(types_path, "r", encoding="utf-8") as f:
        types = [int(line.strip()) for line in f]
    return gt_answers, types


def compute_metrics(
    results: List[dict],
    gt_answers: List[str],
    gt_types: List[int],
    vocab_set: Set[str],
) -> dict:
    """
    Compute evaluation metrics for one model.

    Returns:
        Dictionary with overall accuracy, per-type accuracy, and mapping statistics.
    """
    total = 0
    correct = 0
    match_stats = defaultdict(int)
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for item in results:
        idx = int(item["id"])
        llm_answer = item["answer"]
        gt = gt_answers[idx].lower()
        q_type = gt_types[idx]

        # Map LLM answer to vocab
        mapped, match_type = map_answer_to_vocab(llm_answer, vocab_set)
        match_stats[match_type] += 1

        # Check correctness
        is_correct = (mapped == gt) if mapped is not None else False
        total += 1
        if is_correct:
            correct += 1

        type_stats[q_type]["total"] += 1
        if is_correct:
            type_stats[q_type]["correct"] += 1

    # Compute metrics
    overall_accuracy = correct / total if total > 0 else 0.0

    per_type_accuracy = {}
    for q_type in sorted(type_stats.keys()):
        s = type_stats[q_type]
        per_type_accuracy[str(q_type)] = s["correct"] / s["total"] if s["total"] > 0 else 0.0

    return {
        "total_samples": total,
        "correct": correct,
        "overall_accuracy": round(overall_accuracy, 4),
        "per_type_accuracy": {k: round(v, 4) for k, v in per_type_accuracy.items()},
        "per_type_counts": {str(k): {"correct": v["correct"], "total": v["total"]}
                            for k, v in type_stats.items()},
        "mapping_stats": {
            "exact": match_stats.get("exact", 0),
            "no_article": match_stats.get("no_article", 0),
            "containment": match_stats.get("containment", 0),
            "unmatched": match_stats.get("unmatched", 0),
        },
    }


def evaluate_model(
    model_shortname: str,
    gt_answers: List[str],
    gt_types: List[int],
    vocab_set: Set[str],
    vocab_list: List[str],
) -> Optional[dict]:
    """
    Evaluate a single model's results.

    Returns:
        Metrics dictionary, or None if results file not found.
    """
    results_path = config.LLM_VQA_RESULT_PATHS.get(
        model_shortname,
        config.OUTPUT_DIR / f"llm_vqa_{model_shortname}_results.json",
    )

    if not results_path.exists():
        logger.warning(f"Results file not found for {model_shortname}: {results_path}")
        logger.warning(f"Run 'python scripts/eval_llm_vqa.py --models {model_shortname}' first.")
        return None

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    logger.info(f"Loaded {len(results)} results for {model_shortname}")

    if len(results) == 0:
        logger.warning(f"No results found for {model_shortname}")
        return None

    metrics = compute_metrics(results, gt_answers, gt_types, vocab_set)
    metrics["model"] = model_shortname
    metrics["results_file"] = str(results_path)
    metrics["vocab_size"] = len(vocab_list)

    return metrics


def print_metrics(metrics: dict) -> None:
    """Print metrics summary to logger."""
    logger.info(f"  Model: {metrics['model']}")
    logger.info(f"  Vocab size: {metrics['vocab_size']}")
    logger.info(f"  Total samples: {metrics['total_samples']}")
    logger.info(f"  Correct: {metrics['correct']}")
    logger.info(f"  Overall accuracy: {metrics['overall_accuracy']:.4f}")
    logger.info(f"  Mapping statistics:")
    for match_type, count in metrics["mapping_stats"].items():
        pct = count / metrics["total_samples"] * 100 if metrics["total_samples"] > 0 else 0
        logger.info(f"    {match_type}: {count} ({pct:.1f}%)")
    logger.info(f"  Per-type accuracy:")
    for q_type, acc in sorted(metrics["per_type_accuracy"].items()):
        counts = metrics["per_type_counts"][q_type]
        logger.info(
            f"    Type {q_type}: {acc:.4f} "
            f"({counts['correct']}/{counts['total']})"
        )


def main():
    """Main evaluation function."""
    args = parse_args()

    # Build answer vocabulary from training set
    logger.info("Building answer vocabulary from training set...")
    vocab_set, vocab_list = build_vocab_set(config.VQA_TRAIN_ANSWERS)
    logger.info(f"Answer vocabulary: {len(vocab_list)} classes")

    # Load ground truth
    logger.info("Loading ground truth...")
    gt_answers, gt_types = load_ground_truth(
        config.VQA_TEST_ANSWERS,
        config.VQA_TEST_TYPES,
    )
    logger.info(f"Ground truth: {len(gt_answers)} samples")

    # Evaluate each model
    all_metrics = []
    for model_shortname in args.models:
        logger.info(f"")
        logger.info(f"{'=' * 50}")
        logger.info(f"Evaluating: {model_shortname}")

        metrics = evaluate_model(
            model_shortname, gt_answers, gt_types,
            vocab_set, vocab_list,
        )

        if metrics is None:
            continue

        print_metrics(metrics)
        all_metrics.append(metrics)

    # Save combined results
    if all_metrics:
        output_path = Path(args.output_path) if args.output_path else config.OUTPUT_DIR / "llm_vqa_metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"")
        logger.info(f"Metrics saved to: {output_path}")

    # Summary
    logger.info(f"")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    for m in all_metrics:
        logger.info(
            f"  {m['model']:>15s}: "
            f"accuracy={m['overall_accuracy']:.4f} "
            f"({m['correct']}/{m['total_samples']})"
        )
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log_exception(logger, e, "LLM VQA evaluation failed")
        sys.exit(1)
