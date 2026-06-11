"""
GaussianImageDistribution - LLM VQA Evaluation Script

This script evaluates large language models (LLMs) on the VQA test set
using the SiliconFlow API. Each model receives an image and a question,
and generates a free-form answer.

The results are saved as a JSON file for later evaluation (Strategy C:
free generation + mapping to 430 answer classes).

Usage:
    python scripts/eval_llm_vqa.py
    python scripts/eval_llm_vqa.py --models qwen3.5-4b
    python scripts/eval_llm_vqa.py --start-idx 0 --end-idx 100
    python main.py --task eval_llm_vqa
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.logger import get_logger, log_exception


logger = get_logger("eval_llm_vqa", config.EVAL_LLM_VQA_LOG_PATH)

# System prompt instructing the model to give concise answers
SYSTEM_PROMPT = (
    "You are a helpful visual question answering assistant. "
    "Given an image and a question about it, answer with a single word "
    "or a short phrase. Be concise and direct."
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LLMs on VQA test set")

    # Model selection
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(config.LLM_MODELS.values()),
        default=list(config.LLM_MODELS.values()),
        help="Model shortnames to evaluate",
    )

    # API configuration
    parser.add_argument(
        "--api-config",
        type=str,
        default=str(config.API_CONFIG_PATH),
        help="Path to API configuration JSON file",
    )

    # Data range (for batched processing)
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index in test set (inclusive)",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=-1,
        help="End index in test set (exclusive, -1 for all)",
    )

    # API call settings
    parser.add_argument(
        "--delay",
        type=float,
        default=config.LLM_API_DELAY,
        help="Delay between API calls in seconds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=config.LLM_API_MAX_RETRIES,
        help="Maximum retries for failed API calls",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (overrides per-model default paths)",
    )

    return parser.parse_args()


def load_api_config(config_path: str) -> dict:
    """
    Load API configuration from JSON file.

    Expected format:
    {
        "siliconflow": {
            "api_key": "YOUR_API_KEY",
            "base_url": "https://api.siliconflow.cn/v1"
        }
    }
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"API config file not found: {path}\n"
            f"Please create it with the following format:\n"
            f'{{"siliconflow": {{"api_key": "YOUR_KEY", '
            f'"base_url": "https://api.siliconflow.cn/v1"}}}}'
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_image(image_path: Path) -> Optional[str]:
    """Encode an image file to a base64 string."""
    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return None
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to encode image {image_path}: {e}")
        return None


def query_llm(
    client: OpenAI,
    model: str,
    image_base64: str,
    question: str,
    max_retries: int = 3,
) -> Optional[str]:
    """
    Query an LLM with an image and a question.

    Args:
        client: OpenAI client instance
        model: Full model identifier (e.g., "Qwen/Qwen3.5-4B")
        image_base64: Base64-encoded image string
        question: Question text
        max_retries: Maximum number of retry attempts

    Returns:
        Model response text, or None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                            {"type": "text", "text": question},
                        ],
                    },
                ],
                max_tokens=30,
                timeout=config.LLM_API_TIMEOUT,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(
                f"API call failed (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                wait_time = config.LLM_API_RETRY_WAIT * (attempt + 1)
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"All retries exhausted for question: {question}")
                return None


def load_existing_results(output_path: Path) -> List[dict]:
    """
    Load existing results from a previous run for resume support.

    Returns:
        List of result entries from the output file
    """
    if not output_path.exists():
        return []
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        if not isinstance(results, list):
            return []
        return results
    except Exception as e:
        logger.warning(f"Failed to load existing results: {e}")
        return []


def save_results(results: List[dict], output_path: Path) -> None:
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def get_output_path(model_shortname: str, custom_output: Optional[str]) -> Path:
    """
    Get output file path for a model.

    Args:
        model_shortname: Model short identifier (e.g., "qwen3.5-4b")
        custom_output: User-specified output path override

    Returns:
        Path to the output JSON file
    """
    if custom_output:
        return Path(custom_output)
    return config.LLM_VQA_RESULT_PATHS.get(
        model_shortname,
        config.OUTPUT_DIR / f"llm_vqa_{model_shortname}_results.json",
    )


def main():
    """Main evaluation function."""
    args = parse_args()

    # Load API configuration
    api_config = load_api_config(args.api_config)
    provider = api_config.get("siliconflow", {})
    api_key = provider.get("api_key")
    base_url = provider.get("base_url", "https://api.siliconflow.cn/v1")

    if not api_key or api_key == "YOUR_API_KEY_HERE":
        logger.error("Please set your API key in api_config.json")
        return 1

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Build shortname -> full model name mapping
    model_lookup = {v: k for k, v in config.LLM_MODELS.items()}

    # Load test data
    logger.info("Loading test data...")
    questions_path = config.VQA_TEST_QUESTIONS
    filenames_path = config.VQA_TEST_IMG_FILENAMES
    types_path = config.VQA_TEST_TYPES
    answers_path = config.VQA_TEST_ANSWERS
    images_dir = config.VQA_IMAGES_DIR

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f]
    with open(filenames_path, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f]
    with open(types_path, "r", encoding="utf-8") as f:
        types = [int(line.strip()) for line in f]
    with open(answers_path, "r", encoding="utf-8") as f:
        ground_truths = [line.strip() for line in f]

    n = len(questions)
    assert len(filenames) == n, f"Mismatch: questions={n}, filenames={len(filenames)}"
    assert len(types) == n, f"Mismatch: questions={n}, types={len(types)}"
    assert len(ground_truths) == n, f"Mismatch: questions={n}, answers={len(ground_truths)}"
    logger.info(f"Loaded {n} test samples")

    # Determine processing range
    start_idx = args.start_idx
    end_idx = n if args.end_idx == -1 else min(args.end_idx, n)
    logger.info(f"Processing range: [{start_idx}, {end_idx})")

    # Log configuration
    logger.info("=" * 60)
    logger.info("LLM VQA Evaluation")
    logger.info("=" * 60)
    logger.info(f"Models: {args.models}")
    logger.info(f"Test samples: {n}")
    logger.info(f"Processing range: [{start_idx}, {end_idx})")
    logger.info(f"API delay: {args.delay}s")
    logger.info("=" * 60)

    # Process each model independently
    for model_shortname in args.models:
        full_model_name = model_lookup[model_shortname]
        output_path = get_output_path(model_shortname, args.output)

        logger.info(f"Processing model: {model_shortname} ({full_model_name})")
        logger.info(f"Output file: {output_path}")

        # Load existing results for this model
        results = load_existing_results(output_path)
        processed_ids = {item["id"] for item in results}
        logger.info(f"Found {len(processed_ids)} existing results (will skip these)")

        for idx in range(start_idx, end_idx):
            # Skip already processed samples
            if str(idx) in processed_ids:
                continue

            question = questions[idx]
            filename = filenames[idx]
            q_type = types[idx]
            image_path = images_dir / filename

            # Encode image
            image_base64 = encode_image(image_path)
            if image_base64 is None:
                logger.warning(f"Skipping sample {idx}: image not available")
                continue

            # Query LLM
            predicted = query_llm(
                client, full_model_name, image_base64,
                question, args.max_retries,
            )
            if predicted is None:
                predicted = ""

            # Build result entry
            result = {
                "id": str(idx),
                "model": model_shortname,
                "question": question,
                "question_type": str(q_type),
                "image_name": filename,
                "answer": predicted,
            }

            results.append(result)
            processed_ids.add(str(idx))

            # Periodic save and progress logging
            processed_count = len(results)
            if processed_count % 100 == 0:
                logger.info(
                    f"Progress: {processed_count} samples ({model_shortname})"
                )
                save_results(results, output_path)

            # Rate limiting
            time.sleep(args.delay)

        # Save after model completes
        save_results(results, output_path)
        logger.info(
            f"Model {model_shortname} complete. "
            f"{len(results)} results saved to {output_path}"
        )

    logger.info("=" * 60)
    logger.info("All models evaluated.")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log_exception(logger, e, "LLM VQA evaluation failed")
        sys.exit(1)
