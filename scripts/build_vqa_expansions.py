"""
Build a VQA-as-retrieval expansion dataset (API caption generation).

For each VQA sample, call an OpenAI-compatible chat API (SiliconFlow) to turn the
(question, answer) pair into a single grammatical, natural-English declarative
caption that describes the image. No templates, no placeholders, no regex -- the
LLM emits the final caption directly.

Output JSON (one entry per sample), written to
    outputs/vqa_expansions/{split}_expansions.json
    [
      {
        "imagefilename": "000000397899.jpg",
        "imagepath": "TrainDatasets/mscoco_captions/images/000000397899.jpg",
        "type": 0,
        "question": "...",
        "answer": "...",
        "caption": "<declarative English sentence>"
      },
      ...
    ]

Resumable: streams entries to a .jsonl (one per line), caches generated captions
to a separate {split}_captions.jsonl (keyed by question+answer, so identical
Q/A pairs across images reuse one API call), and consolidates to the final .json
array at the end.

Usage:
    # quick dry run (3 samples, start fresh)
    python scripts/build_vqa_expansions.py --split test --limit 3 --no-resume
    # full run over the whole split
    python scripts/build_vqa_expansions.py --split test --limit 0
    # switch provider / model (aliases defined in api_config.json)
    python scripts/build_vqa_expansions.py --provider siliconflow --model deepseek-v4-flash

API credentials live in api_config.json (gitignored; see api_config.example.json for
the format). Add more providers or models there -- select them with --provider/--model.
NOTE: the existing outputs/vqa_expansions/test_expansions.{json,jsonl} were built
with the old regex backend; pass --no-resume once to regenerate them with the API.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from utils.logger import get_logger, log_exception

logger = get_logger("build_vqa_expansions", config.LOG_DIR / "build_vqa_expansions.log")

TYPE_NAMES = {0: "object", 1: "number", 2: "color", 3: "location"}

# Defaults: which provider + model alias to use from api_config.json. Override on the CLI.
DEFAULT_PROVIDER = "siliconflow"
DEFAULT_MODEL_ALIAS = "deepseek-v4-pro"

# Suffixes: train uses plain names, test uses *_filtered.
_SPLIT_FILES = {
    "train": {
        "questions": "questions", "img_filenames": "img_filenames",
        "types": "types", "answers": "answers",
    },
    "test": {
        "questions": "questions_filtered", "img_filenames": "img_filenames_filtered",
        "types": "types_filtered", "answers": "answers_filtered",
    },
}


# =============================================================================
# Data loading
# =============================================================================
def load_split(split: str) -> List[Tuple[str, int, str, str]]:
    """Return [(filename, type, question, answer), ...] line-aligned."""
    base = config.VQA_TRAIN_QUESTIONS.parent.parent  # .../mscoco_captions
    split_dir = base / split
    names = _SPLIT_FILES[split]
    paths = {k: split_dir / f"{v}.txt" for k, v in names.items()}
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {k} file: {p}")

    def read(p: Path) -> List[str]:
        return [line.rstrip("\n") for line in p.read_text(encoding="utf-8").splitlines()]

    questions = read(paths["questions"])
    filenames = read(paths["img_filenames"])
    types = read(paths["types"])
    answers = read(paths["answers"])
    n = len(questions)
    if not (len(filenames) == len(types) == len(answers) == n):
        raise ValueError(f"Line mismatch in {split}: q={n} f={len(filenames)} "
                         f"t={len(types)} a={len(answers)}")
    samples = []
    for fn, t, q, a in zip(filenames, types, questions, answers):
        t = int(t.strip()) if t.strip() else 0
        samples.append((fn.strip(), t, q.strip(), a.strip()))
    logger.info(f"Loaded {len(samples)} samples from split={split}")
    return samples


def image_relpath(filename: str) -> str:
    """Relative-to-project-root image path (forward slashes), e.g.
    'TrainDatasets/mscoco_captions/images/000000397899.jpg'."""
    full = config.VQA_IMAGES_DIR / filename
    try:
        return str(full.relative_to(config.PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(full).replace("\\", "/")


# =============================================================================
# Caption generator (API-only, OpenAI-compatible via urllib)
# =============================================================================
_CAPTION_SYSTEM = (
    "You convert visual question-answer pairs into natural English declarative "
    "image-description sentences. Always output English. Every sentence MUST be "
    "grammatically correct and read like a fluent caption of what the image shows."
)

_CAPTION_USER = """Rewrite the following question-answer pair as ONE declarative sentence that DESCRIBES THE IMAGE. The answer must appear naturally inside the sentence.

Question: {QUESTION}
Answer: {ANSWER}
Answer type: {TYPE}   (one of: color | number | location | object)

Rules:
- Output ONLY the caption sentence. No quotes, no explanation, no extra text.
- Make it declarative and image-descriptive, not a question.
- GRAMMAR (important): the sentence MUST be grammatically correct, natural English. Fix word order, subject-verb agreement, articles (a/an/the), pluralization, and prepositions. You MAY reorder or lightly rephrase the question's words to achieve fluent grammar. Drop a leading "a"/"an" when the subject is plural (e.g. "carrots are ...", not "a carrots are ..."). For "how many" questions, prefer "there are <answer> ...".

Examples:
Q: what is the man holding a snowboard on top of a snow covered | A: hill | type: object
a man is holding a snowboard on top of a snow covered hill

Q: what are sitting on the counter in different stages of cutting with a knife | A: carrots | type: object
carrots are sitting on the counter in different stages of cutting with a knife

Q: what is the color of the bus | A: green | type: color
the color of the bus is green

Q: how many red motorcycles with riders in protective gear are on the street | A: three | type: number
there are three red motorcycles with riders in protective gear on the street

Q: where is the woman while her baby is sleeping | A: kitchen | type: location
the woman is in the kitchen while her baby is sleeping

Now convert:
Q: {QUESTION} | A: {ANSWER} | type: {TYPE}"""


class CaptionGenerator:
    """Call an OpenAI-compatible chat API to emit a final caption directly."""

    def __init__(self, api_key: str, base_url: str, model: str,
                 max_retries: int = 3, retry_wait: float = 5.0,
                 delay: float = 0.5, timeout: int = 60, max_tokens: int = 96):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.delay = delay
        self.timeout = timeout
        self.max_tokens = max_tokens

    def _chat(self, messages: List[Dict]) -> str:
        url = self.base_url + "/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")
        last_err = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    url, data=body,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, KeyError, ValueError) as e:
                last_err = e
                wait = self.retry_wait * (attempt + 1)
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}; retry in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"API call failed after {self.max_retries} attempts: {last_err}")

    def generate(self, question: str, answer: str, qtype: int) -> str:
        # Use .replace (not .format) so any literal braces survive untouched.
        user = (_CAPTION_USER
                .replace("{QUESTION}", question)
                .replace("{ANSWER}", answer)
                .replace("{TYPE}", TYPE_NAMES.get(qtype, "object")))
        messages = [
            {"role": "system", "content": _CAPTION_SYSTEM},
            {"role": "user", "content": user},
        ]
        text = self._chat(messages)
        if self.delay:
            time.sleep(self.delay)
        # Keep only the last non-empty line (model may add a stray preamble); strip quotes.
        cand = [c.strip().strip('"').strip("'").strip() for c in text.splitlines() if c.strip()]
        caption = cand[-1] if cand else ""
        if not caption:
            logger.warning(f"Empty caption for Q={question!r} A={answer!r}; using answer as caption")
            caption = answer
        return caption


def load_api_config(path: Path) -> Dict:
    """Load api_config.json: {provider: {api_key, base_url, models: {alias: id}}}.

    Top-level keys starting with '_' (e.g. '_readme') are ignored, so the file can
    carry documentation. See api_config.example.json for the schema.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"api_config not found: {path}\nCopy api_config.example.json to "
            f"{path.name} and fill in your key (it is gitignored).")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_provider(cfg: Dict, provider: str) -> Dict:
    """Return the provider block for `provider`."""
    available = [k for k in cfg if not k.startswith("_") and isinstance(cfg[k], dict)]
    prov = cfg.get(provider)
    if not isinstance(prov, dict):
        raise ValueError(f"Provider '{provider}' not in api_config. Available: {available}")
    if not prov.get("api_key") or not prov.get("base_url"):
        raise ValueError(f"Provider '{provider}' must define 'api_key' and 'base_url'")
    return prov


def make_generator(args) -> CaptionGenerator:
    cfg = load_api_config(Path(args.api_config))
    prov = resolve_provider(cfg, args.provider)
    models = prov.get("models") or {}
    model_id = models.get(args.model)
    if model_id is None:
        # Treat --model as a raw model id if it is not a known alias.
        model_id = args.model
        logger.warning(f"Model alias '{args.model}' not in provider '{args.provider}' "
                       f"models map (have {list(models)}); using it as a raw model id")
    logger.info(f"Using API caption generator: provider={args.provider} "
                f"model={args.model} ({model_id}) base_url={prov['base_url']}")
    return CaptionGenerator(
        api_key=prov["api_key"], base_url=prov["base_url"], model=model_id,
        max_retries=config.LLM_API_MAX_RETRIES, retry_wait=config.LLM_API_RETRY_WAIT,
        delay=config.LLM_API_DELAY, timeout=config.LLM_API_TIMEOUT,
    )


# =============================================================================
# Pipeline
# =============================================================================
def caption_cache_key(question: str, answer: str) -> str:
    return f"{question}\t{answer}"


def build_entry(filename: str, qtype: int, question: str, answer: str, caption: str) -> Dict:
    return {
        "imagefilename": filename,
        "imagepath": image_relpath(filename),
        "type": qtype,
        "question": question,
        "answer": answer,
        "caption": caption,
    }


def load_caption_cache(path: Path) -> Dict[str, str]:
    """Return {question+answer -> caption} cache."""
    cache: Dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                cache[rec["key"]] = rec["caption"]
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def consolidate(jsonl_path: Path, json_path: Path) -> int:
    entries = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    json_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(entries)


def parse_args():
    p = argparse.ArgumentParser(description="Build VQA expansion dataset (API caption generation)")
    p.add_argument("--split", type=str, default="test", choices=["train", "test", "both"])
    p.add_argument("--provider", type=str, default=DEFAULT_PROVIDER,
                   help="Provider key in api_config.json (e.g. siliconflow)")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_ALIAS,
                   help="Model alias under the provider's 'models' map (or a raw model id)")
    p.add_argument("--api-config", type=str, default=str(config.API_CONFIG_PATH),
                   help="Path to api_config.json")
    p.add_argument("--limit", type=int, default=20,
                   help="Samples to process this run (0 = all). Default 20 (dry-run safety).")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--no-resume", action="store_true",
                   help="Start from scratch (overwrite existing outputs for the split)")
    return p.parse_args()


def run_split(split: str, args, generator: CaptionGenerator, out_dir: Path) -> Path:
    samples = load_split(split)
    cap_path = out_dir / f"{split}_captions.jsonl"   # caption cache (dedup + resume)
    exp_jsonl = out_dir / f"{split}_expansions.jsonl"
    exp_json = out_dir / f"{split}_expansions.json"

    if args.no_resume:
        for p in (cap_path, exp_jsonl, exp_json):
            if p.exists():
                p.unlink()

    cache = load_caption_cache(cap_path)
    done = count_lines(exp_jsonl)
    logger.info(f"[{split}] resume: {done} entries already written, "
                f"{len(cache)} captions cached")

    total = len(samples)
    limit = args.limit if (args.limit and args.limit > 0) else total
    end = min(total, done + limit)  # process [done, end)
    logger.info(f"[{split}] processing samples [{done}, {end}) of {total} "
                f"(--limit {args.limit})")

    # Open both files in append mode; flush each line for crash safety.
    with open(cap_path, "a", encoding="utf-8") as cf, \
         open(exp_jsonl, "a", encoding="utf-8") as ef:
        for i in range(done, end):
            filename, qtype, question, answer = samples[i]
            key = caption_cache_key(question, answer)
            caption = cache.get(key)
            if caption is None:
                caption = generator.generate(question, answer, qtype)
                cache[key] = caption
                cf.write(json.dumps(
                    {"key": key, "question": question, "answer": answer,
                     "type": qtype, "caption": caption},
                    ensure_ascii=False) + "\n")
                cf.flush()
            entry = build_entry(filename, qtype, question, answer, caption)
            ef.write(json.dumps(entry, ensure_ascii=False) + "\n")
            ef.flush()
            if (i - done + 1) % 1000 == 0:
                logger.info(f"[{split}] {i + 1}/{end} entries written")

    n = consolidate(exp_jsonl, exp_json)
    logger.info(f"[{split}] consolidated {n} entries -> {exp_json}")
    return exp_json


def main():
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else (config.OUTPUT_DIR / "vqa_expansions")
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = make_generator(args)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        run_split(split, args, generator, out_dir)

    logger.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "build_vqa_expansions failed")
        raise
