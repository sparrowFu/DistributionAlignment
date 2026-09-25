"""Set-level retrieval under EQUAL aggregation for CLIP (zero-shot and FT).

Answers: with the same protocol MCDisp uses at set level -- N images vs N
aggregated text centers (mean of the 5 caption embeddings), cosine ranking --
how do zero-shot CLIP and fine-tuned CLIP compare? MCDisp's own eval reports
the same protocol via its moment-matched center (= arithmetic mean of caption
means), so these numbers are directly comparable.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from models.clip_baseline import CLIPFineTuneBaseline
from scripts.evaluate_clip_baseline import extract_features
from utils.dataset_factory import VALID_DATASETS  # noqa: F401
from utils.eval_common import build_eval_dataloader
from utils.retrieval import compute_recall_bidirectional
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger("eval_clip_setlevel", config.LOG_DIR / "eval_clip_setlevel.log")


def evaluate_variant(model, tag, dataloader, device, num_samples, ks):
    img_f, text_mus = extract_features(model, dataloader, device, num_samples)
    # equal-condition aggregation: mean of the K caption embeddings (what
    # moment matching reduces to on the mean path)
    text_center = text_mus.mean(dim=1)
    bidir = compute_recall_bidirectional(
        img_f, text_center, ks, chunk_size=1000, normalize=True)
    return {
        "model": tag,
        "protocol": "set-level N-vs-N, mean-of-5-captions, cosine",
        "per_k": {
            k: {"i2t": bidir[f"recall_i2t@{k}"],
                "t2i": bidir[f"recall_t2i@{k}"],
                "mean": bidir[f"recall@{k}"]}
            for k in ks
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="CLIP-FT checkpoint; skip FT eval when omitted")
    parser.add_argument("--include-zeroshot", action="store_true")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(config.SEED)
    results = []
    dataloader, n = build_eval_dataloader(
        args.dataset, batch_size=config.EVAL_BATCH_SIZE,
        num_workers=config.NUM_WORKERS, num_samples=args.num_samples)

    if args.include_zeroshot:
        zs = CLIPFineTuneBaseline(
            freeze_image=True, freeze_text=True)
        zs = zs.to(args.device)
        results.append(evaluate_variant(
            zs, "CLIP ViT-L/14 zero-shot", dataloader, args.device,
            args.num_samples, config.RECALL_AT_K))
        del zs

    if args.checkpoint:
        ft = CLIPFineTuneBaseline()
        ft.load(args.checkpoint)
        ft = ft.to(args.device)
        results.append(evaluate_variant(
            ft, "CLIP ViT-L/14 fine-tuned", dataloader, args.device,
            args.num_samples, config.RECALL_AT_K))

    out = {"dataset": args.dataset, "num_samples": n,
           "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "results": results}
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved {args.output_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
