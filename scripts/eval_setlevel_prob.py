"""Set-level retrieval eval for ProbVLM / PCME++ (paper-protocol gallery).

Same protocol as scripts/eval_setlevel_all.py: N images vs N caption sets,
text-set center = mean of the K caption mus, bidirectional cosine Recall@K.
Features come from each model's own extractor (evaluate_probvlm /
evaluate_pcmepp) so they match those scripts; the cosine scorer is the
cross-model comparable column (see eval_setlevel_all's docstring).
"""

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from models.pcmepp_model import PCMEPPModel  # noqa: E402
from models.probvlm_model import ProbVLMModel  # noqa: E402
from scripts.evaluate_pcmepp import extract_features as extract_pcmepp  # noqa: E402
from scripts.evaluate_probvlm import extract_features as extract_probvlm  # noqa: E402
from utils.eval_common import build_eval_dataloader  # noqa: E402
from utils.eval_results import append_eval_results  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.retrieval import compute_recall_bidirectional  # noqa: E402
from utils.seed import set_seed  # noqa: E402

logger = get_logger("eval_setlevel_prob", config.LOG_DIR / "eval_setlevel_prob.log")


def _per_k(flat, prefix, ks):
    """Flat {prefix}_i2t@{k} dict -> nested {k: {i2t, t2i}}."""
    return {k: {"i2t": flat[f"{prefix}_i2t@{k}"],
                "t2i": flat[f"{prefix}_t2i@{k}"]} for k in ks}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=["probvlm", "pcmepp"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="coco", choices=["coco", "flickr"])
    ap.add_argument("--num-samples", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(config.SEED)
    ks = list(config.RECALL_AT_K)

    dataloader, n = build_eval_dataloader(
        args.dataset, batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS, num_samples=args.num_samples)
    logger.info(f"=== {args.model} set-level eval (dataset={args.dataset}, N={n}) ===")

    if args.model == "probvlm":
        model, extract = ProbVLMModel(), extract_probvlm
    else:
        model, extract = PCMEPPModel(), extract_pcmepp
    model.load(args.checkpoint)
    model = model.to(args.device)

    img_mu, _img_lv, text_mus, _text_lvs = extract(
        model, dataloader, args.device, args.num_samples)

    # set center = mean caption mu; cosine is the cross-model comparable column
    center = text_mus.mean(dim=1)
    cos = compute_recall_bidirectional(
        img_mu.to(args.device), center.to(args.device), ks, normalize=True)
    per_k = _per_k(cos, "recall", ks)

    metrics = {}
    for k, comps in per_k.items():
        metrics[f"cos_i2t@{k}"] = comps["i2t"]
        metrics[f"cos_t2i@{k}"] = comps["t2i"]
    append_eval_results(args.output_path, {
        "variant": args.model,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "num_samples": n,
        "protocol": "set-level N-vs-N (image vs caption-set center, mean caption mu)",
        "metrics": metrics,
    }, logger)

    header = f"{'variant':<12}" + "".join(f"{'i2t@' + str(k):>9}{'t2i@' + str(k):>9}" for k in ks)
    cells = "".join(f"{per_k[k][d]:>9.4f}" for k in ks for d in ("i2t", "t2i"))
    print(f"\n=== set-level Recall ({args.dataset}, N={n}) ===\n{header}\n{'-' * len(header)}\n{args.model:<12}{cells}\n")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
