#!/usr/bin/env python
"""Aggregate multi-seed retrieval-eval JSONs into a mean±std summary table.

Reads the outputs of scripts/run_eval_multiseed.sh (outputs/eval_multiseed/):

  recall_{model}_{dataset}_seed{S}.json   -- evaluate_*.py records, flat keys
  allhit_{model}_{dataset}_seed{S}.json   -- eval_allhit.py records

Model-native metric family is preferred per model (mcdisp_align score for
MCDisp-Align, csd for ProLIP, cosine for CLIP), falling back to whatever
recall family the record contains. Reported per (model, dataset) across seeds:

  I->T R@1/5/10, T->I R@1/5/10, AllHit@5, cover-rank mean (mean coverage depth)

Usage: python scripts/aggregate_eval_multiseed.py [outputs/eval_multiseed]
"""

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: F401 (paths only)

MODEL_KEYS = ("clip", "prolip", "mcdisp", "mcdisp_kl",
              "clip_zero_shot", "prolip_zero_shot")
DISPLAY = {
    "clip": "CLIP fine-tuned", "prolip": "ProLIP fine-tuned",
    "mcdisp": "MCDisp-Align (std)", "mcdisp_kl": "MCDisp-Align (KL)",
    "clip_zero_shot": "CLIP zero-shot", "prolip_zero_shot": "ProLIP zero-shot",
}
# score-family preference per model (recall side / allhit side)
RECALL_PREF = {"mcdisp": ["mcdisp_align_recall"], "mcdisp_kl": ["mcdisp_align_recall"],
               "prolip": ["csd_recall", "cos_recall", "recall"],
               "default": ["recall", "cos_recall"]}
ALLHIT_PREF = {"mcdisp": ["mcdisp"], "mcdisp_kl": ["mcdisp"],
               "prolip": ["csd", "cos"], "default": ["cos"]}
KS = (1, 5, 10)


def last_record(path: Path):
    """Load the LAST record of an append-style JSON file."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, list) and data:
        return data[-1]
    if isinstance(data, dict):
        return data
    return None


def parse_name(path: Path):
    """recall_clip_coco_seed42.json -> ('recall', 'clip', 'coco', 42)."""
    m = re.match(r"(recall|allhit)_(.+?)_(coco|flickr)(?:_seed(\d+))?\.json$",
                 path.name)
    return m.groups() if m else None


def pick_family(metrics: dict, pref_key: str, side: str):
    """Resolve the metric family actually present in a flat-metrics dict."""
    prefs = (RECALL_PREF if side == "recall" else ALLHIT_PREF).get(pref_key) \
        or (RECALL_PREF if side == "recall" else ALLHIT_PREF)["default"]
    if side == "recall":
        fams = {k.rsplit("_i2t@", 1)[0] for k in metrics if k.endswith("_i2t@1")}
    else:
        fams = {k.rsplit("_allhit@", 1)[0] for k in metrics if "_allhit@" in k}
    for p in prefs:
        if p in fams:
            return p
    return sorted(fams)[0] if fams else None


def collect(out_dir: Path):
    """{(model, dataset): {metric_name: [per-seed values]}}"""
    cells = defaultdict(lambda: defaultdict(list))
    for path in sorted(out_dir.glob("*.json")):
        parsed = parse_name(path)
        if not parsed:
            continue
        side, model, ds, seed = parsed
        rec = last_record(path)
        if rec is None or "metrics" not in rec:
            continue
        metrics = rec["metrics"]
        fam = pick_family(metrics, model, side)
        if fam is None:
            continue
        if side == "recall":
            for k in KS:
                for d in ("i2t", "t2i"):
                    v = metrics.get(f"{fam}_{d}@{k}")
                    if v is not None:
                        cells[(model, ds)][f"{d}_R@{k}"].append(v)
        else:
            for k, key in ((5, f"{fam}_allhit@5"), (None, f"{fam}_coverrank_mean@100")):
                v = metrics.get(key)
                if v is not None:
                    name = "AllHit@5" if k else "CoverRank"
                    cells[(model, ds)][name].append(v)
    return cells


def mean_std(vals):
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/eval_multiseed")
    if not out_dir.exists():
        sys.exit(f"no eval output dir: {out_dir}")
    cells = collect(out_dir)
    if not cells:
        sys.exit(f"no eval JSONs found under {out_dir}")

    cols = [f"I2T R@{k}" for k in KS] + [f"T2I R@{k}" for k in KS] \
        + ["AllHit@5", "CoverRank"]
    lines = ["# 多种子检索评估汇总（mean±std across seeds）", "",
             f"来源：`{out_dir}`（各模型原生打分：MCDisp=uncertainty-discounted，"
             "ProLIP=CSD，CLIP=cosine；CoverRank 为覆盖整个 caption 集的平均检索深度，越小越好）", "",
             "| 模型 | 数据集 | seeds | " + " | ".join(cols) + " |",
             "|---|---|" + "---:|" * (len(cols) + 1)]

    for (model, ds) in sorted(cells, key=lambda x: (x[1], MODEL_KEYS.index(x[0])
                             if x[0] in MODEL_KEYS else 99)):
        vals = cells[(model, ds)]
        row = [DISPLAY.get(model, model), ds]
        # map friendly column names to stored keys
        keymap = {f"I2T R@{k}": f"i2t_R@{k}" for k in KS}
        keymap.update({f"T2I R@{k}": f"t2i_R@{k}" for k in KS})
        keymap["AllHit@5"] = "AllHit@5"
        keymap["CoverRank"] = "CoverRank"
        for c in cols:
            vs = vals.get(keymap[c])
            if not vs:
                row.append("—")
                continue
            m, s = mean_std(vs)
            row.append(f"{m:.4f}±{s:.4f}" if c != "CoverRank" else f"{m:.2f}±{s:.2f}")
        n = max((len(v) for v in vals.values()), default=0)
        row.insert(2, str(n))
        lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines)
    (out_dir / "summary.md").write_text(table + "\n")
    print(table)
    print(f"\nwritten: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
