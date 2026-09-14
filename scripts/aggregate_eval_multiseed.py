#!/usr/bin/env python
"""Aggregate multi-seed retrieval-eval JSONs into mean±std summary tables.

Reads the outputs of scripts/run_eval_multiseed.sh (outputs/eval_multiseed/):

  recall_{model}_{dataset}_seed{S}.json   -- evaluate_*.py records, flat keys
  allhit_{model}_{dataset}_seed{S}.json   -- eval_allhit.py records

Two tables are produced:

  Table 1 (unified multi-caption protocol): N images vs N*K captions,
      any-hit I2T + per-caption T2I under the SAME cosine scorer for every
      model (family ``mc_cos_recall``), plus AllHit@5 and cover-rank mean
      ("mean coverage"; from eval_allhit, already the unified protocol).
  Table 2 (legacy 1:1 protocol, for comparability with old runs): CLIP/ProLIP
      first-caption metrics (``csd_recall``/``recall``), MCDisp-Align
      set-level N-vs-N on the merged caption center (``mcdisp_align_recall``).

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

MODEL_KEYS = ("clip", "prolip", "mcdisp", "mcdisp_kl", "mcdisp_kl_unfrozen",
              "clip_zero_shot", "prolip_zero_shot")
DISPLAY = {
    "clip": "CLIP fine-tuned", "prolip": "ProLIP fine-tuned",
    "mcdisp": "MCDisp-Align (std)", "mcdisp_kl": "MCDisp-Align (KL)",
    "mcdisp_kl_unfrozen": "MCDisp-Align (KL, unfrozen)",
    "clip_zero_shot": "CLIP zero-shot", "prolip_zero_shot": "ProLIP zero-shot",
}
# per-model family for the LEGACY table (unified table is always mc_cos_recall)
LEGACY_PREF = {
    "clip": ["recall"], "clip_zero_shot": ["recall"],
    "prolip": ["csd_recall", "cos_recall", "recall"],
    "prolip_zero_shot": ["csd_recall", "cos_recall", "recall"],
    "mcdisp": ["mcdisp_align_recall"], "mcdisp_kl": ["mcdisp_align_recall"],
}
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


def recall_families(metrics: dict):
    return {k.rsplit("_i2t@", 1)[0] for k in metrics if k.endswith("_i2t@1")}


def pick(fams, prefs):
    for p in prefs:
        if p in fams:
            return p
    return sorted(fams)[0] if fams else None


def collect(out_dir: Path, coverage_dir: Path = None):
    """{(model, dataset): {store_key: {metric: [per-seed values]}}}

    store_key: "unified" | "legacy" | "allhit". `coverage_dir` (optional)
    adds the coverage-evaluation records (evaluate_mcdisp_coverage.py),
    whose mc_cos_recall family is the same unified protocol -- reported as
    the unfrozen MCDisp-Align rows.
    """
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if coverage_dir is not None:
        for path in sorted(coverage_dir.glob("coverage_*_seed*.json")):
            m = re.match(r"coverage_(.+?)_(coco|flickr)_seed(\d+)\.json$", path.name)
            if not m:
                continue
            rec = last_record(path)
            if rec is None or "metrics" not in rec:
                continue
            fams = recall_families(rec["metrics"])
            if "mc_cos_recall" not in fams:
                continue
            model_key = {"kl_unfrozen": "mcdisp_kl_unfrozen"}.get(m.group(1), m.group(1))
            key = (model_key, m.group(2))
            for k in KS:
                for d in ("i2t", "t2i"):
                    v = rec["metrics"].get(f"mc_cos_recall_{d}@{k}")
                    if v is not None:
                        cells[key]["unified"][f"{d}_R@{k}"].append(v)
    for path in sorted(out_dir.glob("*.json")):
        parsed = parse_name(path)
        if not parsed:
            continue
        side, model, ds, _seed = parsed
        rec = last_record(path)
        if rec is None or "metrics" not in rec:
            continue
        metrics = rec["metrics"]

        if side == "recall":
            fams = recall_families(metrics)
            unified = pick(fams, ["mc_cos_recall"])
            legacy = pick(fams, LEGACY_PREF.get(model, ["recall"]))
            for store, fam in (("unified", unified), ("legacy", legacy)):
                if fam is None:
                    continue
                for k in KS:
                    for d in ("i2t", "t2i"):
                        v = metrics.get(f"{fam}_{d}@{k}")
                        if v is not None:
                            cells[(model, ds)][store][f"{d}_R@{k}"].append(v)
        else:  # allhit side: prefer the model-native score family
            fams = {k.rsplit("_allhit@", 1)[0] for k in metrics if "_allhit@" in k}
            fam = pick(fams, {"mcdisp": ["mcdisp"], "mcdisp_kl": ["mcdisp"],
                              "prolip": ["csd", "cos"],
                              "prolip_zero_shot": ["csd", "cos"]}.get(model, ["cos"]))
            if fam is None:
                continue
            v = metrics.get(f"{fam}_allhit@5")
            if v is not None:
                cells[(model, ds)]["allhit"]["AllHit@5"].append(v)
            v = metrics.get(f"{fam}_coverrank_mean@100")
            if v is not None:
                cells[(model, ds)]["allhit"]["CoverRank"].append(v)
    return cells


def mean_std(vals):
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def render_table(cells, title, col_specs, note=""):
    """col_specs: [(header, [(store, key), ...fallbacks], is_rank2), ...]"""
    headers = [c[0] for c in col_specs]
    lines = [f"## {title}", ""]
    if note:
        lines += [note, ""]
    lines += ["| 模型 | 数据集 | seeds | " + " | ".join(headers) + " |",
              "|---|---|" + "---:|" * (len(headers) + 1)]
    for (model, ds) in sorted(cells, key=lambda x: (
            x[1], MODEL_KEYS.index(x[0]) if x[0] in MODEL_KEYS else 99)):
        stores = cells[(model, ds)]
        row_vals, nmax = [], 0
        for _h, sources, rank2 in col_specs:
            vs = None
            for store, key in sources:
                vs = stores.get(store, {}).get(key)
                if vs:
                    break
            if not vs:
                row_vals.append("—")
                continue
            m, s = mean_std(vs)
            nmax = max(nmax, len(vs))
            fmt = "{:.2f}±{:.2f}" if rank2 else "{:.4f}±{:.4f}"
            row_vals.append(fmt.format(m, s))
        if all(v == "—" for v in row_vals):
            continue
        lines.append("| " + " | ".join(
            [DISPLAY.get(model, model), ds, str(nmax) if nmax else "—"] + row_vals) + " |")
    return lines


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/eval_multiseed")
    cov_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "outputs/coverage_eval")
    if not out_dir.exists():
        sys.exit(f"no eval output dir: {out_dir}")
    cells = collect(out_dir, cov_dir if cov_dir.exists() else None)
    if not cells:
        sys.exit(f"no eval JSONs found under {out_dir}")

    lines = ["# 多种子检索评估汇总（mean±std across seeds）", ""]
    rc = [(f"I2T R@{k}", [("unified", f"i2t_R@{k}")], False) for k in KS] \
        + [(f"T2I R@{k}", [("unified", f"t2i_R@{k}")], False) for k in KS]
    rc_allhit = rc + [("AllHit@5", [("allhit", "AllHit@5")], False),
                      ("CoverRank", [("allhit", "CoverRank")], True)]
    lc = [(f"I2T R@{k}", [("legacy", f"i2t_R@{k}")], False) for k in KS] \
        + [(f"T2I R@{k}", [("legacy", f"t2i_R@{k}")], False) for k in KS]

    lines += render_table(
        cells,
        "表 1 · 统一全描述协议（每图全部 5 条 caption 参与检索；N 图 vs N×K 描述候选库；"
        "any-hit I2T / 逐 caption T2I；全模型同一 cosine 打分）",
        rc_allhit,
        note="CoverRank = 覆盖整图全部 5 条描述的平均检索深度（越小越好）；"
             "AllHit@5 = top-5 恰好为全部自身描述的比例。")
    lines.append("")
    lines += render_table(
        cells,
        "表 2 · 原协议对照（CLIP/ProLIP 仅第一条 caption 的 1:1 检索；"
        "MCDisp 为集合中心 N vs N）",
        lc,
        note="仅用于与历史结果对比，协议不一致，不应跨模型直接比较。")

    table = "\n".join(lines)
    (out_dir / "summary.md").write_text(table + "\n")
    print(table)
    print(f"\nwritten: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
