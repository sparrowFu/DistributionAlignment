"""
GaussianImageDistribution - Unified retrieval-eval output.

Every retrieval-eval script describes what it computed as a list of "metric
groups", then calls:

  - print_recall_groups(groups, logger): the terminal table.
  - groups_to_flat(groups):            the flat metrics dict for the JSON record.
  - append_eval_results(path, record): append the record to the results JSON
                                       (a list, never overwriting prior runs),
                                       stamping a `time` field right after
                                       `dataset`.

This keeps the print format and the on-disk schema identical across all models
(mcdisp_align, clip_baseline, prolip, and the two zero-shot variants).

Group shape::

    {"family": "mcdisp_align_recall",                     # flat-key prefix
     "label":  "MCDisp_Align-score Recall@K (primary)",   # terminal header
     "per_k":  {1: {"i2t": .6, "t2i": .6, "mean": .6},   # or {"value": .5}
                5: {...}, ...}}

Flat-key convention (matches the existing mcdisp_align_eval_results.json)::

    mean          -> {family}@{k}        e.g. mcdisp_align_recall@1
    i2t           -> {family}_i2t@{k}    e.g. mcdisp_align_recall_i2t@1
    t2i           -> {family}_t2i@{k}    e.g. mcdisp_align_recall_t2i@1
    single value  -> {family}@{k}        e.g. recall@1
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import get_logger


logger = get_logger("eval_results")


def now_timestamp() -> str:
    """Current local time as 'YYYY-MM-DD HH:MM:SS'."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_recall_groups(groups: List[Dict], log=None) -> None:
    """Print each group: a header line, then one ``R@K`` line per K.

    A group with ``i2t``/``t2i``/``mean`` prints ``R@K: i2t=.. t2i=.. mean=..``;
    a group with only ``value`` prints ``R@K: <value>``.
    """
    log = log or logger
    for g in groups:
        log.info(g["label"])
        for k in sorted(g["per_k"]):
            comps = g["per_k"][k]
            if "value" in comps:
                log.info(f"  R@{k}: {comps['value']:.4f}")
                continue
            parts = []
            if "i2t" in comps:
                parts.append(f"i2t={comps['i2t']:.4f}")
            if "t2i" in comps:
                parts.append(f"t2i={comps['t2i']:.4f}")
            if "mean" in comps:
                parts.append(f"mean={comps['mean']:.4f}")
            log.info(f"  R@{k}: " + " ".join(parts))


def groups_to_flat(groups: List[Dict]) -> Dict[str, float]:
    """Flatten groups into the metrics dict saved to JSON (see key convention)."""
    flat: Dict[str, float] = {}
    for g in groups:
        fam = g["family"]
        for k, comps in g["per_k"].items():
            if "value" in comps:
                flat[f"{fam}@{k}"] = comps["value"]
                continue
            if "i2t" in comps:
                flat[f"{fam}_i2t@{k}"] = comps["i2t"]
            if "t2i" in comps:
                flat[f"{fam}_t2i@{k}"] = comps["t2i"]
            if "mean" in comps:
                flat[f"{fam}@{k}"] = comps["mean"]
    return flat


def _order_record(record: Dict) -> Dict:
    """Return a copy of ``record`` with ``time`` placed immediately after ``dataset``.

    Stamps ``time`` (= now) if missing/empty. If there is no ``dataset`` key,
    ``time`` is appended at the end.
    """
    record = dict(record)
    if not record.get("time"):
        record["time"] = now_timestamp()

    ordered: Dict = {}
    placed = False
    for key, val in record.items():
        if key == "time":
            continue
        ordered[key] = val
        if key == "dataset":
            ordered["time"] = record["time"]
            placed = True
    if not placed:
        ordered["time"] = record["time"]
    return ordered


def append_eval_results(path, record: Dict, log=None) -> None:
    """Append ``record`` to the JSON results file as a list.

    On-disk shape is always a JSON array:

      - missing file                 -> ``[record]``
      - existing list                -> ``existing + [record]``
      - existing single object (old) -> ``[old_object, record]``

    ``record`` is normalized via :func:`_order_record` first (``time`` after
    ``dataset``).
    """
    log = log or logger
    path = Path(path)
    record = _order_record(record)

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = None
        if isinstance(existing, list):
            records = existing
        elif isinstance(existing, dict):
            records = [existing]
        else:
            records = []
    else:
        records = []

    records.append(record)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    log.info(f"Results appended to {path} (now {len(records)} record(s))")
