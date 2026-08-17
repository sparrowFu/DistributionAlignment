"""Phase 0 of the ablation plan: data audit + immutable image-exclusive manifests.

Builds ``train``/``dev``/``test`` manifest JSONs from the MSCOCO caption parquet
(``dataset="coco"``), one entry per image so an image can only land in one split
(plan §3.1). Each manifest stores the TRUE valid captions (empty/non-string
dropped, exact duplicates deduplicated -- plan phase-0 item 3) plus ``n_valid``,
and a ``.sha256`` checksum file for reproducibility. An audit report JSON records
caption-count histograms, duplicate/missing-image statistics, and split sizes.

Output layout::

    {out_dir}/manifest_{dataset}_{split}.json      (split in train/dev/test)
    {out_dir}/manifest_{dataset}_{split}.json.sha256
    {out_dir}/audit_report_{dataset}.json
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import config
from utils.logger import get_logger


logger = get_logger("ablation_data_audit")


def _load_coco_entries(captions_path: Path, images_dir: Path, limit: Optional[int]) -> tuple:
    """Read the parquet into per-image entries with true valid captions."""
    import pandas as pd

    data = pd.read_parquet(captions_path)
    data = data[data["image_file_name"].notna()].reset_index(drop=True)

    entries: List[Dict] = []
    missing_images, dup_caption_entries, dup_captions_total = 0, 0, 0
    short_entries = 0
    for row in data.itertuples(index=False):
        raw = getattr(row, "caption", None)
        if isinstance(raw, str):
            raw = [raw]
        elif not isinstance(raw, (list, tuple)):
            try:
                raw = list(raw)
            except (TypeError, ValueError):
                raw = []
        caps = [c for c in raw if isinstance(c, str) and c.strip()]
        n_raw = len(caps)
        # Exact-duplicate captions carry zero semantic spread (plan §3.2: only
        # valid AND non-duplicate captions may enter spread/subspace targets).
        seen, dedup = set(), []
        for c in caps:
            if c not in seen:
                seen.add(c)
                dedup.append(c)
        if len(dedup) < n_raw:
            dup_caption_entries += 1
            dup_captions_total += n_raw - len(dedup)
        if not dedup:
            short_entries += 1
            continue
        if not (images_dir / row.image_file_name).exists():
            missing_images += 1
            continue
        entries.append({"image": str(row.image_file_name), "captions": dedup, "n_valid": len(dedup)})
        if limit is not None and len(entries) >= limit:
            break

    logger.info(
        f"Loaded {len(entries)} entries: {missing_images} missing images dropped, "
        f"{short_entries} without any valid caption, "
        f"{dup_caption_entries} entries had exact duplicates ({dup_captions_total} dup captions removed)"
    )
    audit = {"missing_images": missing_images, "entries_with_dup_captions": dup_caption_entries,
             "dup_captions_removed": dup_captions_total, "entries_no_valid_caption": short_entries}
    return entries, audit


def _write_manifest(path: Path, dataset: str, split: str, entries: List[Dict], seed: int) -> str:
    blob = {
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "num_entries": len(entries),
        "entries": entries,
    }
    payload = json.dumps(blob, ensure_ascii=False, sort_keys=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    with open(str(path) + ".sha256", "w", encoding="utf-8") as f:
        f.write(digest + "\n")
    logger.info(f"Wrote {path} ({len(entries)} entries, sha256={digest[:12]}…)")
    return digest


def build_manifests(
    dataset: str = "coco",
    out_dir: Optional[Path] = None,
    seed: int = config.SEED,
    train_frac: float = 0.8,
    dev_frac: float = 0.1,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    """Build the three image-exclusive manifests + audit report. Returns the report.

    The remaining ``1 - train_frac - dev_frac`` fraction is the test split. All
    ablation variants MUST consume these exact files (verified by checksum) so
    that train/dev/test are identical across configs (plan §3.1).
    """
    if dataset != "coco":
        raise ValueError(f"Manifest building currently supports dataset='coco', got {dataset!r}")
    out_dir = Path(out_dir) if out_dir else config.OUTPUT_DIR / "ablation_study" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries, audit = _load_coco_entries(config.CAPTIONS_PATH, config.IMAGES_DIR, limit)

    rng = random.Random(seed)
    rng.shuffle(entries)
    n = len(entries)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)
    splits = {
        "train": entries[:n_train],
        "dev": entries[n_train:n_train + n_dev],
        "test": entries[n_train + n_dev:],
    }

    hist = Counter(min(e["n_valid"], 10) for e in entries)
    report: Dict[str, object] = {
        "dataset": dataset,
        "seed": seed,
        "limit": limit,
        "total_entries": n,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "caption_count_histogram_capped_at_10": {str(k): hist[k] for k in sorted(hist)},
        "entries_below_5_captions": sum(1 for e in entries if e["n_valid"] < 5),
        "cleaning": audit,
        "manifest_sha256": {},
    }
    for split, split_entries in splits.items():
        path = out_dir / f"manifest_{dataset}_{split}.json"
        report["manifest_sha256"][split] = _write_manifest(path, dataset, split, split_entries, seed)

    report_path = out_dir / f"audit_report_{dataset}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Audit report -> {report_path}")
    return report


def verify_manifest(path: Path) -> bool:
    """Recompute a manifest's checksum and compare with its .sha256 sidecar."""
    path = Path(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.exists():
        return False
    return digest == sidecar.read_text().strip()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ablation phase 0: build manifests + audit")
    ap.add_argument("--dataset", default="coco")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--limit", type=int, default=None, help="cap entries (smoke tests)")
    args = ap.parse_args()
    build_manifests(args.dataset, args.out_dir, args.seed, limit=args.limit)
