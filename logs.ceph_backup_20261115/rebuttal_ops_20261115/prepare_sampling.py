"""Validate immutable exp5 splits and sample exp4 annotation candidates.

Runs on the training server. Never assigns bins or invents hard negatives.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

import torch
import config
from data.manifest_caption_dataset import ManifestCaptionDataset
from scripts.ablation_study.data_audit import verify_manifest
from scripts.ablation_study.experiments import build_train_config
from scripts.exp4_complementarity import make_subset_template, legacy_split_overlap
from utils.dataset_factory import build_train_dataset

OUT = ROOT / "outputs/rebuttal_exp4_20261115"
OUT.mkdir(parents=True, exist_ok=True)
MANIFESTS = ROOT / "outputs/ablation_study/manifests"


def save_new(path, value):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


report = {"commit": subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True).strip(), "splits": {}}
splits, ids = {}, {}
for split in ("train", "dev", "test"):
    path = MANIFESTS / f"manifest_coco_{split}.json"
    if not verify_manifest(path):
        raise ValueError(f"Invalid manifest checksum: {path}")
    blob = json.loads(path.read_text(encoding="utf-8"))
    entries = blob["entries"]
    names = [e["image"] for e in entries]
    if len(names) != len(set(names)):
        raise ValueError(f"Repeated image IDs in {split}")
    splits[split], ids[split] = entries, set(names)
    report["splits"][split] = {
        "raw_images": len(entries),
        "at_least_5_valid_distinct_captions": sum(
            len({c for c in e["captions"] if isinstance(c, str) and c.strip()}) >= 5
            for e in entries),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
    overlap = ids[a] & ids[b]
    if overlap:
        raise ValueError(f"Image leakage between {a} and {b}: {len(overlap)}")
report["split_pairs_disjoint"] = True

# Exercise the actual dataset/config path without any model updates.
datasets, configs = {}, {}
for k in (1, 3, 5):
    name = f"full_kl_k{k}"
    cfg = build_train_config(name, 42)
    ds = ManifestCaptionDataset(cfg.train_manifest, config.IMAGES_DIR,
        num_captions=cfg.manifest_num_captions,
        sample_mode=cfg.manifest_sample_mode,
        min_valid_captions=cfg.manifest_min_captions,
        perm_seed=cfg.manifest_perm_seed)
    datasets[k] = ds
    configs[name] = {"seed": cfg.seed, "epochs": cfg.epochs,
        "batch_size": cfg.batch_size, "freeze_clip": cfg.freeze_clip,
        "train_K": cfg.manifest_num_captions, "dev_K": cfg.dev_num_captions,
        "sample_mode": cfg.manifest_sample_mode, "min_valid": cfg.manifest_min_captions,
        "perm_seed": cfg.manifest_perm_seed, "cov_rank": cfg.cov_rank,
        "lambda_kl": cfg.lambda_kl, "lambda_cover_pos": cfg.lambda_cover_pos,
        "lambda_cov": cfg.lambda_cov, "lambda_reg": cfg.lambda_reg,
        "select_by": cfg.select_by, "no_early_stop": cfg.no_early_stop,
        "effective_train_images": len(ds), "best_path": str(cfg.best_path)}
base = datasets[5].entries
for k in (1, 3):
    ds = datasets[k]
    if len(ds) != len(base):
        raise ValueError("K groups use different image counts")
    for x, y in zip(ds.entries, base):
        if x["image"] != y["image"] or x["order"] != y["order"]:
            raise ValueError("K groups do not share image order and caption permutation")
report["exp5_configs"] = configs
report["caption_permutations_checked_for_all_training_images"] = True

# Reconstruct the ORIGINAL 90/10 split using the same loader and RNG.
# The 80/10/10 audit test alone is unsafe for evaluating old checkpoints.
legacy = build_train_dataset("coco")
n = len(legacy)
n_val = int(n * 0.1)
train_sub, val_sub = torch.utils.data.random_split(
    legacy, [n - n_val, n_val], generator=torch.Generator().manual_seed(42))
legacy_train = set(legacy.data.loc[list(train_sub.indices), "image_file_name"])
legacy_val = set(legacy.data.loc[list(val_sub.indices), "image_file_name"])
if legacy_train & legacy_val:
    raise ValueError("Legacy train/validation image IDs overlap")
candidates = [e for e in splits["test"] if e["image"] in legacy_val
              and e["image"] not in legacy_train
              and len(e["captions"]) >= 5]
provenance = {
    "sampling_rule": "audit_test intersect reconstructed legacy_validation_seed42; >=5 distinct valid captions",
    "selected_before_model_outputs": True,
    "legacy_seed": 42, "legacy_train_images": len(legacy_train),
    "legacy_validation_images": len(legacy_val),
    "eligible_candidates": len(candidates),
    "legacy_train_overlap": 0,
    "scope": "Old-checkpoint validation diagnostics; NOT an independent unseen test. These images were used for legacy checkpoint selection. Audit-test status applies to future models trained only on audit train.",
    "source_parquet_sha256": hashlib.sha256(Path(config.CAPTIONS_PATH).read_bytes()).hexdigest(),
    "source_audit_test_sha256": report["splits"]["test"]["sha256"],
}
if len(candidates) < 90:
    raise ValueError(f"Only {len(candidates)} leakage-screened annotation candidates")
pool_path = OUT / "annotation_candidate_manifest.json"
save_new(pool_path, {"provenance": provenance, "entries": candidates})
template_path = OUT / "subset_template.json"
make_subset_template(pool_path, 90, 42, template_path)
template = json.loads(template_path.read_text(encoding="utf-8"))
template["provenance"] = provenance
# The newly created, unannotated file is enriched once, never regenerated.
template_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
overlap = legacy_split_overlap([e["image"] for e in template["entries"]], seed=42)
if overlap["n_overlap"]:
    raise ValueError("Sample unexpectedly overlaps legacy training")
for e in template["entries"]:
    if not (Path(config.IMAGES_DIR) / e["image"]).is_file():
        raise FileNotFoundError(e["image"])
save_new(OUT / "legacy_overlap.json", overlap)
report["exp4_sampling"] = {**provenance, "sampled_images": 90,
    "subset_template": str(template_path),
    "annotation_status": "awaiting_human_review_no_bins_or_negatives_fabricated"}
report["missing_checkpoints"] = [
    name for name, path in {
        "clip_ft": ROOT / "checkpoints/clip_baseline_coco_best.pt",
        "no_cover": ROOT / "outputs/ablation_study/no_cover_kl/seed42/abl_no_cover_kl_kl_coco_best.pt",
        "diag_only": ROOT / "outputs/ablation/checkpoints/diag_only/best.pt",
    }.items() if not path.exists()]
save_new(OUT / "preparation_report.json", report)
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
