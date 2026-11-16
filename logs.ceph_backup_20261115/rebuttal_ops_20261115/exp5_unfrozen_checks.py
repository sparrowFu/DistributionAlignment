"""Read-only guards and output verification for the sequential exp5 jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import torch
from scripts.ablation_study.experiments import build_train_config

SHAS = {
    "train": "b76e417997c33426fe6b3de12220ac234df7c80af9b56f2838f059fee416a4b0",
    "dev": "d8fcf8e803a155d86fc12c5ba44b7ce2cea31838079897045d9cb83cb5cbe2a2",
    "test": "91a0234e8dc93148f7c7e0ea207601673d1884c0f51cd9278b0eaf214585e0c7",
}
OPS = ROOT / "logs/rebuttal_ops_20261115/exp5_unfrozen"
OPS.mkdir(exist_ok=True, parents=True)
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("preflight", "verify-train", "verify-eval", "smoke"))
ap.add_argument("experiment")
ap.add_argument("seed", type=int)
args = ap.parse_args()
cfg = build_train_config(args.experiment, args.seed, epochs=10, batch_size=32)
for split, sha in SHAS.items():
    p = ROOT / f"outputs/ablation_study/manifests/manifest_coco_{split}.json"
    if hashlib.sha256(p.read_bytes()).hexdigest() != sha or Path(str(p)+".sha256").read_text().strip() != sha:
        raise RuntimeError(f"Fixed {split} manifest checksum changed")

if args.mode == "preflight":
    # This process and ancestors are the scheduler, not competing project jobs.
    ancestors = {os.getpid()}
    p = os.getppid()
    while p > 1 and p not in ancestors:
        ancestors.add(p)
        try:
            p = int(Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError):
            break
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in ancestors:
            continue
        try:
            if (d / "cwd").resolve() != ROOT:
                continue
            command = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if "python" in command:
                raise RuntimeError(f"Another project task is active: {d.name} {command}")
        except (OSError, PermissionError):
            pass
    if shutil.disk_usage(cfg.best_path.parent).free < 16 * 1024**3:
        raise RuntimeError("Insufficient free disk for atomic checkpoint saves (<16GiB)")
    assert not cfg.freeze_clip and cfg.dev_num_captions == 5
    assert cfg.manifest_min_captions == 5 and cfg.manifest_sample_mode == "perm"
    assert cfg.loss_name == "kl" and cfg.no_early_stop
    record = {k: str(v) if isinstance(v, Path) else v for k,v in vars(cfg).items()}
    record.update(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), manifest_sha256=SHAS)
    tag = os.environ.get("REBUTTAL_STAGE", f"{args.experiment}_seed{args.seed}")
    (OPS / f"{tag}.config.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"preflight": "OK", "experiment": args.experiment, "seed": args.seed,
                      "K": cfg.manifest_num_captions, "best_path": str(cfg.best_path)}, indent=2))
elif args.mode == "verify-train":
    for p in (cfg.best_path, cfg.last_path):
        assert p.is_file() and p.stat().st_size > 1000000, p
        state = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        assert state.get("manifest_sha256") == SHAS, (p, "split identity")
        if p == cfg.last_path:
            assert state["epoch"] == 10 and state["seed"] == args.seed
            assert state["loss_name"] == "kl" and state["freeze_clip"] is False
        weights = state["model_state_dict"]
        for name, tensor in weights.items():
            if "clip" not in name and tensor.is_floating_point():
                assert bool(torch.isfinite(tensor).all()), (p, name, "nonfinite parameters")
        del state
    print("Training verified: epoch10, best+last readable, correct split hashes, finite head parameters")
elif args.mode == "smoke":
    from types import SimpleNamespace
    from scripts.ablation_study import run as driver
    driver.ROOT = ROOT / "outputs/rebuttal_exp5_20261115/smoke_unfrozen"
    driver.phase_eval(SimpleNamespace(device="cuda", seed=args.seed,
        checkpoint=str(cfg.best_path), experiment=args.experiment,
        batch_size=16, num_samples=64))
    print("Smoke evaluation saved separately from formal eval.json")
else:
    p = ROOT / f"outputs/ablation_study/{args.experiment}/seed{args.seed}/eval.json"
    result = json.loads(p.read_text())
    assert result["num_images"] == 11817 and result["K"] == 5
    split_check = result["split_check"]
    assert split_check["recorded"] == SHAS
    assert not split_check["mismatches"] and not split_check["missing_on_disk"]
    for kind in ("set_mc", "set_cos"):
        for k in (1, 5, 10):
            for direction in ("i2t", "t2i"):
                value = result["retrieval"][kind][f"{direction}_r@{k}"]
                assert 0 <= value <= 1, (kind, k, direction, value)
    print("Full evaluation verified: 11817 images, K5, matching split hashes, valid set retrieval recalls")
