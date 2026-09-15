#!/usr/bin/env python
"""OOD-detection PILOT (main-experiment feasibility check, no retraining).

Three probabilistic models, one score family: mean sigma^2 per image as the
OOD score (higher = more anomalous). In-domain: MSCOCO eval subset. OOD:
SVHN / CIFAR-10 test splits (downloaded via torchvision into
TrainDatasets/ood). Reports AUROC per (model, ood set).

    MCDisp-Align (KL, seed42)   sigma supervised by caption dispersion
    ProLIP fine-tuned (seed42)  sigma learned freely
    ProLIP zero-shot            reference
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.calibration import compute_auroc
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger("ood_pilot", config.LOG_DIR / "ood_pilot.log")

CLIP_NORM = transforms.Normalize(mean=[0.48145457, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])


def ood_loader(name, n, data_dir, batch_size):
    tf = transforms.Compose([transforms.Resize((224, 224)),
                             transforms.ToTensor(), CLIP_NORM])
    if name == "svhn":
        ds = datasets.SVHN(root=str(data_dir), split="test", download=True, transform=tf)
    elif name == "cifar10":
        ds = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=tf)
    else:
        raise ValueError(name)
    if n < len(ds):
        g = np.random.RandomState(0)
        ds = Subset(ds, g.choice(len(ds), n, replace=False).tolist())
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)


@torch.no_grad()
def sigma_scores(model_key, model, loader, device):
    """mean sigma^2 per image for either model family."""
    model.eval()
    out = []
    for batch in tqdm(loader, desc=f"sigma[{model_key}]", leave=False):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch["image"]
        if isinstance(batch, (list, tuple)):
            feats = model.clip_model.get_image_features(images.to(device)).pooler_output \
                if model_key.startswith("mcdisp") else None
            if model_key.startswith("mcdisp"):
                logvar = model._floor_logvar(model.img_logvar_head(feats))
            else:
                logvar = model.encode_images(images.to(device), normalize=False)["std"]
        else:
            pil = batch["image"]
            if model_key.startswith("mcdisp"):
                pv = model.process_images(pil).to(device)
                feats = model.clip_model.get_image_features(pv).pooler_output
                logvar = model._floor_logvar(model.img_logvar_head(feats))
            else:
                logvar = model.encode_images(
                    model.process_images(pil).to(device), normalize=False)["std"]
        out.append(torch.exp(logvar).mean(dim=-1).float().cpu())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    set_seed(0)

    data_dir = Path("TrainDatasets/ood")
    data_dir.mkdir(parents=True, exist_ok=True)

    # in-domain: coco eval subset images only
    from utils.eval_common import build_eval_dataloader
    in_loader, _ = build_eval_dataloader("coco", batch_size=args.batch_size,
                                         num_workers=4, num_samples=args.num_samples)
    from models.mcdisp_align_model import MCDispAlignModel
    from models.prolip_model import ProLIPModel

    models = {
        "mcdisp_kl_s42": ("mcdisp", MCDispAlignModel()),
        "prolip_ft_s42": ("prolip", ProLIPModel()),
        "prolip_zero_shot": ("prolip", ProLIPModel(freeze=True)),
    }
    models["mcdisp_kl_s42"][1].load("checkpoints/seed42/mcdisp_align_kl_coco_best.pt")
    models["prolip_ft_s42"][1].load("checkpoints/seed42/prolip_coco_best.pt")

    ood_sets = {name: ood_loader(name, args.num_samples, data_dir, args.batch_size)
                for name in ("svhn", "cifar10")}

    results = {}
    for key, (fam, model) in models.items():
        model = model.to(args.device)
        s_in = sigma_scores(key, model, in_loader, args.device)
        results[key] = {"in_mean": float(s_in.mean())}
        for name, loader in ood_sets.items():
            s_ood = sigma_scores(key, model, loader, args.device)
            # compute_auroc expects CONFIDENCE (higher = in-domain); sigma-bar
            # is an ANOMALY score, so pass its negation.
            auroc = compute_auroc(-s_in, -s_ood)
            results[key][f"auroc_{name}"] = float(auroc)
            logger.info(f"{key} | {name}: AUROC={auroc:.4f} "
                        f"(in {s_in.mean():.4f} ood {s_ood.mean():.4f})")
        del model
        torch.cuda.empty_cache()

    out = Path("outputs/ood_pilot")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    logger.info(f"written: {out / 'results.json'}")


if __name__ == "__main__":
    main()
