#!/usr/bin/env python
"""Inference-time set-aware I2T scoring test (zero training).

Premise of the paper: captions arrive in SETS (5 per image). At retrieval
time the gallery knows each caption's set, so a candidate caption t of image
j can also be scored via its set center:

    S(i, t) = cos(mu_v_i, mu_t) + beta * cos(mu_v_i, merged_j)

FAIRNESS: the CLIP baseline gets the identical trick (mean of its 5 caption
embeddings as the fusion term). Reports I2T R@1/5/10 for beta in {0, 0.3,
0.5, 1.0} on both models, coco test protocol.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.eval_common import build_eval_dataloader
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger("set_aware", config.LOG_DIR / "set_aware_scoring.log")


@torch.no_grad()
def extract(model, kind, loader, device, n):
    imgs, caps = [], []
    got = 0
    for batch in loader:
        if batch is None:
            continue
        pil, cl = batch["image"], batch["captions"]
        B, K = len(pil), len(cl[0])
        flat = [c for l in cl for c in l]
        ti = model.process_text(flat)
        if kind == "mcdisp":
            out = model(model.process_images(pil).to(device),
                        ti["input_ids"].to(device).view(B, K, -1),
                        ti["attention_mask"].to(device).view(B, K, -1))
            imgs.append(out["img_mu"].float().cpu())
            caps.append(out["text_mus"].float().cpu())
        else:  # clip
            im, tx = model(images=model.process_images(pil).to(device),
                           input_ids=ti["input_ids"].to(device),
                           attention_mask=ti["attention_mask"].to(device),
                           normalize=True)
            imgs.append(im.float().cpu())
            caps.append(tx.float().cpu().view(B, K, -1))
        got += B
        if got >= n:
            break
    return torch.cat(imgs), torch.cat(caps)


def i2t_recall(img_mu, cap_mus, betas):
    N, K, D = cap_mus.shape
    img_n = F.normalize(img_mu, dim=-1)
    cap_n = F.normalize(cap_mus.reshape(N * K, D), dim=-1)
    merged = F.normalize(F.normalize(cap_mus, dim=-1).mean(1), dim=-1)  # set centers
    C_cap = img_n @ cap_n.T                       # (N, N*K)
    C_set = img_n @ merged.T                      # (N, N)
    C_set_exp = C_set.repeat_interleave(K, dim=1) # align to caption columns
    out = {}
    for beta in betas:
        S = C_cap + beta * C_set_exp
        top = torch.argsort(-S, dim=1, stable=True)[:, :10]
        gi = torch.arange(N).unsqueeze(1)
        hits = ((top >= gi * K) & (top < gi * K + K))
        r = {k: hits[:, :k].any(dim=1).float().mean().item() for k in (1, 5, 10)}
        out[str(beta)] = r
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    set_seed(config.SEED)
    loader, _ = build_eval_dataloader("coco", batch_size=args.batch_size,
                                      num_workers=4, num_samples=args.num_samples)
    betas = [0.0, 0.3, 0.5, 1.0]

    from models.mcdisp_align_model import MCDispAlignModel
    from models.clip_baseline import CLIPFineTuneBaseline

    results = {}
    m = MCDispAlignModel()
    m.load("checkpoints/unfrozen_seed42/mcdisp_align_kl_coco_best.pt")
    m = m.to(args.device).eval()
    img, caps = extract(m, "mcdisp", loader, args.device, args.num_samples)
    results["mcdisp_kl_unfrozen_s42"] = i2t_recall(img, caps, betas)
    del m
    torch.cuda.empty_cache()

    c = CLIPFineTuneBaseline()
    c.load("checkpoints/seed42/clip_baseline_coco_best.pt")
    c = c.to(args.device).eval()
    img, caps = extract(c, "clip", loader, args.device, args.num_samples)
    results["clip_ft_s42"] = i2t_recall(img, caps, betas)

    for model, by_beta in results.items():
        for beta, r in by_beta.items():
            logger.info(f"{model} beta={beta}: R@1={r[1]:.4f} R@5={r[5]:.4f} R@10={r[10]:.4f}")
    out = Path("outputs/set_aware_scoring")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
