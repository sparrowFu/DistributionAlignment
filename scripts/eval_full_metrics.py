#!/usr/bin/env python
"""Full metric suite for the frozen-head MCDisp experiments (and baselines).

One extraction, then on the FULL test gallery (N images vs N*K captions):
  - unified bidirectional Recall@1/5/10 (cosine on means)
  - AllHit@5        : top-5 retrieved captions are exactly the image's own 5
  - mean-coverage   : average retrieval depth (max rank over the image's own
                      K captions, 1-based; lower is better)
  - dispersion      : within-image caption cosine distance (inside),
                      cross-image (outside), and inside/outside ratio
                      (text-only; baselines can be measured identically)

Usage:
  python scripts/eval_full_metrics.py --tag bal_s42 --model mcdisp \
      --checkpoint checkpoints/bal_seed42/mcdisp_align_kl_coco_best.pt --dataset coco
  python scripts/eval_full_metrics.py --tag clip_s42 --model clip \
      --checkpoint checkpoints/seed42/clip_baseline_coco_best.pt --dataset coco
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

logger = get_logger("eval_full_metrics", config.LOG_DIR / "eval_full_metrics.log")


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
            imgs.append(out["img_mu"].float())
            caps.append(out["text_mus"].float())
        elif kind == "clip":
            im, tx = model(images=model.process_images(pil).to(device),
                           input_ids=ti["input_ids"].to(device),
                           attention_mask=ti["attention_mask"].to(device),
                           normalize=False)
            imgs.append(im.float())
            caps.append(tx.float().view(B, K, -1))
        else:  # prolip
            out = model(model.process_images(pil).to(device),
                        ti["input_ids"].to(device))
            imgs.append(out["img_mu"].float())
            caps.append(out["text_mu"].float().view(B, K, -1))
        got += B
        if got >= n:
            break
    return torch.cat(imgs).cpu(), torch.cat(caps).cpu()


def full_metrics(img_mu, cap_mus, device):
    N, K, D = cap_mus.shape
    M = N * K
    img_n = F.normalize(img_mu, dim=-1)
    cap_n = F.normalize(cap_mus.reshape(M, D), dim=-1)
    C = (img_n.to(device) @ cap_n.to(device).T)

    # MCDisp's own inference criterion may hubness-correct the cosine
    # (CSLS, k=5); report under the "c_" prefix alongside plain cosine.
    r_img = C.topk(5, dim=1).values.mean(1, keepdim=True)
    r_cap = C.topk(5, dim=0).values.mean(0, keepdim=True)
    Cc = 2 * C - r_img - r_cap

    def _recalls(S):
        t10 = torch.topk(S, 10, dim=1).indices
        gi_ = torch.arange(N, device=device).unsqueeze(1)
        ow = (t10 >= gi_ * K) & (t10 < gi_ * K + K)
        t10c = torch.topk(S, 10, dim=0).indices.T
        out = {}
        for k in (1, 5, 10):
            out[f"i2t@{k}"] = ow[:, :k].any(1).float().mean().item()
            out[f"t2i@{k}"] = (t10c[:, :k] == cids.unsqueeze(1)).any(1).float().mean().item()
        return out

    cids = torch.arange(N, device=device).repeat_interleave(K)
    res = _recalls(C)
    cres = _recalls(Cc)
    for k, v in cres.items():
        res[f"c_{k}"] = v

    # AllHit@5 + mean-coverage (need ranks up to M for the farthest caption)
    def _coverage(S):
        all_idx = torch.argsort(S, dim=1, descending=True)
        ranks = torch.empty_like(all_idx)
        q = torch.arange(M, device=device).unsqueeze(0).expand(N, -1)
        ranks.scatter_(1, all_idx, q)
        ar = torch.arange(N, device=device)
        own = ranks[ar.unsqueeze(1), (ar * K).unsqueeze(1) + torch.arange(K, device=device).unsqueeze(0)]
        return (own.max(1).values <= 5).float().mean().item(), (own.max(1).values + 1).float().mean().item()

    res["allhit@5"], res["mean_coverage"] = _coverage(C)
    res["c_allhit@5"], res["c_mean_coverage"] = _coverage(Cc)

    # dispersion (text-only, cosine distance)
    cap_flat = cap_n
    inside, outside = [], []
    g = torch.Generator().manual_seed(0)
    for i in torch.randperm(N, generator=g)[:1000]:
        sel = cap_flat[i * K:(i + 1) * K].to(device)
        sims = sel @ sel.T
        inside.append((1 - sims)[torch.triu(torch.ones(K, K, device=device), 1) > 0].mean().item())
    for _ in range(2000):
        i, j = torch.randint(N, (1,), generator=g).item(), torch.randint(N, (1,), generator=g).item()
        if i == j:
            continue
        a = cap_flat[(i * K + torch.randint(K, (1,), generator=g)).item()].to(device)
        b = cap_flat[(j * K + torch.randint(K, (1,), generator=g)).item()].to(device)
        outside.append((1 - (a @ b)).item())
    res["inside_dist"] = sum(inside) / len(inside)
    res["outside_dist"] = sum(outside) / len(outside)
    res["in_out_ratio"] = res["inside_dist"] / max(res["outside_dist"], 1e-9)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--model", choices=["mcdisp", "clip", "prolip"], default="mcdisp")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", choices=["coco", "flickr"], default="coco")
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    set_seed(config.SEED)

    if args.model == "mcdisp":
        from models.mcdisp_align_model import MCDispAlignModel
        model = MCDispAlignModel()
    elif args.model == "clip":
        from models.clip_baseline import CLIPFineTuneBaseline
        model = CLIPFineTuneBaseline()
    else:
        from models.prolip_model import ProLIPModel
        model = ProLIPModel()
    model.load(args.checkpoint)
    model = model.to(args.device).eval()

    loader, n_avail = build_eval_dataloader(args.dataset, batch_size=args.batch_size,
                                            num_workers=4, num_samples=args.num_samples)
    n = min(args.num_samples, n_avail)
    img_mu, cap_mus = extract(model, args.model, loader, args.device, n)
    del model
    torch.cuda.empty_cache()

    res = full_metrics(img_mu, cap_mus, args.device)
    logger.info("[%s] %s", args.tag, json.dumps({k: round(v, 4) for k, v in res.items()}))

    out = Path("outputs/full_metrics")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}_{args.dataset}.json").write_text(json.dumps({
        "tag": args.tag, "model": args.model, "checkpoint": args.checkpoint,
        "dataset": args.dataset, "num_samples": n, "metrics": res,
    }, indent=2))


if __name__ == "__main__":
    main()
