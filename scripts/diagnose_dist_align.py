"""One-off diagnostic: why is training-log R@1 ~0.004 while val NCE -> 0.75?

Loads checkpoints/dist_align_best.pt and measures, on a fixed val subset:
  - raw CLIP feature R@1 (sanity baseline — should be ~0.3+ if data/pipeline OK)
  - dist_align mu R@1 under THREE scorers: cosine, uncertainty-calibrated, loglik
  - positive-vs-negative similarity gap (what recall actually needs)
  - feature norms + sigma^2 (collapse / scale checks)

Run:
  python scripts/diagnose_dist_align.py --num-samples 2000
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import config
from data.caption_dataset import ImageCaptionDataset, filter_none_collate
from models.dist_align_model import DistributionAlignmentModel
from utils.retrieval import compute_recall_chunked, compute_recall_uc_chunked
from scripts.evaluate_dist_align import compute_recall_loglik_chunked


@torch.no_grad()
def extract(model, loader, device, num_samples):
    model.eval()
    g = []
    out = {k: [] for k in
           ["img_mu", "text_mu", "img_logvar", "text_logvar", "img_U",
            "img_feat", "text_feat"]}
    n = 0
    for batch in loader:
        if batch is None:
            continue
        pil = batch["image"]; caps = batch["captions"]
        B = len(pil); K = len(caps[0])
        pv = model.process_images(pil).to(device)
        flat = [c for cl in caps for c in cl]
        ti = model.process_text(flat)
        ids = ti["input_ids"].view(B, K, -1).to(device)
        am = ti["attention_mask"].view(B, K, -1).to(device)
        o = model(pv, ids, am)
        out["img_mu"].append(o["img_mu"].cpu())
        out["text_mu"].append(o["text_mu"].cpu())
        out["img_logvar"].append(o["img_logvar"].cpu())
        out["text_logvar"].append(o["text_logvar"].cpu())
        U = o["img_U"]
        out["img_U"].append(U.cpu() if U is not None else None)
        out["img_feat"].append(o["img_features"].cpu())
        out["text_feat"].append(o["text_features"].cpu())
        n += B
        if n >= num_samples:
            break
    cat = {}
    for k, v in out.items():
        if k == "img_U" and any(x is None for x in v):
            cat[k] = None
        else:
            cat[k] = torch.cat(v, dim=0)[:num_samples]
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/dist_align_best.pt")
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = DistributionAlignmentModel(
        freeze_clip=config.DIST_ALIGN_FREEZE_CLIP,
        distribution_merging=config.DIST_ALIGN_DISTRIBUTION_MERGING,
        cov_rank=config.MSDA_COV_RANK,
    )
    model.load(args.checkpoint)
    model = model.to(args.device)

    ds = ImageCaptionDataset(
        captions_path=config.CAPTIONS_PATH, images_dir=config.IMAGES_DIR,
        num_captions=config.NUM_CAPTIONS,
    )
    gen = torch.Generator().manual_seed(config.SEED)
    idx = torch.randperm(len(ds), generator=gen)[:args.num_samples].tolist()
    ds = Subset(ds, idx)
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=config.NUM_WORKERS, collate_fn=filter_none_collate)

    print(f"Extracting features on {args.num_samples} samples...")
    f = extract(model, loader, args.device, args.num_samples)

    img_mu, text_mu = f["img_mu"].to(args.device), f["text_mu"].to(args.device)
    img_lv, text_lv = f["img_logvar"].to(args.device), f["text_logvar"].to(args.device)
    img_U = f["img_U"].to(args.device) if f["img_U"] is not None else None
    img_feat, text_feat = f["img_feat"].to(args.device), f["text_feat"].to(args.device)

    print("\n===== NORM / SCALE =====")
    print(f"img_mu   norm: mean={img_mu.norm(dim=-1).mean():.3f} std={img_mu.norm(dim=-1).std():.3f}")
    print(f"text_mu  norm: mean={text_mu.norm(dim=-1).mean():.3f} std={text_mu.norm(dim=-1).std():.3f}")
    print(f"img_feat norm: mean={img_feat.norm(dim=-1).mean():.3f} (raw CLIP image proj)")
    print(f"text_feat norm: mean={text_feat.norm(dim=-1).mean():.3f} (raw CLIP text proj, avg over K)")
    print(f"sigma^2 img  mean: {torch.exp(img_lv).mean():.4f}  (training log ended ~0.53)")
    print(f"sigma^2 text mean: {torch.exp(text_lv).mean():.4f}")

    def pos_neg_gap(a, b):
        a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
        S = a @ b.T
        n = S.shape[0]
        pos = S.diag().mean().item()
        neg = (S.sum() - S.diag().sum()) / (n * n - n)
        return pos, neg.item()

    print("\n===== POS vs NEG COSINE GAP (what recall needs) =====")
    for name, a, b in [("dist_align mu", img_mu, text_mu),
                       ("raw CLIP feat", img_feat, text_feat)]:
        pos, neg = pos_neg_gap(a, b)
        print(f"  {name:16s}: pos={pos:.3f}  neg={neg:.3f}  gap={pos-neg:+.3f}")

    K = [1, 5, 10]
    print("\n===== RETRIEVAL R@1/5/10 =====")
    r_cos = compute_recall_chunked(img_mu, text_mu, K, normalize=True)
    print(f"  dist_align COSINE  : {[round(r_cos[k],3) for k in K]}   <- what the training log reports")
    try:
        r_uc = compute_recall_uc_chunked(img_mu, img_lv, text_mu, text_lv, K,
                                         temperature=config.MSDA_TAU)
        print(f"  dist_align UC-sim   : {[round(r_uc[f'uc_recall@{k}'],3) for k in K]}")
    except Exception as e:
        print(f"  dist_align UC-sim   : SKIPPED ({type(e).__name__}: pre-existing device bug in utils/retrieval.py compute_recall_uc_chunked)")
    r_ll = compute_recall_loglik_chunked(
        img_mu, img_lv, img_U, text_mu, K,
        per_dim_normalize=config.MSDA_PER_DIM_NORMALIZE,
        use_logdet=config.MSDA_USE_LOGDET)
    print(f"  dist_align LOGLIK   : i2t={[round(r_ll[f'loglik_i2t_recall@{k}'],3) for k in K]}  "
          f"t2i={[round(r_ll[f'loglik_t2i_recall@{k}'],3) for k in K]}  <- matches training objective")
    r_clip = compute_recall_chunked(img_feat, text_feat, K, normalize=True)
    print(f"  raw CLIP COSINE     : {[round(r_clip[k],3) for k in K]}   <- sanity baseline")


if __name__ == "__main__":
    main()
