"""Forensic diagnosis of the PCME++ sigma-branch runaway.

Compares two checkpoints (epoch-2 clean from the 5-epoch run; epoch-8 backup
from the crashed 25-epoch run) along the std pathway:

  image: block22 (split point) -> unc transformer (no LN) -> GPO sort-pool
         -> Linear -> std (logsigma, unbounded)
  text:  txt tower tokens -> Linear(768->1024) per token -> GPO sort-pool -> std

Prints parameter drift (Part 1, CPU) and activation statistics on real val
batches (Part 2, GPU, eval mode, fp32 + bf16 forward comparison).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.pcmepp_dataset import PCMEPPCaptionView, pcmepp_collate
from models.pcmepp_model import PCMEPPModel
from utils.dataset_factory import build_train_dataset
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger("diagnose_pcmepp_std", config.LOG_DIR / "diagnose_pcmepp_std.log")

CKPTS = {
    "epoch2(5ep-run,pre-explosion)": config.CHECKPOINT_DIR / "pcmepp_coco_last_epoch2_clean.pt",
    "epoch8(25ep-run,healthy)": config.CHECKPOINT_DIR / "pcmepp_coco_best_crashed25ep_run.pt",
}


def param_forensics(model: PCMEPPModel, tag: str):
    """Part 1: which parameter groups drifted between checkpoints."""
    rows = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not any(k in name for k in (
                    "image_std_encoder", "std_gpool", "std_linear",
                    "uncertainty_transformer", "criterion.")):
                continue
            rows.append((name, p.norm().item(), p.abs().max().item(), p.numel()))
    logger.info(f"--- Part1 param forensics [{tag}] ---")
    logger.info(f"{'parameter':<70}{'L2':>10}{'max|w|':>10}{'n':>10}")
    for name, l2, mx, n in rows:
        logger.info(f"{name:<70}{l2:>10.3f}{mx:>10.3f}{n:>10}")
    logger.info(f"criterion shift={model.criterion.shift.item():.4f} "
                f"negative_scale={model.criterion.negative_scale.item():.4f}")


@torch.no_grad()
def activation_forensics(model: PCMEPPModel, loader, device, tag, precision):
    """Part 2: activation statistics along the std pathway on val batches."""
    stats = {}
    hooks = []

    def mk_hook(key, transform):
        def hook(mod, inp, out):
            t = transform(out)
            stats.setdefault(key, []).append(
                (t.float().mean().item(),
                 t.float().pow(2).sum(-1).sqrt().max().item()  # max token/sample L2
                 if t.dim() >= 2 else t.float().abs().max().item()))
        return hook

    bb = model.img_enc.backbone
    hooks.append(bb.transformer.resblocks._modules["22"].register_forward_hook(
        mk_hook("img/block22-out(split)", lambda o: o)))
    hooks.append(bb.uncertainty_transformer.register_forward_hook(
        mk_hook("img/unc-transformer-out", lambda o: o["mean"])))
    hooks.append(model.img_enc.image_std_encoder.gpool.register_forward_hook(
        mk_hook("img/std-GPO-pooled", lambda o: o[0])))
    hooks.append(model.img_enc.image_std_encoder.fc.register_forward_hook(
        mk_hook("img/std=fc(pooled) LOGSIGMA", lambda o: o)))

    model.eval().to(device)
    n_batches = 0
    std_all, sigma_diag = [], []
    use_bf16 = precision == "bf16"
    for batch in loader:
        if n_batches >= 4:
            break
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        lengths = torch.Tensor(batch["lengths"]).to(device)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)
        with ctx:
            img_emb = model.img_enc(images)
            cap_emb = model.txt_enc(input_ids, lengths)
        img_emb = {k: v.float() for k, v in img_emb.items()}
        cap_emb = {k: v.float() for k, v in cap_emb.items()}
        std = img_emb["std"]
        std_all.append(std.flatten())
        # per-sample sigma contribution to a matched pair (diag):
        sigma_diag.append((img_emb["std"].exp() + cap_emb["std"].exp()).sum(-1))
        n_batches += 1
    for h in hooks:
        h.remove()

    logger.info(f"--- Part2 activations [{tag}] precision={precision} "
                f"batches={n_batches} x{images.size(0)} ---")
    for key, vals in stats.items():
        means = [v[0] for v in vals]
        maxs = [v[1] for v in vals]
        logger.info(f"{key:<34} mean(avg)={sum(means)/len(means):>12.3f} "
                    f"maxL2(max)={max(maxs):>12.3f}")
    s = torch.cat(std_all)
    qs = torch.tensor([0.5, 0.99, 0.999, 1.0], device=s.device)
    q = torch.quantile(s, qs)
    logger.info(f"img std(logsigma) dist: mean={s.mean():.3f} "
                f"p50={q[0]:.3f} p99={q[1]:.3f} p99.9={q[2]:.3f} max={q[3]:.3f}")
    sd = torch.cat(sigma_diag)
    logger.info(f"matched-pair sigma_pdist: p50={torch.quantile(sd, .5):.1f} "
                f"p99={torch.quantile(sd, .99):.1f} "
                f"p99.9={torch.quantile(sd, .999):.1f} max={sd.max():.1f} "
                f"(cliff ref: loss~0.014*ns*sigma)")


def main():
    set_seed(config.SEED)
    device = torch.device("cuda")

    logger.info("Building val loader (same split as train_pcmepp)...")
    full = build_train_dataset(dataset="coco", captions_path=None, images_dir=None)
    val_size = int(len(full) * 0.1)
    _, val = torch.utils.data.random_split(
        full, [len(full) - val_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED))
    loader = torch.utils.data.DataLoader(
        PCMEPPCaptionView(val, train=False), batch_size=16, shuffle=False,
        num_workers=4, collate_fn=pcmepp_collate)

    for tag, path in CKPTS.items():
        if not path.exists():
            logger.warning(f"missing {path}, skip")
            continue
        model = PCMEPPModel()
        model.load(str(path))
        param_forensics(model, tag)
        activation_forensics(model, loader, device, tag, "32-true")
        activation_forensics(model, loader, device, tag, "bf16")
        del model
        torch.cuda.empty_cache()
    logger.info("done")


if __name__ == "__main__":
    main()
