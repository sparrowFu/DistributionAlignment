"""Rebuttal exp 4: caption complementarity x hard negatives on a verified subset.

Input: ONE human-verified subset JSON (checklist §4.1-4.2), shared by every
model. The tool only CONSUMES it -- bin assignment and hard-negative edits are
human work; selection happens before any model result is looked at.

    {
      "entries": [
        {
          "image": "000000xxxxxx.jpg",
          "bin": "low" | "mid" | "high",          # 互补性分档
          "captions": ["...", x5],
          "hard_negatives": [                      # 最小事实对照,可为空
            {"caption": "...", "replaces": 2, "note": "红伞->蓝伞"}
          ]
        }
      ]
    }

Outputs (per model, per bin):
  1. set retrieval under the COMMON centroid cosine (N images vs N centers,
     mean of the 5 caption embeddings, same protocol as eval_clip_setlevel):
     set I2T / T2I R@1; Full - Set-FT delta per bin when both tags exist.
  2. hard-negative checks in the same cosine space (CLIP models):
     score(true) - score(hard) win rate, and T2I top-1 flips 错->对 / 对->错
     when the true caption is swapped for its hard negative.
  3. Mahalanobis checks (MCDisp models): true-caption coverage rate at m_pos,
     hard-negative false-coverage rate, and margin = min_neg d2 - max_true d2
     (「最近负例距离减最远真描述距离」).

Usage:
  python scripts/exp4_complementarity.py --subset subset.json \\
      --clip setft=checkpoints/clip_setft_coco_best.pt \\
      --clip clip=checkpoints/clip_baseline_coco_best.pt \\
      --mcdisp full=outputs/ablation_study/full_kl/seed42/best.pt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from utils.logger import get_logger  # noqa: E402

from losses.mcdisp_align_losses import MCDispAlignLoss  # noqa: E402
from models.mcdisp_align_model import MCDispAlignModel  # noqa: E402
from scripts.ablation_study.feature_extraction import extract_features as mcdisp_feats
from scripts.ablation_study.h2_coverage import positive_mahalanobis
from scripts.evaluate_clip_baseline import extract_features as clip_feats
from models.clip_baseline import CLIPFineTuneBaseline

logger = get_logger("exp4_complementarity",
                    config.LOG_DIR / "exp4_complementarity.log")

BINS = ("low", "mid", "high")


# ---------------------------------------------------------------- data plumbing
def _subset_manifest_path(subset: Dict, workdir: Path) -> Path:
    """The subset AS a manifest, so both feature extractors see one file."""
    workdir.mkdir(parents=True, exist_ok=True)
    p = workdir / "subset_manifest.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"entries": [
            {"image": e["image"], "captions": e["captions"]} for e in subset["entries"]
        ]}, f, ensure_ascii=False)
    return p


def _loader(manifest: Path, batch_size: int):
    from torch.utils.data import DataLoader
    from data.manifest_caption_dataset import ManifestCaptionDataset
    from data.caption_dataset import filter_none_collate
    ds = ManifestCaptionDataset(manifest, config.IMAGES_DIR,
                                num_captions=5, sample_mode="first")
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=2, collate_fn=filter_none_collate)


def _assert_aligned(n_feats: int, subset: Dict):
    if n_feats != len(subset["entries"]):
        raise RuntimeError(
            f"feature/image misalignment: {n_feats} encoded vs "
            f"{len(subset['entries'])} subset entries (a broken image was "
            f"dropped by collate). Fix the subset file -- the verified set "
            f"must load completely.")


def _collect_pairs(entries: List[Dict]):
    """[(image_idx, entry_idx_in_hard_list, hard_dict)] flattened."""
    pairs = []
    for i, e in enumerate(entries):
        for h in e.get("hard_negatives", []):
            pairs.append((i, h))
    return pairs


def _per_bin(entries: List[Dict], values: Dict[int, float]) -> Dict[str, Dict[str, float]]:
    """Mean of per-image ``values`` inside each bin (one value per image)."""
    out = {}
    for b in BINS:
        vals = [v for i, v in values.items() if entries[i]["bin"] == b]
        out[b] = {"n": len(vals), "mean": sum(vals) / len(vals) if vals else float("nan")}
    return out


# ---------------------------------------------------------------- set retrieval
def set_recall_by_bin(img: torch.Tensor, centers: torch.Tensor,
                      entries: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Centroid-cosine set retrieval (N vs N, diagonal positives), per bin."""
    q = F.normalize(img.float(), dim=-1)
    g = F.normalize(centers.float(), dim=-1)
    sim = q @ g.T
    i2t_top1 = sim.argmax(dim=1)
    t2i_top1 = sim.argmax(dim=0)
    bins = [e["bin"] for e in entries]
    out = {}
    for b in BINS + ("overall",):
        idx = [i for i, s in enumerate(bins) if s == b] if b != "overall" \
            else list(range(len(bins)))
        if not idx:
            continue
        t = torch.tensor(idx)
        out[b] = {
            "n_images": len(idx),
            "set_i2t_r1": float((i2t_top1[t] == t).float().mean()),
            "set_t2i_r1": float((t2i_top1[t] == t).float().mean()),
        }
    return out


# ---------------------------------------------------------------- CLIP family
def _encode_texts_clip(model, captions: List[str], device) -> torch.Tensor:
    ti = model.process_text(list(captions))
    f = model.encode_text(ti["input_ids"].to(device),
                          ti["attention_mask"].to(device), normalize=True)
    return f.float().cpu()


def hard_negative_checks_clip(model, img_f: torch.Tensor, text_mus: torch.Tensor,
                              entries: List[Dict], device) -> Dict[str, float]:
    """共同余弦空间的最小事实对照: score diff / win rate / T2I top-1 flips."""
    img_n = F.normalize(img_f.float(), dim=-1)                   # (N, D)
    cap_n = F.normalize(text_mus.float(), dim=-1)                # (N, K, D)
    own = torch.einsum("nd,nkd->nk", img_n, cap_n)               # (N, K)
    t2i = torch.einsum("nkd,md->nkm", cap_n, img_n)              # (N, K, N)

    pairs = _collect_pairs(entries)
    if not pairs:
        return {"n_pairs": 0}
    h_n = F.normalize(_encode_texts_clip(model, [h["caption"] for _, h in pairs],
                                         device), dim=-1)
    h_cos = h_n @ img_n.T                                         # (P, N)

    diff_per_image: Dict[int, float] = {}
    wins, to_right, to_wrong = 0, 0, 0
    for p, (i, h) in enumerate(pairs):
        j = int(h.get("replaces", 0))
        s_true = float(own[i, j])
        s_hard = float(h_cos[p, i])
        win = s_true > s_hard
        wins += win
        diff_per_image[i] = diff_per_image.get(i, 0.0) + (s_true - s_hard)

        rank_true = int((t2i[i, j] > t2i[i, j, i]).sum())         # 0 = top-1
        h_t2i = h_n[p] @ img_n.T                                  # (N,)
        rank_hard = int((h_t2i > h_t2i[i]).sum())
        to_right += rank_true > 0 and rank_hard == 0              # 错->对
        to_wrong += rank_true == 0 and rank_hard > 0              # 对->错

    n = len(pairs)
    images_with_pairs = sorted(diff_per_image)
    return {
        "n_pairs": n,
        "win_rate": wins / n,
        "mean_score_diff": sum(diff_per_image.values()) / n,
        "flips_wrong_to_right": to_right,
        "flips_right_to_wrong": to_wrong,
        "score_diff_by_bin": _per_bin(
            entries, {i: v / max(1, sum(1 for ii, _ in pairs if ii == i))
                      for i, v in diff_per_image.items()}),
    }


def clip_model_results(tag: str, ckpt: Path, subset: Dict, manifest: Path,
                       device: str, batch_size: int) -> Dict:
    model = CLIPFineTuneBaseline()
    model.load(str(ckpt))
    model = model.to(device).eval()
    img_f, text_mus = clip_feats(model, _loader(manifest, batch_size), device)
    _assert_aligned(img_f.shape[0], subset)
    return {
        "checkpoint": str(ckpt),
        "set_recall": set_recall_by_bin(img_f, text_mus.mean(dim=1), subset["entries"]),
        "hard_negatives": hard_negative_checks_clip(
            model, img_f, text_mus, subset["entries"], device),
    }


# ---------------------------------------------------------------- MCDisp family
def _encode_text_mus_mcdisp(model, captions: List[str], device) -> torch.Tensor:
    """Text-only per-caption mu, same pipeline as MCDispAlignModel.forward."""
    ti = model.process_text(list(captions))
    tf = model.clip_model.get_text_features(
        input_ids=ti["input_ids"].to(device),
        attention_mask=ti["attention_mask"].to(device)).pooler_output
    return model.text_mu_head(tf).float().cpu()


def mahalanobis_checks_mcdisp(model, feats: Dict[str, torch.Tensor],
                              entries: List[Dict], device,
                              m_pos: float) -> Dict:
    """真描述覆盖率 / 难负误覆盖率 / 最近负例-最远真描述 margin."""
    pairs = _collect_pairs(entries)
    img_mu, img_var = feats["img_mu"], torch.exp(feats["img_logvar"])
    img_U = feats["img_U"]
    D = img_mu.shape[-1]
    d2_true = positive_mahalanobis(img_mu, img_var, img_U, feats["text_mus"])  # (N, K)

    d2_hard: Dict[int, List[float]] = {}
    if pairs:
        h_mus = _encode_text_mus_mcdisp(model, [h["caption"] for _, h in pairs], device)
        for p, (i, h) in enumerate(pairs):
            diff = (h_mus[p] - img_mu[i]).unsqueeze(0)
            U_i = img_U[i].unsqueeze(0) if img_U is not None else None
            d2 = float(MCDispAlignLoss._mahalanobis(
                diff, img_var[i].unsqueeze(0), U_i, 1e-6)[0] / D)
            d2_hard.setdefault(i, []).append(d2)

    true_rate = (d2_true <= m_pos).float().mean(dim=1)           # (N,)
    margin_per_image: Dict[int, float] = {}
    false_rate_per_image: Dict[int, float] = {}
    for i, ds in d2_hard.items():
        false_rate_per_image[i] = sum(1 for d in ds if d <= m_pos) / len(ds)
        margin_per_image[i] = min(ds) - float(d2_true[i].max())

    out = {
        "m_pos": m_pos,
        "true_coverage_rate": float(true_rate.mean()),
        "true_coverage_by_bin": _per_bin(entries, {i: float(true_rate[i])
                                                   for i in range(len(entries))}),
        "n_pairs": len(pairs),
    }
    if pairs:
        out.update({
            "hard_false_coverage_rate": (
                sum(false_rate_per_image.values()) / len(false_rate_per_image)),
            "hard_false_coverage_by_bin": _per_bin(entries, false_rate_per_image),
            "margin_min_neg_minus_max_true_mean": (
                sum(margin_per_image.values()) / len(margin_per_image)),
            "margin_by_bin": _per_bin(entries, margin_per_image),
            "frac_margin_positive": (
                sum(1 for v in margin_per_image.values() if v > 0)
                / len(margin_per_image)),
        })
    return out


def mcdisp_model_results(tag: str, ckpt: Path, subset: Dict, manifest: Path,
                         device: str, batch_size: int, m_pos: float) -> Dict:
    feats = mcdisp_feats(ckpt, manifest, num_captions=5,
                         batch_size=batch_size, device=device)
    _assert_aligned(feats["img_mu"].shape[0], subset)
    model = MCDispAlignModel()
    model.load(str(ckpt))
    model = model.to(device).eval()
    return {
        "checkpoint": str(ckpt),
        "set_recall": set_recall_by_bin(feats["img_mu"],
                                        feats["text_mus"].mean(dim=1),
                                        subset["entries"]),
        "mahalanobis": mahalanobis_checks_mcdisp(
            model, feats, subset["entries"], device, m_pos),
    }


# ---------------------------------------------------------------- aggregation / CLI
def _delta_per_bin(a: Dict, b: Dict) -> Dict[str, Dict[str, float]]:
    """a - b per bin for every recall key both have."""
    out = {}
    for binn, row in a.items():
        if binn not in b:
            continue
        out[binn] = {k: row[k] - b[binn][k] for k in row
                     if isinstance(row.get(k), float) and k in b[binn]}
    return out


def main():
    ap = argparse.ArgumentParser(description="Rebuttal exp 4: complementarity x hard negatives")
    ap.add_argument("--subset", required=True, help="human-verified subset JSON")
    ap.add_argument("--clip", action="append", default=[], metavar="TAG=PATH",
                    help="CLIP-family checkpoint (clip_baseline / clip_setft)")
    ap.add_argument("--mcdisp", action="append", default=[], metavar="TAG=PATH",
                    help="MCDisp-family checkpoint (full / no_cover / diag ...)")
    ap.add_argument("--m-pos", type=float, default=config.MCDISP_ALIGN_M_POS)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    with open(args.subset, "r", encoding="utf-8") as f:
        subset = json.load(f)
    for e in subset["entries"]:
        if e.get("bin") not in BINS:
            raise ValueError(f"entry {e['image']}: bin must be one of {BINS}")
        K = len(e["captions"])
        for h in e.get("hard_negatives", []):
            j = int(h.get("replaces", 0))
            if not 0 <= j < K:
                raise ValueError(
                    f"entry {e['image']}: hard negative 'replaces'={j} "
                    f"outside caption range [0, {K})")
    workdir = Path(args.subset).parent / "exp4_work"
    manifest = _subset_manifest_path(subset, workdir)

    results: Dict[str, Dict] = {"subset": args.subset,
                                "n_entries": len(subset["entries"]),
                                "clip": {}, "mcdisp": {}}
    for spec in args.clip:
        tag, path = spec.split("=", 1)
        logger.info(f"CLIP model {tag} <- {path}")
        results["clip"][tag] = clip_model_results(
            tag, Path(path), subset, manifest, args.device, args.batch_size)
    for spec in args.mcdisp:
        tag, path = spec.split("=", 1)
        logger.info(f"MCDisp model {tag} <- {path}")
        results["mcdisp"][tag] = mcdisp_model_results(
            tag, Path(path), subset, manifest, args.device, args.batch_size,
            args.m_pos)

    # Full - Set-FT per-bin delta (checklist 结果表 column)
    if "full" in results["mcdisp"] and "setft" in results["clip"]:
        results["full_minus_setft_by_bin"] = _delta_per_bin(
            results["mcdisp"]["full"]["set_recall"],
            results["clip"]["setft"]["set_recall"])

    out = Path(args.output) if args.output else (
        workdir / "exp4_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"exp4 results -> {out}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
