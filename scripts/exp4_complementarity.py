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
  1. set retrieval under the COMMON centroid cosine. Protocol order is FIXED
     for every model family: average the RAW caption features, then L2-
     normalize the center -- i.e. norm(mean_k f_k), which is both the Set-FT
     training aggregation and the MCDisp moment-matching mean path. For CLIP
     models a SECONDARY "normed_mean" table (mean of per-caption NORMALIZED
     features, the legacy eval_clip_setlevel protocol) is also reported; the
     two differ and must not be mixed in one table.
  2. hard-negative checks in the same cosine space, for BOTH families
     (CLIP features / MCDisp mu): score(true) - score(hard) win rate, and
     per-model "text-swap" T2I top-1 flips when the true caption is swapped
     for its hard negative (named text_swap_* -- this is a within-model
     sensitivity readout, NOT model-vs-model correction).
  3. model-vs-model correction counts on set I2T top-1 (错->对 / 对->错 per
     bin) for every ordered pair of provided models.
  4. Mahalanobis checks (MCDisp models): true-caption coverage rate at m_pos,
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
    """[(image_idx, hard_dict)] flattened over all entries."""
    pairs = []
    for i, e in enumerate(entries):
        for h in e.get("hard_negatives", []):
            pairs.append((i, h))
    return pairs


def _per_bin(entries: List[Dict], values: Dict[int, float]) -> Dict[str, Dict[str, float]]:
    """Mean of per-image ``values`` inside each bin."""
    out = {}
    for b in BINS:
        vals = [v for i, v in values.items() if entries[i]["bin"] == b]
        out[b] = {"n": len(vals), "mean": sum(vals) / len(vals) if vals else float("nan")}
    return out


# ---------------------------------------------------------------- set retrieval
def set_top1_vectors(img: torch.Tensor, centers: torch.Tensor):
    """Set-level N-vs-N top-1 hits (raw inputs; normalized inside).

    Returns (i2t_hit (N,) bool, t2i_hit (N,) bool) -- one query per image on
    each side at set level, directly comparable across model families.
    """
    q = F.normalize(img.float(), dim=-1)
    g = F.normalize(centers.float(), dim=-1)
    sim = q @ g.T
    idx = torch.arange(sim.shape[0])
    return sim.argmax(dim=1) == idx, sim.argmax(dim=0) == idx


def set_recall_by_bin(img: torch.Tensor, centers: torch.Tensor,
                      entries: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Centroid-cosine set retrieval (N vs N, diagonal positives), per bin."""
    i2t_hit, t2i_hit = set_top1_vectors(img, centers)
    out = {}
    for b in BINS + ("overall",):
        idx = [i for i, e in enumerate(entries) if e["bin"] == b] if b != "overall" \
            else list(range(len(entries)))
        if not idx:
            continue
        t = torch.tensor(idx)
        out[b] = {
            "n_images": len(idx),
            "set_i2t_r1": float(i2t_hit[t].float().mean()),
            "set_t2i_r1": float(t2i_hit[t].float().mean()),
        }
    return out


# ---------------------------------------------------------------- cosine features
def _clip_raw_features(model, loader, device):
    """RAW (unnormalized) CLIP features: img (N, D), captions (N, K, D).

    normalize=False is essential: the common protocol averages RAW caption
    features before the single normalization (norm(mean(f_k))); the shared
    eval extractor normalizes per caption, which gives a different center.
    """
    imgs, caps = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            B, K = len(batch["captions"]), len(batch["captions"][0])
            flat = [c for cl in batch["captions"] for c in cl]
            pixel_values = model.process_images(batch["image"]).to(device)
            ti = model.process_text(flat)
            img_f = model.encode_image(pixel_values, normalize=False)
            cap_f = model.encode_text(ti["input_ids"].to(device),
                                      ti["attention_mask"].to(device),
                                      normalize=False)
            imgs.append(img_f.float().cpu())
            caps.append(cap_f.float().cpu().view(B, K, -1))
    return torch.cat(imgs), torch.cat(caps)


def _encode_texts_clip(model, captions: List[str], device) -> torch.Tensor:
    """RAW caption features (see _clip_raw_features for why normalize=False)."""
    ti = model.process_text(list(captions))
    f = model.encode_text(ti["input_ids"].to(device),
                          ti["attention_mask"].to(device), normalize=False)
    return f.float().cpu()


def _encode_text_mus_mcdisp(model, captions: List[str], device) -> torch.Tensor:
    """Text-only per-caption mu, same pipeline as MCDispAlignModel.forward."""
    ti = model.process_text(list(captions))
    tf = model.clip_model.get_text_features(
        input_ids=ti["input_ids"].to(device),
        attention_mask=ti["attention_mask"].to(device)).pooler_output
    return model.text_mu_head(tf).float().cpu()


# ---------------------------------------------------------------- hard negatives
def hard_negative_checks_cosine(img_raw: torch.Tensor, cap_raw: torch.Tensor,
                                hard_raw: torch.Tensor,
                                entries: List[Dict]) -> Dict:
    """最小事实对照 in the common cosine space (any feature family).

    ``text_swap_*`` counts are WITHIN-MODEL, deliberately NEUTRAL readouts:
    they say whether the model's T2I top-1 for the image is gained or lost
    when the true caption query is replaced by its hard negative. Top-1
    "gained" via a factually-wrong caption is NOT model correction -- model
    correction (错->对 / 对->错 between models) lives in ``model_contrasts``.
    """
    img_n = F.normalize(img_raw.float(), dim=-1)                  # (N, D)
    cap_n = F.normalize(cap_raw.float(), dim=-1)                  # (N, K, D)
    own = torch.einsum("nd,nkd->nk", img_n, cap_n)                # (N, K)
    t2i = torch.einsum("nkd,md->nkm", cap_n, img_n)               # (N, K, N)
    hard_n = F.normalize(hard_raw.float(), dim=-1)                # (P, D)

    pairs = _collect_pairs(entries)
    diff_per_image: Dict[int, float] = {}
    count_per_image: Dict[int, int] = {}
    pair_wins = []          # per pair: s_true > s_hard, aligned with _collect_pairs
    swap_wrong_to_right, swap_right_to_wrong = 0, 0
    for p, (i, h) in enumerate(pairs):
        j = int(h.get("replaces", 0))
        s_true = float(own[i, j])
        s_hard = float(hard_n[p] @ img_n[i])
        pair_wins.append(s_true > s_hard)
        diff_per_image[i] = diff_per_image.get(i, 0.0) + (s_true - s_hard)
        count_per_image[i] = count_per_image.get(i, 0) + 1

        rank_true = int((t2i[i, j] > t2i[i, j, i]).sum())          # 0 = top-1
        h_t2i = hard_n[p] @ img_n.T                                 # (N,)
        rank_hard = int((h_t2i > h_t2i[i]).sum())
        swap_wrong_to_right += rank_true > 0 and rank_hard == 0
        swap_right_to_wrong += rank_true == 0 and rank_hard > 0

    n = len(pairs)
    return {
        "n_pairs": n,
        "win_rate": (sum(pair_wins) / n) if n else float("nan"),
        "mean_score_diff": (sum(diff_per_image.values()) / n) if n else float("nan"),
        "text_swap_top1_gained": swap_wrong_to_right,
        "text_swap_top1_lost": swap_right_to_wrong,
        "score_diff_by_bin": _per_bin(
            entries, {i: diff_per_image[i] / count_per_image[i]
                      for i in diff_per_image}),
        # per-pair verdicts for the cross-model correction table; popped
        # before the JSON dump
        "_pair_wins": torch.tensor(pair_wins, dtype=torch.bool),
    }


# ---------------------------------------------------------------- model families
def clip_model_results(tag: str, ckpt: Path, subset: Dict, manifest: Path,
                       device: str, batch_size: int) -> Dict:
    model = CLIPFineTuneBaseline()
    model.load(str(ckpt))
    model = model.to(device).eval()
    img_raw, cap_raw = _clip_raw_features(model, _loader(manifest, batch_size), device)
    _assert_aligned(img_raw.shape[0], subset)
    entries = subset["entries"]

    pairs = _collect_pairs(entries)
    hard_raw = (_encode_texts_clip(model, [h["caption"] for _, h in pairs], device)
                if pairs else torch.zeros(0, cap_raw.shape[-1]))

    out = {
        "checkpoint": str(ckpt),
        # common protocol: norm(mean(raw f_k))
        "set_recall": set_recall_by_bin(img_raw, cap_raw.mean(dim=1), entries),
        "hard_negatives": hard_negative_checks_cosine(
            img_raw, cap_raw, hard_raw, entries) if pairs else {"n_pairs": 0},
        # secondary, legacy eval_clip_setlevel protocol: norm(mean(norm f_k))
        "set_recall_normed_mean": set_recall_by_bin(
            img_raw, F.normalize(cap_raw, dim=-1).mean(dim=1), entries),
        "_img_top1": set_top1_vectors(img_raw, cap_raw.mean(dim=1))[0],  # contrasts
    }
    return out


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

    true_rate = (d2_true <= m_pos).float().mean(dim=1)            # (N,)
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
    entries = subset["entries"]

    pairs = _collect_pairs(entries)
    hard_mu = (_encode_text_mus_mcdisp(model, [h["caption"] for _, h in pairs], device)
               if pairs else torch.zeros(0, feats["img_mu"].shape[-1]))

    return {
        "checkpoint": str(ckpt),
        # common protocol on the mu space: norm(mean(mu_k))
        "set_recall": set_recall_by_bin(feats["img_mu"],
                                        feats["text_mus"].mean(dim=1), entries),
        "hard_negatives": hard_negative_checks_cosine(
            feats["img_mu"], feats["text_mus"], hard_mu, entries) if pairs
        else {"n_pairs": 0},
        "mahalanobis": mahalanobis_checks_mcdisp(
            model, feats, entries, device, m_pos),
        "_img_top1": set_top1_vectors(feats["img_mu"],
                                      feats["text_mus"].mean(dim=1))[0],
    }


# ---------------------------------------------------------------- aggregation / CLI
def _model_contrasts(results: Dict, entries: List[Dict]) -> Dict[str, Dict]:
    """Set-I2T top-1 model-vs-model correction counts per bin (错->对/对->错).

    For an ordered pair "A>B": wrong_to_right = A wrong AND B right (B fixes
    A's errors); right_to_wrong = A right AND B wrong (B breaks A's hits).
    """
    vectors = {}
    for fam in ("clip", "mcdisp"):
        for tag, r in results.get(fam, {}).items():
            vectors[f"{fam}.{tag}"] = r["_img_top1"]
    out = {}
    names = sorted(vectors)
    for a in names:
        for b in names:
            if a >= b:
                continue
            va, vb = vectors[a], vectors[b]
            row = {}
            for binn in BINS + ("overall",):
                idx = ([i for i, e in enumerate(entries) if e["bin"] == binn]
                       if binn != "overall" else list(range(len(entries))))
                if not idx:
                    continue
                t = torch.tensor(idx)
                aw, bw = va[t], vb[t]
                row[binn] = {
                    "n": len(idx),
                    f"{a}_wrong_{b}_right": int((~aw & bw).sum()),
                    f"{a}_right_{b}_wrong": int((aw & ~bw).sum()),
                }
            out[f"{a}_vs_{b}"] = row
    return out


def _hardneg_model_contrasts(results: Dict, entries: List[Dict]) -> Dict[str, Dict]:
    """Per-PAIR model-vs-model verdicts on the minimal-fact controls.

    For an ordered pair "A>B": ``A_wrong_B_right`` counts the pairs where A
    scores the TRUE caption no better than its hard negative (判错) while B
    scores the true caption higher (判对) -- "CLIP 判错、Full 判对". Pairs are
    ordered by ``_collect_pairs``, identical for every model.
    """
    vectors = {}
    for fam in ("clip", "mcdisp"):
        for tag, r in results.get(fam, {}).items():
            v = r.get("hard_negatives", {}).get("_pair_wins")
            if v is not None and len(v) > 0:
                vectors[f"{fam}.{tag}"] = v.bool()
    if len(vectors) < 2:
        return {}
    pair_bins = [entries[i]["bin"] for i, _ in _collect_pairs(entries)]
    out = {}
    names = sorted(vectors)
    for a in names:
        for b in names:
            if a >= b:
                continue
            va, vb = vectors[a], vectors[b]
            row = {}
            for binn in BINS + ("overall",):
                idx = ([p for p, s in enumerate(pair_bins) if s == binn]
                       if binn != "overall" else list(range(len(pair_bins))))
                if not idx:
                    continue
                t = torch.tensor(idx)
                row[binn] = {
                    "n_pairs": len(idx),
                    f"{a}_wrong_{b}_right": int((~va[t] & vb[t]).sum()),
                    f"{a}_right_{b}_wrong": int((va[t] & ~vb[t]).sum()),
                }
            out[f"{a}_vs_{b}"] = row
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

    # set-I2T model-vs-model correction counts (checklist 错->对 / 对->错)
    results["model_contrasts"] = _model_contrasts(results, subset["entries"])
    # per-pair verdicts on the minimal-fact hard negatives (same 错->对 /
    # 对->错 question, restricted to the hard-negative pairs)
    results["hardneg_model_contrasts"] = _hardneg_model_contrasts(
        results, subset["entries"])
    # Full - Set-FT per-bin delta (checklist 结果表 column), common protocol
    if "full" in results["mcdisp"] and "setft" in results["clip"]:
        a = results["mcdisp"]["full"]["set_recall"]
        b = results["clip"]["setft"]["set_recall"]
        results["full_minus_setft_by_bin"] = {
            binn: {k: a[binn][k] - b[binn][k] for k in a[binn]
                   if isinstance(a[binn].get(k), float)}
            for binn in a if binn in b}

    # internal tensors are not JSON-serializable; drop before dumping
    for fam in ("clip", "mcdisp"):
        for r in results[fam].values():
            r.pop("_img_top1", None)
            r.get("hard_negatives", {}).pop("_pair_wins", None)
    out = Path(args.output) if args.output else (workdir / "exp4_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"exp4 results -> {out}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
