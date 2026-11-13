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

Usage (checkpoint paths are the trainer's REAL naming -- the ablation
framework writes e.g. outputs/ablation_study/full_kl/seed42/
abl_full_kl_kl_coco_best.pt, and the Set-FT run writes into its timestamped
log dir; always pass the actual file):

  python scripts/exp4_complementarity.py --make-template 90 --seed 42 \\
      --output subset_template.json          # step 1: sample (no overwrite)
  # step 2 (human): fill bin + hard_negatives
  python scripts/exp4_complementarity.py --subset subset_template.json \\
      --check-overlap                        # step 3: legacy-split overlap
  python scripts/exp4_complementarity.py --subset subset_template.json \\
      --clip clip=checkpoints/clip_baseline_coco_best.pt \\
      --mcdisp full=outputs/ablation_study/full_kl/seed42/abl_full_kl_kl_coco_best.pt
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
def _validate_subset(subset: Dict) -> List[str]:
    """Hard checks: bin in {low,mid,high}, EXACTLY 5 non-empty caption
    strings, and for every hard negative a non-empty 'caption' plus an
    integer 'replaces' in [0, 5) -- 1.7 is rejected, not silently truncated
    to 1. Returns the images without hard negatives (soft note -- they
    simply stay out of the hard-negative statistics)."""
    missing_neg = []
    for e in subset["entries"]:
        if e.get("bin") not in BINS:
            raise ValueError(f"entry {e['image']}: bin must be one of {BINS}")
        caps = e.get("captions")
        if not isinstance(caps, list) or len(caps) != 5:
            n = len(caps) if isinstance(caps, list) else type(caps).__name__
            raise ValueError(
                f"entry {e['image']}: needs EXACTLY 5 captions, got {n} "
                f"(the eval encodes five captions per image; re-sample the "
                f"template with >=5-valid entries)")
        if any(not isinstance(c, str) or not c.strip() for c in caps):
            raise ValueError(
                f"entry {e['image']}: every caption must be a non-empty string")
        for h in e.get("hard_negatives", []):
            cap = h.get("caption")
            if not isinstance(cap, str) or not cap.strip():
                raise ValueError(
                    f"entry {e['image']}: each hard negative needs a non-empty "
                    f"'caption' string")
            j = h.get("replaces")
            if isinstance(j, bool) or not isinstance(j, int):
                raise ValueError(
                    f"entry {e['image']}: hard negative 'replaces' must be an "
                    f"integer index in [0, 5), got {j!r}")
            if not 0 <= j < 5:
                raise ValueError(
                    f"entry {e['image']}: hard negative 'replaces'={j} "
                    f"outside caption range [0, 5)")
        if not e.get("hard_negatives"):
            missing_neg.append(e["image"])
    return missing_neg


def make_subset_template(manifest: Path, n: int, seed: int, out: Path,
                         force: bool = False) -> None:
    """Sample ``n`` TEST images BEFORE any model result is seen and emit the
    human-annotation template for the verified subset.

    Only entries with at least 5 valid captions are eligible (the eval loader
    drops shorter ones, which would break the alignment assert later), and an
    existing output file is never overwritten unless ``force`` -- regenerating
    would silently WIPE the human bin/hard-negative annotations.
    """
    if out.exists() and not force:
        raise FileExistsError(
            f"{out} already exists; re-generating would wipe the human "
            f"annotations. Pass --force only for a fresh, un-annotated template.")
    import random as _random
    blob = json.loads(manifest.read_text(encoding="utf-8"))
    entries = blob["entries"] if isinstance(blob, dict) else blob
    eligible = [e for e in entries
                if sum(1 for c in e["captions"]
                       if isinstance(c, str) and c.strip()) >= 5]
    if n > len(eligible):
        raise ValueError(
            f"requested {n} images but {manifest} only has {len(eligible)} "
            f"entries with >=5 valid captions ({len(entries) - len(eligible)} "
            f"ineligible entries were skipped)")
    rng = _random.Random(seed)
    picked = rng.sample(eligible, n)
    template = {
        "_说明": ("人工填写每项: bin = low/mid/high(互补性分档); hard_negatives 每项 "
                  "{'caption': 改一个事实的句子(非空), 'replaces': 被改句子的下标(必填整数 0-4), "
                  "'note': 改了什么}。先抽样后看模型结果,所有模型共用这一份。"),
        "seed": seed,
        "source_manifest": str(manifest),
        "entries": [
            {
                "image": e["image"],
                "bin": "",
                "captions": [c for c in e["captions"]
                             if isinstance(c, str) and c.strip()][:5],
                "hard_negatives": [],
            }
            for e in picked
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(template, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"subset template ({n} images, {len(eligible)} eligible in manifest) -> {out}\n"
          f"fill 'bin' and 'hard_negatives', then run exp4 with --subset {out}")


def legacy_split_overlap(image_names, captions_path=None, images_dir=None,
                         seed=None, val_split: float = 0.1) -> Dict[str, object]:
    """Intersection of ``image_names`` with the legacy 90/10 training split.

    The OLD runs (paper Full / CLIP-FT / Set-FT) trained on a random_split of
    the full caption pool (``torch.Generator().manual_seed(seed)``), which is
    exactly reproducible; the audit manifests are a DIFFERENT split of the
    same pool. Before claiming the exp-4 subset is unseen for those models,
    report this overlap: images in it were TRAINING images for the legacy
    checkpoints (rebuttal checklist: 旧权重需核对训练图像交集,不得直接宣称未见).
    """
    from torch.utils.data import random_split
    from utils.dataset_factory import build_train_dataset
    eff_seed = config.SEED if seed is None else seed
    ds = build_train_dataset(dataset="coco", captions_path=captions_path,
                             images_dir=images_dir)
    n_total = len(ds)
    n_val = int(n_total * val_split)
    train_sub, _ = random_split(
        ds, [n_total - n_val, n_val],
        generator=torch.Generator().manual_seed(eff_seed))
    legacy_train = set(
        ds.data.loc[list(train_sub.indices), "image_file_name"])
    overlap = sorted(set(image_names) & legacy_train)
    return {
        "legacy_seed": eff_seed, "legacy_val_split": val_split,
        "n_legacy_train": len(legacy_train), "n_query": len(set(image_names)),
        "n_overlap": len(overlap), "overlap_images": overlap,
        "note": "overlap images were TRAINING images for the legacy "
                "Full/CLIP/Set-FT checkpoints; exclude or report them",
    }


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
    ap.add_argument("--subset", default=None, help="human-verified subset JSON")
    ap.add_argument("--make-template", type=int, default=None, metavar="N",
                    help="sample N test images and emit the annotation "
                         "template instead of evaluating (step 1 of §4.1; "
                         "requires --manifest, honors --seed/--output)")
    ap.add_argument("--manifest", default=None,
                    help="audit test manifest for --make-template "
                         "(default: outputs/ablation_study/manifests/"
                         "manifest_coco_test.json)")
    ap.add_argument("--seed", type=int, default=config.SEED,
                    help="sampling seed for --make-template / legacy split seed "
                         "for --check-overlap")
    ap.add_argument("--force", action="store_true",
                    help="--make-template: overwrite an existing output file")
    ap.add_argument("--check-overlap", action="store_true",
                    help="report the overlap of --subset images with the LEGACY "
                         "90/10 training split (old Full/CLIP/Set-FT runs); "
                         "needs the caption parquet (server)")
    ap.add_argument("--clip", action="append", default=[], metavar="TAG=PATH",
                    help="CLIP-family checkpoint (clip_baseline / clip_setft)")
    ap.add_argument("--mcdisp", action="append", default=[], metavar="TAG=PATH",
                    help="MCDisp-family checkpoint (full / no_cover / diag ...)")
    ap.add_argument("--m-pos", type=float, default=config.MCDISP_ALIGN_M_POS)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.make_template is not None:
        manifest = Path(args.manifest or (
            config.OUTPUT_DIR / "ablation_study" / "manifests" / "manifest_coco_test.json"))
        out = Path(args.output) if args.output else Path("subset_template.json")
        make_subset_template(manifest, args.make_template, args.seed, out,
                             force=args.force)
        return

    if not args.subset:
        ap.error("--subset is required unless --make-template is given")

    with open(args.subset, "r", encoding="utf-8") as f:
        subset = json.load(f)

    # overlap check FIRST and with image IDs only: a freshly generated,
    # not-yet-annotated template must be able to query it before the human
    # annotation effort is spent on leaky images
    if args.check_overlap:
        for e in subset["entries"]:
            if not isinstance(e.get("image"), str) or not e["image"].strip():
                raise ValueError("every subset entry needs a non-empty 'image' field")
        report = legacy_split_overlap(
            [e["image"] for e in subset["entries"]], seed=args.seed)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    missing_neg = _validate_subset(subset)
    if missing_neg:
        print(f"NOTE: {len(missing_neg)}/{len(subset['entries'])} images have no "
              f"hard negative (fine for the bin retrieval tables; they just do "
              f"not enter the hard-negative statistics)")
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

    # legacy-split leakage: computed, recorded and labeled on EVERY eval run
    # (finding overlap is not enough -- with overlap > 0 the numbers are
    # seen-data diagnostics for the legacy checkpoints, never "unseen test")
    try:
        results["legacy_split_overlap"] = legacy_split_overlap(
            [e["image"] for e in subset["entries"]], seed=args.seed)
        n_ov = results["legacy_split_overlap"]["n_overlap"]
        results["subset_unseen_for_legacy_models"] = (n_ov == 0)
        if n_ov:
            print(f"WARNING: {n_ov} subset images were TRAINING images for the "
                  f"legacy Full/CLIP/Set-FT runs -- exp4 numbers on them are "
                  f"seen-data diagnostics, not unseen-test results")
    except Exception as e:  # noqa: BLE001 - data may be absent on this machine
        results["legacy_split_overlap"] = {"error": str(e)}
        results["subset_unseen_for_legacy_models"] = None
        print(f"NOTE: legacy-split overlap not computed ({e})")

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
