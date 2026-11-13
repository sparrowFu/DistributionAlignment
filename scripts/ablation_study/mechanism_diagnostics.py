"""Rebuttal exp 3: mechanism diagnostics on existing checkpoints.

Four readouts, all from ONE checkpoint + one fixed manifest split, no
parameter updates (plan: 固定少量真实 batch,用 torch.autograd.grad 单独计算各项梯度):

  1. coverage_activation   per-caption d^2/D under the TRAINING metric
                           (Sigma_v = diag(sigma^2) + U U^T, per-D normalized,
                           same Woodbury code as the loss): median / q90 / q99,
                           fraction above m_pos, the weighted hinge loss value.
  2. marginal_variance     diag(Sigma^v) vs diag(Sigma^C): quantiles of the
                           difference (image base variance + low-rank marginal
                           vs caption spread + mean caption variance) and the
                           low-rank variance share.
  3. text_variance         per-caption variance quantiles + per-dimension
                           cross-caption variation ("文本方差是否接近常数").
  4. constant_var_rerank   replace EVERY caption variance by target_var (0.04),
                           keep means and between-caption dispersion, re-rank
                           with the mc scorer: R@1 deltas + top-1 flip rates.
                           T2I must be flip-free (the constant becomes a row
                           constant for each caption query) -- a non-zero T2I
                           flip rate flags a scorer/rounding bug.
  5. term_gradients        per-loss-term torch.autograd.grad norms into the
                           low-rank factor U, the img_cov_head (低秩预测头) and
                           the variance heads, each classified as
                           disabled_lambda / stage_zero / no_path / active /
                           active_zero_grad given the staged schedule epoch.

Usage:
  python scripts/ablation_study/mechanism_diagnostics.py \
      --checkpoint outputs/ablation_study/full_kl/seed42/best.pt \
      --epoch 7 --total-epochs 10
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config  # noqa: E402
from utils.logger import get_logger  # noqa: E402

from losses.mcdisp_align_losses_kl import MCDispAlignKLLoss  # noqa: E402
from utils.mcdisp_align_trainer import (  # noqa: E402
    alpha_schedule, stage_multipliers, var_ramp,
)
from scripts.ablation_study.feature_extraction import extract_features  # noqa: E402
from scripts.ablation_study.h2_coverage import positive_mahalanobis  # noqa: E402
from scripts.ablation_study import mc_ranking  # noqa: E402

logger = get_logger("mechanism_diagnostics",
                    config.OUTPUT_DIR / "ablation_study" / "mechanism_diagnostics.log")

ROOT = config.OUTPUT_DIR / "ablation_study"
K_VALUES = (1, 5, 10)


# ---------------------------------------------------------------- offline stats
def coverage_activation_stats(feats: Dict[str, torch.Tensor],
                              m_pos: float, lambda_cover_pos: float) -> Dict[str, float]:
    """Checklist 3 row 1: is the coverage term actually active?"""
    img_var = torch.exp(feats["img_logvar"])
    d2 = positive_mahalanobis(feats["img_mu"], img_var, feats["img_U"],
                              feats["text_mus"])            # (N, K) = d^2/D
    flat = d2.flatten()

    def q(p):
        return float(torch.quantile(flat.float(), p))

    hinge = torch.relu(d2 - m_pos)
    return {
        "d2_over_D_median": q(0.5), "d2_over_D_q90": q(0.9), "d2_over_D_q99": q(0.99),
        "d2_over_D_max": float(flat.max()),
        "frac_above_m_pos": float((flat > m_pos).float().mean()),
        "m_pos": m_pos,
        "weighted_cover_pos": lambda_cover_pos * float(hinge.mean()),
    }


def marginal_variance_stats(feats: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Checklist 3 row 3: is the image TOTAL marginal variance too large?

    diag(Sigma^v) = sigma_v^2 + ||U_d||^2 (per-dim low-rank marginal),
    diag(Sigma^C) = caption spread s^2 + (1/K) sum_k sigma_k^2.
    """
    sigma_v = torch.exp(feats["img_logvar"])                       # (N, D)
    u_var = torch.zeros_like(sigma_v)
    if feats["img_U"] is not None:
        u_var = (feats["img_U"] ** 2).sum(dim=-1)                  # (N, D)
    img_marginal = sigma_v + u_var

    spread2 = ((feats["text_mus"] - feats["text_mus"].mean(dim=1, keepdim=True))
               ** 2).mean(dim=1)                                   # (N, D) = s^2
    mean_cap_var = torch.exp(feats["text_logvars"]).mean(dim=1)    # (N, D)
    text_marginal = spread2 + mean_cap_var

    diff = (img_marginal - text_marginal).flatten()

    def q(p):
        return float(torch.quantile(diff.float(), p))

    total = float(img_marginal.sum())
    return {
        "img_minus_text_marginal_median": q(0.5),
        "img_minus_text_marginal_q10": q(0.10), "img_minus_text_marginal_q90": q(0.90),
        "frac_img_larger": float((diff > 0).float().mean()),
        "lowrank_variance_share": float(u_var.sum()) / total if total > 0 else 0.0,
        "spread_over_capvar": float(spread2.sum() / mean_cap_var.sum()),
    }


def text_variance_stats(feats: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Checklist 3 row 4a: are caption variances close to a constant?"""
    cap_var = torch.exp(feats["text_logvars"])                     # (N, K, D)
    flat = cap_var.flatten()
    cross = cap_var.std(dim=1)                                     # (N, D) per-dim across K
    return {
        "cap_var_median": float(torch.quantile(flat.float(), 0.5)),
        "cap_var_q10": float(torch.quantile(flat.float(), 0.10)),
        "cap_var_q90": float(torch.quantile(flat.float(), 0.90)),
        "cap_var_mean": float(flat.mean()),
        "cross_caption_std_mean": float(cross.mean()),
        "cross_caption_std_median": float(torch.quantile(cross.flatten().float(), 0.5)),
        "cross_caption_cv": float(cross.mean() / cap_var.mean()),  # relative dispersion
    }


def constant_variance_intervention(feats: Dict[str, torch.Tensor],
                                   target_var: float = 0.04,
                                   tau: float = 0.07) -> Dict[str, float]:
    """Checklist 3 row 4b: variance := 0.04 on every caption, means AND the
    between-caption dispersion untouched, then re-score 集合检索 with the
    moment-matched set variance (Eq. 4):

        v_C = Var_k(mu_k) + (1/K) sum_k v_k   ->   v_C' = Var_k(mu_k) + 0.04

    PRIMARY output is the SET-level N x N retrieval (each image vs each
    caption set; positive = diagonal). The per-caption N x (N*K) variant is
    reported separately ("per_caption_*"): it answers a different question
    (independent-caption retrieval) and is kept only as a secondary readout.

    float32 scoring on purpose: the T2I invariance must survive rounding only
    if it is genuinely there (float16 near-ties would fake a flip).
    """
    N, K, D = feats["text_mus"].shape
    eps = config.MCDISP_ALIGN_COV_EPS
    img_mu, img_lv = feats["img_mu"], feats["img_logvar"]
    t_mus, t_lvs = feats["text_mus"], feats["text_logvars"]

    # ---- set level (primary): merged set variance under Eq. 4 ----
    set_mu = t_mus.mean(dim=1)                                     # (N, D)
    spread2 = ((t_mus - set_mu.unsqueeze(1)) ** 2).mean(dim=1)     # Var_k(mu_k)
    base_lv = torch.log(spread2 + torch.exp(t_lvs).mean(dim=1) + eps)
    interv_lv = torch.log(spread2 + target_var + eps)              # dispersion kept

    def _set_stats(sim):
        r = mc_ranking.ranks_from_sim(sim, 1, K_VALUES)            # K=1: own set = diagonal
        return r

    sim0 = mc_ranking.mc_sim_rows(img_mu, img_lv, set_mu, base_lv,
                                  tau=tau, dtype=torch.float32)
    sim1 = mc_ranking.mc_sim_rows(img_mu, img_lv, set_mu, interv_lv,
                                  tau=tau, dtype=torch.float32)
    r0, r1 = _set_stats(sim0), _set_stats(sim1)
    i2t_flip = float((sim0.argmax(1) != sim1.argmax(1)).float().mean())
    t2i_flip = float((sim0.argmax(0) != sim1.argmax(0)).float().mean())

    out = {
        "target_var": target_var,
        "set_i2t_r1_base": r0["i2t_r"][1], "set_i2t_r1_const_var": r1["i2t_r"][1],
        "set_i2t_r1_delta": r1["i2t_r"][1] - r0["i2t_r"][1],
        "set_t2i_r1_base": r0["t2i_r"][1], "set_t2i_r1_const_var": r1["t2i_r"][1],
        "set_t2i_r1_delta": r1["t2i_r"][1] - r0["t2i_r"][1],
        "set_i2t_top1_flip_rate": i2t_flip,
        "set_t2i_top1_flip_rate": t2i_flip,  # expected 0: the constant set
        # variance is a row constant for each set query (checklist's check)
    }

    # ---- per-caption level (secondary, independent-caption retrieval) ----
    cap_mu = t_mus.reshape(N * K, D)
    cap_lv = t_lvs.reshape(N * K, D)
    p0 = mc_ranking.mc_sim_rows(img_mu, img_lv, cap_mu, cap_lv,
                                tau=tau, dtype=torch.float32)
    p1 = mc_ranking.mc_sim_rows(img_mu, img_lv, cap_mu,
                                torch.full_like(cap_lv, math.log(target_var)),
                                tau=tau, dtype=torch.float32)
    q0 = mc_ranking.ranks_from_sim(p0, K, K_VALUES)
    q1 = mc_ranking.ranks_from_sim(p1, K, K_VALUES)
    out.update({
        "per_caption_i2t_r1_base": q0["i2t_r"][1],
        "per_caption_i2t_r1_const_var": q1["i2t_r"][1],
        "per_caption_i2t_r1_delta": q1["i2t_r"][1] - q0["i2t_r"][1],
        "per_caption_t2i_r1_base": q0["t2i_r"][1],
        "per_caption_t2i_r1_const_var": q1["t2i_r"][1],
        "per_caption_t2i_r1_delta": q1["t2i_r"][1] - q0["t2i_r"][1],
        "per_caption_t2i_top1_flip_rate": float(
            (p0.argmax(1) != p1.argmax(1)).float().mean()),
    })
    return out


# ---------------------------------------------------------------- gradient probes
def _batch_tensors(model, loader, device, max_batches: int):
    """Yield forward-output dicts on the first ``max_batches`` real batches."""
    got = 0
    for batch in loader:
        if batch is None:
            continue
        B = len(batch["captions"])
        K = len(batch["captions"][0])
        caps = [c for cl in batch["captions"] for c in cl]     # image-major (B*K,)
        pixel_values = model.process_images(batch["image"]).to(device)
        ti = model.process_text(caps)
        # forward expects (B, K, L); process_text returns flat (B*K, L)
        input_ids = ti["input_ids"].to(device).view(B, K, -1)
        attention_mask = ti["attention_mask"].to(device).view(B, K, -1)
        out = model(pixel_values, input_ids, attention_mask)
        yield B, out
        got += 1
        if got >= max_batches:
            break


def _grad_norm(term, inputs):
    """Total grad norm over ``inputs``.

    Returns None when EVERY input is unused by the graph (structurally no
    gradient path, e.g. a detached target) -- distinct from 0.0, which means
    the path exists but this batch contributed no gradient.
    """
    if not term.requires_grad:
        return None          # constant term (e.g. lambda_cov=0 -> new_zeros)
    inputs = [t for t in inputs if t.requires_grad]
    if not inputs:
        return None
    grads = torch.autograd.grad(term, inputs, retain_graph=True,
                                allow_unused=True)
    if all(g is None for g in grads):
        return None
    sq = None
    for g in grads:
        if g is None:
            continue
        s = g.detach().pow(2).sum()
        sq = s if sq is None else sq + s
    return float(torch.sqrt(sq)) if sq is not None else None


def term_gradient_probe(checkpoint: Path, manifest: Path, device: str,
                        batch_size: int, num_batches: int,
                        epoch: int, step: int, total_epochs: int,
                        no_staged: bool,
                        crit: MCDispAlignKLLoss) -> Dict[str, Dict[str, float]]:
    """Checklist 3 rows 1-2 + 4a: which gradients actually reach U / the
    low-rank head / the variance heads, per loss term, on real batches.

    ``epoch`` is the TRAINER's 0-based epoch and ``step`` the batch index
    within it, so the per-step variance schedules the trainer applies
    (``var_ramp`` for the KL term, ``alpha_schedule`` for the L_set ->
    variance straight-through) are reproduced here: the criterion's
    ``uncertainty_grad_alpha`` is set to the scheduled value before probing,
    and the KL classification gate is the product of the per-epoch stage
    gate and the per-step ramp. Reported ``mean_grad_norm`` values are the
    RAW term gradients; ``effective_grad_norm`` scales them by
    lambda * stage gate (* ramp for KL) -- what the optimizer would see.
    """
    from torch.utils.data import DataLoader
    from data.caption_dataset import filter_none_collate
    from data.manifest_caption_dataset import ManifestCaptionDataset
    from models.mcdisp_align_model import MCDispAlignModel

    steps_per_epoch = max(1, len(ManifestCaptionDataset(
        manifest, config.IMAGES_DIR, num_captions=5, sample_mode="first")) // batch_size)
    ramp = var_ramp(epoch, step, steps_per_epoch, total_epochs)
    alpha = alpha_schedule(epoch, step, steps_per_epoch, total_epochs)
    crit.uncertainty_grad_alpha = alpha   # trainer sets this per step

    model = MCDispAlignModel()
    model.load(str(checkpoint))
    model = model.to(device).eval()

    ds = ManifestCaptionDataset(manifest, config.IMAGES_DIR,
                                num_captions=5, sample_mode="first")
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=2,
        collate_fn=filter_none_collate)

    probes = {
        "cover_pos": {"inputs": ["img_U", "img_cov_head", "img_logvar_head"],
                      "fn": lambda o: crit._cover_loss(
                          o["img_mu"], torch.exp(o["img_logvar"]), o["img_U"],
                          o["text_mus"], o["text_mu"])[0]},
        "cov": {"inputs": ["img_U", "img_cov_head"],
                "fn": lambda o: crit._cov_loss(o["img_mu"], o["img_U"], o["text_mus"])},
        "kl": {"inputs": ["img_mu_head", "img_logvar_head",
                          "text_mu_head", "text_logvar_head"],  # text: detached target -> expect no_path
               "fn": lambda o: crit._kl_loss(o["img_mu"], o["img_logvar"],
                                             o["text_mu"], o["text_logvar"])[0]},
        "set_nce": {"inputs": ["text_mu_head", "text_logvar_head"],
                    "fn": lambda o: crit._set_nce(o["img_mu"], o["img_logvar"],
                                                  o["text_mu"], o["text_logvar"])},
        "reg": {"inputs": ["text_logvar_head", "img_logvar_head"],
                "fn": lambda o: crit._reg_loss(o["img_logvar"], o["text_logvars"])},
    }

    def _param_inputs(names):
        out = []
        for n in names:
            head = getattr(model, n, None)
            if head is not None:
                out += [p for p in head.parameters() if p.requires_grad]
        return out

    accum = {t: {} for t in probes}
    stage = stage_multipliers(epoch, total_epochs, no_staged)
    lambdas = {"cover_pos": crit.lambda_cover_pos, "cov": crit.lambda_cov,
               "kl": crit.lambda_kl, "set_nce": crit.lambda_ctr, "reg": crit.lambda_reg}

    with torch.enable_grad():
        for b, out in _batch_tensors(model, loader, device, num_batches):
            tensors = {
                "img_U": [out["img_U"]] if out["img_U"] is not None else [],
                "img_cov_head": _param_inputs(["img_cov_head"]),
                "img_mu_head": _param_inputs(["img_mu_head"]),
                "img_logvar_head": _param_inputs(["img_logvar_head"]),
                "text_mu_head": _param_inputs(["text_mu_head"]),
                "text_logvar_head": _param_inputs(["text_logvar_head"]),
            }
            for name, spec in probes.items():
                term = spec["fn"](out)
                for key in spec["inputs"]:
                    inputs = tensors[key]
                    if not inputs:
                        accum[name].setdefault(key, []).append(None)
                        continue
                    accum[name].setdefault(key, []).append(_grad_norm(term, inputs))

    def _classify(term: str, norms) -> Dict[str, float]:
        vals = [n for n in norms if n is not None]      # batches with a path
        lam = lambdas[term]
        gate = {"cover_pos": stage["cover_pos"], "cov": stage["cov"],
                "kl": stage["var"] * ramp,               # KL rides var_ramp per step
                "set_nce": stage["ctr"], "reg": stage["reg"]}[term]
        if lam <= 0:
            status = "disabled_lambda"
        elif gate <= 0:
            status = "stage_zero"
        elif not vals:
            status = "no_path"                          # unused by the graph on every batch
        else:
            g = sum(vals) / len(vals)
            status = "active" if g > 0 else "active_zero_grad"
        return {"mean_grad_norm": (sum(vals) / len(vals)) if vals else 0.0,
                "effective_grad_norm": (sum(vals) / len(vals)) * lam * gate
                if vals else 0.0,
                "lambda": lam, "stage_gate": gate, "status": status,
                "batches_with_path": len(vals)}

    result = {}
    for term, per_input in accum.items():
        result[term] = {k: _classify(term, v) for k, v in per_input.items()}
    result["_schedule"] = {"epoch": epoch, "step": step,
                           "total_epochs": total_epochs,
                           "steps_per_epoch": steps_per_epoch,
                           "stage": stage["stage"],
                           "var_ramp": ramp, "uncertainty_grad_alpha": alpha,
                           "gates": {k: stage[k] for k in
                                     ("var", "cover_pos", "cover_neg", "cov")}}
    return result


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Rebuttal exp 3: mechanism diagnostics")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "manifests" / "manifest_coco_test.json"))
    ap.add_argument("--num-images", type=int, default=1000,
                    help="images for the offline feature stats / rerank checks")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-batches", type=int, default=4,
                    help="real batches for the autograd probes")
    ap.add_argument("--epoch", type=int, default=None,
                    help="TRAINER's 0-based epoch (same numbering as "
                         "stage_multipliers) used to classify stage gates "
                         "(default: total_epochs, i.e. the Full stage)")
    ap.add_argument("--step", type=int, default=0,
                    help="batch index within the epoch, for the per-step "
                         "var_ramp / uncertainty_grad_alpha schedules")
    ap.add_argument("--total-epochs", type=int, default=config.MCDISP_ALIGN_EPOCHS)
    ap.add_argument("--no-staged", action="store_true",
                    help="the diagnosed run used a flat schedule (no stages)")
    # loss weights must MATCH the diagnosed run; defaults = full_kl recipe
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--lambda-cover-pos", type=float, default=0.5)
    ap.add_argument("--lambda-cov", type=float, default=config.MCDISP_ALIGN_LAMBDA_COV)
    ap.add_argument("--lambda-reg", type=float, default=0.01)
    ap.add_argument("--m-pos", type=float, default=config.MCDISP_ALIGN_M_POS,
                    help="coverage margin; used by BOTH the coverage stats and "
                         "the criterion's cover hinge (must match the run)")
    ap.add_argument("--target-var", type=float, default=0.04,
                    help="ANALYSIS-ONLY variance the captions are replaced with "
                         "in the constant-variance intervention")
    ap.add_argument("--reg-target-var", type=float, default=0.04,
                    help="TRAINING sigma_0^2 prior of L_reg (the diagnosed "
                         "run's value; drives the reg-term gradient probe)")
    ap.add_argument("--no-sg", action="store_true",
                    help="the diagnosed run is the A7 'no sg' ablation: turn "
                         "off the three target detaches (kl/cover/cov)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    manifest = Path(args.manifest)
    epoch = args.total_epochs if args.epoch is None else args.epoch

    feats = extract_features(ckpt, manifest, num_captions=5,
                             batch_size=args.batch_size, device=args.device,
                             max_images=args.num_images)

    crit = MCDispAlignKLLoss(lambda_kl=args.lambda_kl,
                             lambda_cover_pos=args.lambda_cover_pos,
                             lambda_cov=args.lambda_cov,
                             lambda_reg=args.lambda_reg,
                             tau=config.MCDISP_ALIGN_TAU,
                             m_pos=args.m_pos,
                             target_var=args.reg_target_var,
                             kl_detach_target=not args.no_sg,
                             cover_detach_target=not args.no_sg,
                             cov_detach_target=not args.no_sg)

    result = {
        "checkpoint": str(ckpt), "manifest": str(manifest),
        "num_images": int(feats["img_mu"].shape[0]),
        "coverage_activation": coverage_activation_stats(
            feats, args.m_pos, args.lambda_cover_pos),
        "marginal_variance": marginal_variance_stats(feats),
        "text_variance": text_variance_stats(feats),
        "constant_variance_intervention": constant_variance_intervention(
            feats, target_var=args.target_var, tau=config.MCDISP_ALIGN_TAU),
        "term_gradients": term_gradient_probe(
            ckpt, manifest, args.device, args.batch_size, args.grad_batches,
            epoch, args.step, args.total_epochs, args.no_staged, crit),
    }

    out = Path(args.output) if args.output else (
        ROOT / "mechanism_diagnostics" / f"{ckpt.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"diagnostics -> {out}")
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("coverage_activation", "marginal_variance",
                               "text_variance", "constant_variance_intervention")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
