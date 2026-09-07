"""CLI driver for the MCDisp_Align ablation study (experiment plan §10).

Phases:
  audit         §10 phase 0 -- build image-exclusive manifests + audit report
  train         §10 phases 1–3 -- train one experiment x seed via the shared
                trainer (unified controls: same manifests/optimizer/budget,
                select_by dev mR, no early stop)
  eval          §10 phases 2–3 -- full test-manifest evaluation of one
                checkpoint: multi-caption retrieval (mc/cos/likelihood mR),
                H1/H2/H3 metric JSONs + per-query hit vectors for the
                paired bootstrap
  interventions §10 phase 4 -- scorer table + sigma/U interventions on one
                Full checkpoint
  report        §10 phase 5 -- aggregate every (experiment, seed) eval JSON:
                seed mean/std/CI, deltas vs Full, paired bootstrap + Holm for
                the three primary contrasts; writes CSV + Markdown tables

Examples::

  python scripts/ablation_study/run.py --phase audit
  python scripts/ablation_study/run.py --phase train --experiment full --seed 42
  python scripts/ablation_study/run.py --phase eval --experiment full --seed 42
  python scripts/ablation_study/run.py --phase interventions --experiment full --seed 42
  python scripts/ablation_study/run.py --phase report
  # end-to-end smoke (tiny manifests, 1 epoch):
  python scripts/ablation_study/run.py --phase audit --limit 200
  python scripts/ablation_study/run.py --phase train --experiment full --seed 42 --epochs 1
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config  # noqa: E402
from utils.logger import get_logger, setup_logger  # noqa: E402

from scripts.ablation_study import mc_ranking  # noqa: E402
from scripts.ablation_study.data_audit import build_manifests, verify_manifest  # noqa: E402
from scripts.ablation_study.experiments import (  # noqa: E402
    EXPERIMENTS, SEEDS, build_train_config,
)
from scripts.ablation_study.feature_extraction import extract_features  # noqa: E402
from scripts.ablation_study.gaussian_scorer import likelihood_sim_rows  # noqa: E402
from scripts.ablation_study.h1_semantic_range import h1_metrics  # noqa: E402
from scripts.ablation_study.h2_coverage import h2_metrics  # noqa: E402

logger = get_logger("ablation_run", config.OUTPUT_DIR / "ablation_study" / "run.log")

ROOT = config.OUTPUT_DIR / "ablation_study"
MANIFESTS = ROOT / "manifests"
TEST_MANIFEST = MANIFESTS / "manifest_coco_test.json"
K_VALUES = (1, 5, 10)


def _ckpt_path(experiment: str, seed: int) -> Path:
    cfg = build_train_config(experiment, seed)
    return cfg.best_path


def _device(args) -> str:
    return args.device or ("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- phases

def phase_audit(args) -> None:
    report = build_manifests(dataset="coco", out_dir=MANIFESTS, seed=config.SEED,
                             limit=args.limit)
    for split in ("train", "dev", "test"):
        p = MANIFESTS / f"manifest_coco_{split}.json"
        assert verify_manifest(p), f"manifest checksum mismatch: {p}"
    logger.info(f"Audit OK: {report['split_sizes']}")


def phase_train(args) -> None:
    from utils.mcdisp_align_trainer import run_mcdisp_align_training

    for seed in ([args.seed] if args.seed is not None else SEEDS):
        for split in ("train", "dev"):
            p = MANIFESTS / f"manifest_coco_{split}.json"
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found -- run `--phase audit` first")
            assert verify_manifest(p), f"manifest checksum mismatch for {split}"
        cfg = build_train_config(
            args.experiment, seed, manifests_dir=MANIFESTS,
            epochs=args.epochs, batch_size=args.batch_size, device=_device(args),
            loss=args.loss,
        )
        logger.info(f"=== train {args.experiment} seed={seed} ===")
        run_mcdisp_align_training(cfg, logger)


def phase_eval(args) -> None:
    device = _device(args)
    seed = args.seed if args.seed is not None else SEEDS[0]
    ckpt = Path(args.checkpoint) if args.checkpoint else _ckpt_path(args.experiment, seed)
    out_dir = ROOT / args.experiment / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = extract_features(
        ckpt, TEST_MANIFEST, num_captions=5,
        batch_size=args.batch_size or 32, device=device,
        max_images=args.num_samples,
    )
    img_mu, img_lv = feats["img_mu"], feats["img_logvar"]
    img_var = torch.exp(img_lv)
    t_mus, t_lvs = feats["text_mus"], feats["text_logvars"]
    N, K, D = t_mus.shape
    cap_mu = t_mus.reshape(N * K, D)
    cap_lv = t_lvs.reshape(N * K, D)

    # --- retrieval under the three scorers (plan §6.1, §6.4.5) ---
    r_mc = mc_ranking.full_ranking(img_mu, img_lv, t_mus, t_lvs, K_VALUES,
                                   tau=config.MCDISP_ALIGN_TAU)
    r_cos = mc_ranking.full_ranking(img_mu, img_lv, t_mus, t_lvs, K_VALUES,
                                    sim_builder=mc_ranking.cos_sim_rows)
    r_lik = mc_ranking.ranks_from_sim(
        likelihood_sim_rows(img_mu, img_lv, cap_mu, feats["img_U"]), K, K_VALUES)

    # --- hypothesis-specific metrics ---
    h1 = h1_metrics(img_var, t_mus, r_mc["i2t_hit"][1])
    h2 = h2_metrics(img_mu, img_var, feats["img_U"], t_mus, r_mc,
                    m_pos=config.MCDISP_ALIGN_M_POS)
    h3 = h3_metrics_safe(feats["img_U"], img_var, t_mus)

    def _retr(r):
        return {
            **{f"i2t_r@{k}": r["i2t_r"][k] for k in K_VALUES},
            **{f"t2i_r@{k}": r["t2i_r"][k] for k in K_VALUES},
            "mr": mc_ranking.mr_from(r, K_VALUES),
        }

    result = {
        "experiment": args.experiment, "seed": seed, "checkpoint": str(ckpt),
        "num_images": N, "K": K,
        "retrieval": {"mc": _retr(r_mc), "cos": _retr(r_cos), "likelihood": _retr(r_lik)},
        "h1_semantic_range": h1,
        "h2_coverage": h2,
        "h3_subspace": h3,
    }
    out_json = out_dir / "eval.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    # per-query R@1 hit vectors (paired bootstrap inputs, plan §9.3)
    torch.save({"i2t": r_mc["i2t_hit"][1], "t2i": r_mc["t2i_hit"][1]}, out_dir / "hits_r1.pt")
    logger.info(f"eval -> {out_json} (mR mc={result['retrieval']['mc']['mr']:.4f})")


def h3_metrics_safe(img_U, img_var, t_mus):
    from scripts.ablation_study.h3_subspace import h3_metrics
    try:
        return h3_metrics(img_U, img_var, t_mus)
    except ValueError as e:
        logger.warning(f"H3 metrics skipped: {e}")
        return {"s_sub": float("nan")}


def phase_interventions(args) -> None:
    from scripts.ablation_study.interventions import intervention_suite, query_side_invariance_check

    device = _device(args)
    seed = args.seed if args.seed is not None else SEEDS[0]
    ckpt = Path(args.checkpoint) if args.checkpoint else _ckpt_path(args.experiment, seed)
    feats = extract_features(ckpt, TEST_MANIFEST, num_captions=5,
                             batch_size=args.batch_size or 32, device=device,
                             max_images=args.num_samples)
    N, K, D = feats["text_mus"].shape
    cap_mu = feats["text_mus"].reshape(N * K, D)
    cap_lv = feats["text_logvars"].reshape(N * K, D)

    inv = intervention_suite(
        feats["img_mu"], feats["img_logvar"], feats["text_mus"], feats["text_logvars"],
        img_U=feats["img_U"], k_values=K_VALUES, tau=config.MCDISP_ALIGN_TAU)
    invar = query_side_invariance_check(
        feats["img_mu"], feats["img_logvar"], cap_mu, cap_lv, K,
        tau=config.MCDISP_ALIGN_TAU)

    out_dir = ROOT / args.experiment / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "interventions.json", "w", encoding="utf-8") as f:
        json.dump({"invariance_checks": invar, "suite": inv}, f, indent=2)
    logger.info(f"interventions -> {out_dir / 'interventions.json'} invariance={invar}")


def phase_report(args) -> None:
    from scripts.ablation_study.stats import aggregate_seeds, bootstrap_primary_contrasts

    rows = []
    for exp in EXPERIMENTS:
        per_seed, hit_files = [], []
        for seed in SEEDS:
            p = ROOT / exp / f"seed{seed}" / "eval.json"
            if p.exists():
                per_seed.append(json.loads(p.read_text()))
                hit_files.append(ROOT / exp / f"seed{seed}" / "hits_r1.pt")
        if not per_seed:
            continue
        flat = []
        for rec in per_seed:
            flat.append({
                **{f"mc_{k}": v for k, v in rec["retrieval"]["mc"].items()},
                **{f"cos_{k}": v for k, v in rec["retrieval"]["cos"].items()},
                **{f"lik_{k}": v for k, v in rec["retrieval"]["likelihood"].items()},
                **rec["h1_semantic_range"], **rec["h2_coverage"], **rec["h3_subspace"],
            })
        rows.append((exp, flat, per_seed, hit_files))

    full_mean = next((f[0].get("mc_mr") for e, f, *_ in rows if e == "full"), None)
    out_csv = ROOT / "report_summary.csv"
    metric_keys = sorted(set().union(*[set(f[0]) for _, f, *_ in rows]))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment"] + [f"{k}:{stat}" for k in metric_keys
                                     for stat in ("mean", "std", "ci95")])
        for exp, flat, _, _ in rows:
            agg = aggregate_seeds(flat, baseline=full_mean, keys=metric_keys)
            w.writerow([exp] + [f"{agg[k][s]:.6f}" if agg[k][s] == agg[k][s] else "nan"
                                for k in metric_keys for s in ("mean", "std", "ci95")])

    # primary contrasts (plan §9.3): F vs A1/A2/A3 on mR (mean over seeds of
    # per-seed paired bootstrap over queries)
    contrasts = {}
    by_name = {e: (f, recs, hits) for e, f, recs, hits in rows}
    if "full" in by_name:
        for other, key in (("no_var", "A1"), ("no_cover", "A2"), ("no_cov", "A3")):
            if other not in by_name or len(by_name[other][2]) != len(by_name["full"][2]):
                continue
            i2t_a = torch.cat([torch.load(p)["i2t"] for p in by_name["full"][2]])
            t2i_a = torch.cat([torch.load(p)["t2i"] for p in by_name["full"][2]])
            i2t_b = torch.cat([torch.load(p)["i2t"] for p in by_name[other][2]])
            t2i_b = torch.cat([torch.load(p)["t2i"] for p in by_name[other][2]])
            contrasts[key] = (i2t_a, i2t_b, t2i_a, t2i_b)
    raw, holm = (bootstrap_primary_contrasts(contrasts) if contrasts else ({}, {}))

    out_md = ROOT / "report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# MCDisp_Align 消融实验汇总（自动生成）\n\n")
        f.write(f"实验数：{len(rows)}；种子：{SEEDS}\n\n")
        f.write("## 主表（mc scorer mR + 创新点专属指标均值）\n\n")
        cols = ["mc_mr", "mc_i2t_r@1", "mc_t2i_r@1", "lik_mr", "e_var",
                "rho_sem_spearman", "coverage_rate", "pair_count@10",
                "worst_rank_median", "s_sub"]
        f.write("| experiment | " + " | ".join(cols) + " |\n")
        f.write("|---" * (len(cols) + 1) + "|\n")
        for exp, flat, _, _ in rows:
            agg = aggregate_seeds(flat, keys=metric_keys)
            cells = []
            for c in cols:
                v = agg.get(c, {}).get("mean", float("nan"))
                cells.append(f"{v:.4f}" if v == v else "—")
            f.write(f"| {exp} | " + " | ".join(cells) + " |\n")
        if raw:
            f.write("\n## 主对照 paired bootstrap（F vs A1/A2/A3，Holm 校正）\n\n")
            f.write("| contrast | direction | delta | 95% CI | CI excludes 0 | p_adj |\n|---|---|---|---|---|---|\n")
            for name, r in raw.items():
                base = name.split(":")[0]
                p = holm.get(base, float("nan"))
                f.write(f"| {name} | mr | {r['delta']:.4f} | "
                        f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] | "
                        f"{r['ci_excludes_zero']} | {p:.4f} |\n")
    logger.info(f"report -> {out_csv}, {out_md}")


# --------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="MCDisp_Align ablation study driver")
    ap.add_argument("--phase", required=True,
                    choices=["audit", "train", "eval", "interventions", "report"])
    ap.add_argument("--experiment", default="full", choices=list(EXPERIMENTS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-samples", type=int, default=None,
                    help="cap test images in eval/interventions (smoke/debug)")
    ap.add_argument("--checkpoint", default=None, help="explicit checkpoint for eval")
    ap.add_argument("--limit", type=int, default=None, help="audit: cap manifest entries")
    ap.add_argument("--loss", default=None, choices=["standard", "kl"],
                    help="Training objective override (default: the experiment's own "
                         "'loss' field). standard = L_mu/L_var separate terms; "
                         "kl = one KL(p_v||p_t) alignment term (lambda_kl).")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    setup_logger("ablation_run", ROOT / "run.log")

    {"audit": phase_audit, "train": phase_train, "eval": phase_eval,
     "interventions": phase_interventions, "report": phase_report}[args.phase](args)


if __name__ == "__main__":
    main()
