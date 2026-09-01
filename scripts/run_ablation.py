"""MCDisp_Align 消融实验 v2（四组目标版）。

设计文档: docs/superpowers/specs/2026-08-24-ablation-v2-design.md

判据有二：多描述检索协议（N 图像 vs N*5 描述，纯余弦 MCDisp_Align 打分器，
与主实验 evaluate_mcdisp_align 同口径）+ 对齐诊断（预期验证关系表）。

核心 5 变体（论文主消融表，单开关规则——每个变体只清零一个 λ，不动模型结构）：

    label            variant       开关                  目的
    Full             full          —                     完整模型
    w/o Coverage     no_cov        lambda_cov=0          验证逐 caption 覆盖约束（保留 Gaussian overlap matching）
    w/o Mean         no_mu         lambda_mu=0           验证共享中心显式对齐
    w/o Variance     no_var        lambda_var=0          验证完整边缘方差对齐（保留弱先验 R_prior 防方差数值失控）
    w/o Direction    no_dir_loss   lambda_dir=0          验证变化子空间对齐（保留低秩因子 U，不改结构）

预期验证关系（eval 时直接度量，λ=0 只清零权重/梯度，不清零诊断读数）：
    w/o Coverage  -> coverage@caption / coverage@set 下降
    w/o Mean      -> center MSE 上升
    w/o Variance  -> 完整方差 log-MSE 上升
    w/o Direction -> 子空间投影误差上升

附加 3 变体（单列）：diagonal_only（结构消融：cov_rank=0 + lambda_dir=0）、
no_reg（仅关弱先验 R_prior）、cosine_match（匹配分数退化纯余弦对照）。

用法:
    python scripts/run_ablation.py train --variant full
    python scripts/run_ablation.py eval  --variant no_var
    python scripts/run_ablation.py report
    python scripts/run_ablation.py all
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

SEED = config.SEED                    # 与主实验一致的划分种子（42）
K_VALUES = (1, 5, 10)
ABLATION_CKPT_DIR = config.CHECKPOINT_DIR / "ablation_v2"
ABLATION_OUT_DIR = config.OUTPUT_DIR / "ablation"

# 变体 -> 相对主配置的覆盖项。full 为空 dict = 全部取 config 默认
# （即主实验超参 match1.0/cov0.01/mu0.5/var1.0/reg0.01/dir0.5, r=4, gaussian）。
# 关闭一项损失就是权重清零，不做重新归一化。核心 5 变体严格单开关
# （只清零一个 λ，不动 cov_rank / match_score 等结构字段）。
VARIANTS = {
    "full": {
        "label": "Full",
        "desc": "完整方法（复刻主实验配置）",
        "overrides": {},
    },
    "no_cov": {
        "label": "w/o Coverage",
        "desc": "lambda_cov=0：仅关逐 caption 覆盖约束（保留 Gaussian overlap matching）",
        "overrides": {"lambda_cov": 0.0},
    },
    "no_mu": {
        "label": "w/o Mean",
        "desc": "lambda_mu=0：仅关共享中心显式对齐",
        "overrides": {"lambda_mu": 0.0},
    },
    "no_var": {
        "label": "w/o Variance",
        "desc": "lambda_var=0：仅关完整边缘方差对齐（保留弱先验 R_prior）",
        "overrides": {"lambda_var": 0.0},
    },
    "no_dir_loss": {
        "label": "w/o Direction",
        "desc": "lambda_dir=0：仅关变化子空间对齐（保留低秩因子 U）",
        "overrides": {"lambda_dir": 0.0},
    },
    "diagonal_only": {
        "label": "Diagonal-only",
        "desc": "结构消融：cov_rank=0 且关方向（移除低秩模块）",
        "overrides": {"cov_rank": 0, "lambda_dir": 0.0},
    },
    "no_reg": {
        "label": "w/o Prior",
        "desc": "仅关弱先验 R_prior",
        "overrides": {"lambda_reg": 0.0},
    },
    "cosine_match": {
        "label": "Cosine Match",
        "desc": "对照：匹配分数退化为纯余弦（无方差数据监督）",
        "overrides": {"match_score": "cosine"},
    },
}

# 论文主消融表的 5 个核心变体（严格单开关）；其余为附加对照。
CORE_VARIANTS = ("full", "no_cov", "no_mu", "no_var", "no_dir_loss")


def build_variant_config(variant, epochs=None, batch_size=None, device="cuda"):
    """变体名 -> MCDispAlignTrainConfig。

    未覆盖字段全部走 config 默认（= 主实验配置）；所有变体共用同一 seed，
    train/val random_split 划分完全一致。训练器 import 延迟到函数内，
    保证单元测试导入本模块轻量（不拉 torch/CLIP 之外的重组件）。
    """
    from utils.mcdisp_align_trainer import MCDispAlignTrainConfig

    if variant not in VARIANTS:
        raise KeyError(f"未知变体 {variant!r}；可选: {list(VARIANTS)}")
    kwargs = dict(VARIANTS[variant]["overrides"])
    if epochs is not None:
        kwargs["epochs"] = epochs
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    return MCDispAlignTrainConfig(
        dataset="coco",
        tag=f"abl2/{variant}",
        seed=SEED,
        select_by="recall",      # 与主实验同一 checkpoint 选择标准
        no_early_stop=True,      # 固定预算
        model_name=variant,      # -> {ABLATION_CKPT_DIR}/{variant}_coco_best.pt
        checkpoint_dir=ABLATION_CKPT_DIR,
        device=device,
        **kwargs,
    )


_REPORT_COLS = ["variant", "label", "desc", "mR", "delta_mR"] + [
    f"{d} R@{k}" for k in K_VALUES for d in ("i2t", "t2i")
]


def build_report_rows(results):
    """{variant: eval JSON} -> 检索报告行列表（按 VARIANTS 顺序，缺失变体跳过）。

    delta_mR = 该变体 mR - full mR（full 行为 None；full 缺失时全部为 None）。
    """
    full_mr = results.get("full", {}).get("mR")
    rows = []
    for name in VARIANTS:
        if name not in results:
            continue
        r = results[name]
        row = {
            "variant": name,
            "label": VARIANTS[name]["label"],
            "desc": VARIANTS[name]["desc"],
            "mR": r["mR"],
            "delta_mR": (r["mR"] - full_mr)
            if full_mr is not None and name != "full" else None,
        }
        for k in K_VALUES:
            row[f"i2t R@{k}"] = r["metrics"][f"mc_recall_i2t@{k}"]
            row[f"t2i R@{k}"] = r["metrics"][f"mc_recall_t2i@{k}"]
        rows.append(row)
    return rows


# 预期验证关系诊断表：核心 5 变体 × 5 诊断量（Δ = 变体 − full）。
# 每个单开关变体的"目标诊断"预期恶化：w/o Coverage -> Δcov 为负；
# w/o Mean -> Δmu_mse > 0；w/o Variance -> Δvar_err > 0；w/o Direction -> Δdir_err > 0。
_DIAG_FIELDS = (
    ("cov@cap", "coverage_caption"),   # 逐 caption 覆盖率（均值在 α 椭球内）
    ("cov@set", "coverage_set"),       # 图级覆盖率（全部 K 个 caption 都在椭球内）
    ("mu_mse", "center_mse"),          # 中心对齐误差（raw 坐标 MSE）
    ("var_err", "var_log_mse"),        # 完整边缘方差对齐误差（log 空间 MSE）
    ("dir_err", "dir_proj_err"),       # 子空间投影误差（2q − 2‖QvᵀQt‖²）
)
_DIAG_COLS = (["label", "switch"]
              + [c for c, _ in _DIAG_FIELDS]
              + [f"Δ{c}" for c, _ in _DIAG_FIELDS])


def build_diag_rows(results):
    """{variant: eval JSON} -> 预期验证关系诊断表行（仅 CORE_VARIANTS，
    缺 alignment 的变体跳过；Δ 列 = 该变体诊断 − full 诊断）。"""
    full_al = results.get("full", {}).get("alignment")
    rows = []
    for name in CORE_VARIANTS:
        r = results.get(name)
        if not r or "alignment" not in r:
            continue
        a = r["alignment"]
        row = {"label": VARIANTS[name]["label"],
               "switch": VARIANTS[name]["desc"]}
        for col, key in _DIAG_FIELDS:
            row[col] = a.get(key)
            row[f"Δ{col}"] = (a.get(key) - full_al.get(key)
                              if full_al is not None and name != "full"
                              and a.get(key) is not None
                              and full_al.get(key) is not None else None)
        rows.append(row)
    return rows


def format_markdown_table(rows, cols=_REPORT_COLS):
    if not rows:
        return "（暂无评测结果——先运行 eval 子命令）\n"
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for row in rows:
        cells = []
        for c in cols:
            v = row.get(c)
            cells.append("—" if v is None
                         else (f"{v:.4f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(results, out_dir=ABLATION_OUT_DIR):
    """写 report.md + report.csv + report_diag.csv，返回 report.md 路径。"""
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_report_rows(results)
    diag_rows = build_diag_rows(results)

    md = (
        "# MCDisp_Align 消融实验（多描述检索协议，N vs N×5）\n\n"
        "## 检索指标（主判据）\n\n"
        "主指标 mR = 6 个多描述 Recall（i2t 任一命中 + t2i，K=1/5/10）的均值；"
        "delta_mR = 该变体 − full（预期为负）。\n\n"
        + format_markdown_table(rows, _REPORT_COLS)
        + "\n## 对齐诊断（预期验证关系，核心 5 变体单开关）\n\n"
        "Δ = 该变体 − full。预期：w/o Coverage -> Δcov@cap/Δcov@set < 0；"
        "w/o Mean -> Δmu_mse > 0；w/o Variance -> Δvar_err > 0；"
        "w/o Direction -> Δdir_err > 0。cov@cap = 逐 caption 覆盖率，"
        "cov@set = 全部 K 个 caption 均在 α 置信椭球内的图像比例"
        f"（α={config.MCDISP_ALIGN_COV_ALPHA}）。\n\n"
        + format_markdown_table(diag_rows, _DIAG_COLS)
    )
    md_path = out_dir / "report.md"
    md_path.write_text(md, encoding="utf-8")

    csv_cols = [c for c in _REPORT_COLS if c != "desc"]
    with open(out_dir / "report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    with open(out_dir / "report_diag.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_DIAG_COLS, extrasaction="ignore")
        w.writeheader()
        for row in diag_rows:
            w.writerow(row)
    return md_path


def _get_logger():
    from utils.logger import get_logger
    return get_logger("run_ablation", config.LOG_DIR / "run_ablation_v2.log")


def cmd_train(args):
    from utils.cpu_affinity import apply_cpu_affinity
    from utils.mcdisp_align_trainer import run_mcdisp_align_training

    apply_cpu_affinity()
    logger = _get_logger()
    ABLATION_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_variant_config(
        args.variant, epochs=args.epochs, batch_size=args.batch_size,
        device=args.device or "cuda")
    logger.info(f"=== ablation_v2 train {args.variant} (seed={SEED}) ===")
    run_mcdisp_align_training(cfg, logger)


def cmd_eval(args):
    import torch

    from losses.mcdisp_align_losses import MCDispAlignLoss
    from models.mcdisp_align_model import MCDispAlignModel
    from utils.eval_common import build_eval_dataloader
    from utils.retrieval import compute_multicaption_recall

    logger = _get_logger()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = ABLATION_CKPT_DIR / f"{args.variant}_coco_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"{ckpt} 不存在——先运行: "
            f"python scripts/run_ablation.py train --variant {args.variant}")

    model = MCDispAlignModel()   # load() 按 checkpoint 的 cov_rank 自动重建协方差头
    model.load(str(ckpt))
    model = model.to(device).eval()

    loader, n_eval = build_eval_dataloader(
        "coco",
        batch_size=args.batch_size or config.MCDISP_ALIGN_BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        num_samples=args.num_samples,   # 默认 5000：与主实验同口径（底层数据池是
        # 全量 118k 图像，不传上限会得到与主表不可比的数字，见 spec §5）
    )

    # 对齐诊断 criterion：λ 权重与诊断读数无关（λ=0 只清零加权贡献/梯度，
    # 不清零 mu/var/dir/cov 值），match_score 取 cosine 纯粹是省掉全对
    # Gaussian overlap 计算——诊断量不依赖 match 分数。
    diag_crit = MCDispAlignLoss(
        match_score="cosine", cov_alpha=config.MCDISP_ALIGN_COV_ALPHA)
    diag_acc = {k: 0.0 for k in ("mu", "var", "cov_viol", "cov_viol_img")}
    dir_valid_sum = dir_total_sum = 0
    dir_weighted_sum = 0.0
    n_diag = 0

    feats = {k: [] for k in ("img_mu", "img_logvar", "text_mus", "text_logvars")}
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            pil_images, caption_lists = batch["image"], batch["captions"]
            B, K = len(pil_images), len(caption_lists[0])
            pixel_values = model.process_images(pil_images).to(device)
            flat = [c for cs in caption_lists for c in cs]
            ti = model.process_text(flat)
            outputs = model(
                pixel_values,
                ti["input_ids"].view(B, K, -1).to(device),
                ti["attention_mask"].view(B, K, -1).to(device),
            )
            feats["img_mu"].append(outputs["img_mu"].cpu())
            feats["img_logvar"].append(outputs["img_logvar"].cpu())
            feats["text_mus"].append(outputs["text_mus"].cpu())
            feats["text_logvars"].append(outputs["text_logvars"].cpu())

            _, d = diag_crit(
                outputs["img_mu"], outputs["img_logvar"], outputs["img_U"],
                outputs["text_mu"], outputs["text_logvar"],
                outputs["text_mus"], outputs["text_logvars"])
            for k in diag_acc:
                diag_acc[k] += d[k] * B
            dir_valid_sum += d["dir_valid"]
            dir_total_sum += d["dir_total"]
            dir_weighted_sum += d["dir"] * d["dir_valid"]
            n_diag += B

    # 打分在 GPU 上做（与主实验 eval 一致）；累积时用 CPU 控制
    # 显存，cat 完整表后一次性搬回 device（5000×25000 相似度 CPU 上会慢数分钟）。
    img_mu = torch.cat(feats["img_mu"]).to(device)
    img_lv = torch.cat(feats["img_logvar"]).to(device)
    t_mus = torch.cat(feats["text_mus"]).to(device)
    t_lvs = torch.cat(feats["text_logvars"]).to(device)

    metrics = compute_multicaption_recall(
        img_mu, img_lv, t_mus, t_lvs, list(K_VALUES), tau=config.MCDISP_ALIGN_TAU)
    mr = sum(metrics[f"mc_recall@{k}"] for k in K_VALUES) / len(K_VALUES)

    # 预期验证关系诊断（按图像数加权平均；dir 按通过秩守卫的样本数加权）
    alignment = {
        "coverage_caption": 1.0 - diag_acc["cov_viol"] / max(n_diag, 1),
        "coverage_set": 1.0 - diag_acc["cov_viol_img"] / max(n_diag, 1),
        "center_mse": diag_acc["mu"] / max(n_diag, 1),
        "var_log_mse": diag_acc["var"] / max(n_diag, 1),
        "dir_proj_err": dir_weighted_sum / max(dir_valid_sum, 1),
        "dir_valid_frac": dir_valid_sum / max(dir_total_sum, 1),
    }

    result = {
        "variant": args.variant,
        "label": VARIANTS[args.variant]["label"],
        "desc": VARIANTS[args.variant]["desc"],
        "checkpoint": str(ckpt),
        "num_samples": n_eval,
        "tau": config.MCDISP_ALIGN_TAU,
        "seed": SEED,
        "mR": mr,
        "metrics": metrics,
        "alignment": alignment,
    }
    ABLATION_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = ABLATION_OUT_DIR / f"{args.variant}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        f"{args.variant}: mR={mr:.4f} (N={n_eval}) | "
        f"cov@cap={alignment['coverage_caption']:.3f} "
        f"cov@set={alignment['coverage_set']:.3f} "
        f"mu={alignment['center_mse']:.4f} var={alignment['var_log_mse']:.4f} "
        f"dir={alignment['dir_proj_err']:.3f} -> {out}")


def cmd_report(args):
    logger = _get_logger()
    results = {}
    for name in VARIANTS:
        p = ABLATION_OUT_DIR / f"{name}.json"
        if p.exists():
            results[name] = json.loads(p.read_text(encoding="utf-8"))
        else:
            logger.warning(f"缺少 {p}，报告中跳过 {name}")
    md_path = write_report(results)
    logger.info(f"report -> {md_path}")


def cmd_all(args):
    for name in VARIANTS:
        args.variant = name
        cmd_train(args)
        cmd_eval(args)
    cmd_report(args)


def main():
    ap = argparse.ArgumentParser(description="MCDisp_Align 消融实验 v2（检索判据）")
    ap.add_argument("command", choices=["train", "eval", "report", "all"])
    ap.add_argument(
        "--variant", "--config", dest="variant", default="full",
        choices=list(VARIANTS),
        help="消融变体（--config 为 main.py 兼容别名）")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument(
        "--num-samples", type=int, default=5000,
        help="eval 评测图像数（默认 5000，与主实验同口径；底层数据池是全量 "
             "118k 图像，不设上限的数字与主表不可比）")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval,
     "report": cmd_report, "all": cmd_all}[args.command](args)


if __name__ == "__main__":
    main()
