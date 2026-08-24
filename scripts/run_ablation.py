"""MCDisp_Align 消融实验 v2。

设计文档: docs/superpowers/specs/2026-08-24-ablation-v2-design.md

4 个变体，每个 = 完整方法拿掉一个创新模块，以多描述检索协议
（N 图像 vs N*5 描述，MCDisp_Align 打分器，与主实验 evaluate_mcdisp_align
同口径）为唯一判据：

    full      完整方法（复刻主实验配置，λ_cov=0.01, r=4）
    no_var    lambda_var=0             方差<-多描述离散度监督（核心创新）
    no_dir    cov_rank=0, lambda_cov=0 低秩协方差<-主要变化方向（连头移除）
    no_cover  lambda_cover_pos=0       逐描述覆盖

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
# （即主实验实测超参 ctr1.0/mu0.5/var1.0/cover_pos0.5/cov0.01/reg0.01, r=4）。
# 关闭一项损失就是权重清零，不做重新归一化。
VARIANTS = {
    "full": {
        "desc": "完整方法（= 主实验配置）",
        "overrides": {},
    },
    "no_var": {
        "desc": "去掉 L_var：方差←多描述离散度监督（核心创新）",
        "overrides": {"lambda_var": 0.0},
    },
    "no_dir": {
        "desc": "去掉低秩模块：cov_rank=0 且 λ_cov=0（协方差与覆盖距离均退化为对角）",
        "overrides": {"cov_rank": 0, "lambda_cov": 0.0},
    },
    "no_cover": {
        "desc": "去掉 L_cover：逐描述覆盖",
        "overrides": {"lambda_cover_pos": 0.0},
    },
}


def build_variant_config(variant, epochs=None, batch_size=None, device="cuda"):
    """变体名 -> MCDispAlignTrainConfig。

    未覆盖字段全部走 config 默认（= 主实验配置）；4 个变体共用同一 seed，
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


_REPORT_COLS = ["variant", "desc", "mR", "delta_mR", "cos_mR"] + [
    f"{d} R@{k}" for k in K_VALUES for d in ("i2t", "t2i")
]


def build_report_rows(results):
    """{variant: eval JSON} -> 报告行列表（按 VARIANTS 顺序，缺失变体跳过）。

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
            "desc": VARIANTS[name]["desc"],
            "mR": r["mR"],
            "cos_mR": r.get("cos_mR"),
            "delta_mR": (r["mR"] - full_mr)
            if full_mr is not None and name != "full" else None,
        }
        for k in K_VALUES:
            row[f"i2t R@{k}"] = r["metrics"][f"mc_recall_i2t@{k}"]
            row[f"t2i R@{k}"] = r["metrics"][f"mc_recall_t2i@{k}"]
        rows.append(row)
    return rows


def format_markdown_table(rows):
    if not rows:
        return "（暂无评测结果——先运行 eval 子命令）\n"
    lines = [
        "| " + " | ".join(_REPORT_COLS) + " |",
        "|" + "---|" * len(_REPORT_COLS),
    ]
    for row in rows:
        cells = []
        for c in _REPORT_COLS:
            v = row.get(c)
            cells.append("—" if v is None
                         else (f"{v:.4f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(results, out_dir=ABLATION_OUT_DIR):
    """写 report.md + report.csv，返回 report.md 路径。"""
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_report_rows(results)

    md = (
        "# MCDisp_Align 消融实验（多描述检索协议，N vs N×5）\n\n"
        "主指标 mR = 6 个多描述 Recall（i2t 任一命中 + t2i，K=1/5/10）的均值；"
        "delta_mR = 该变体 − full（预期为负）。cos_mR 为余弦打分器参考列。\n\n"
        + format_markdown_table(rows)
    )
    md_path = out_dir / "report.md"
    md_path.write_text(md, encoding="utf-8")

    csv_cols = [c for c in _REPORT_COLS if c != "desc"]
    with open(out_dir / "report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return md_path
