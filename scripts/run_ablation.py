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
