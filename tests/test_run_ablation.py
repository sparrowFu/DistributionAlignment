"""scripts/run_ablation.py（消融 v2）单元测试：变体→配置映射、报告拼装。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

from scripts.run_ablation import VARIANTS, build_variant_config


def test_variant_set_and_overrides():
    assert list(VARIANTS) == ["full", "no_var", "no_dir", "no_cover"]

    # full 不覆盖任何字段 = config 默认 = 主实验实测超参
    full = build_variant_config("full")
    assert full.lambda_var == config.MCDISP_ALIGN_LAMBDA_VAR
    assert full.lambda_cov == config.MCDISP_ALIGN_LAMBDA_COV
    assert full.cov_rank == config.MCDISP_ALIGN_COV_RANK
    assert full.lambda_cover_pos == config.MCDISP_ALIGN_LAMBDA_COVER_POS

    assert build_variant_config("no_var").lambda_var == 0.0

    no_dir = build_variant_config("no_dir")
    assert no_dir.cov_rank == 0
    assert no_dir.lambda_cov == 0.0

    assert build_variant_config("no_cover").lambda_cover_pos == 0.0


def test_shared_training_controls():
    for name in VARIANTS:
        cfg = build_variant_config(name)
        assert cfg.seed == config.SEED == 42
        assert cfg.dataset == "coco"
        assert cfg.select_by == "recall"      # 与主实验同 checkpoint 选择标准
        assert cfg.no_early_stop is True      # 固定预算
        assert cfg.checkpoint_dir == config.CHECKPOINT_DIR / "ablation_v2"
        assert str(cfg.best_path).endswith(f"{name}_coco_best.pt")


def test_unknown_variant_raises():
    try:
        build_variant_config("no_such")
    except KeyError:
        return
    raise AssertionError("未知变体应抛 KeyError")
