"""scripts/run_ablation.py（消融 v2）单元测试：变体→配置映射、报告拼装。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

from scripts.run_ablation import (
    VARIANTS,
    build_variant_config,
    build_report_rows,
    format_markdown_table,
)


# 变体可覆盖的目标字段（VARIANTS overrides 的键空间）——每个变体只允许翻转
# 自己声明的键，其余字段必须与 full（= config 默认 = 主实验超参）一致。
OBJECTIVE_FIELDS = ("lambda_match", "lambda_mu", "lambda_var", "lambda_reg",
                    "lambda_dir", "match_score", "cov_rank", "tau_match")


def test_variant_set_and_overrides():
    assert list(VARIANTS) == ["full", "no_mu", "no_var", "no_dir_loss",
                              "diagonal_only", "no_reg", "cosine_match"]

    # full 不覆盖任何字段 = config 默认 = 主实验超参
    full = build_variant_config("full")
    assert full.lambda_match == config.MCDISP_ALIGN_LAMBDA_MATCH
    assert full.lambda_mu == config.MCDISP_ALIGN_LAMBDA_MU
    assert full.lambda_var == config.MCDISP_ALIGN_LAMBDA_VAR
    assert full.lambda_reg == config.MCDISP_ALIGN_LAMBDA_REG
    assert full.lambda_dir == config.MCDISP_ALIGN_LAMBDA_DIR
    assert full.match_score == config.MCDISP_ALIGN_MATCH_SCORE
    assert full.tau_match == config.MCDISP_ALIGN_TAU_MATCH
    assert full.cov_rank == config.MCDISP_ALIGN_COV_RANK

    # 每个变体只翻转声明的 override 键，其余权重与 full 完全一致
    for name, spec in VARIANTS.items():
        cfg = build_variant_config(name)
        declared = spec["overrides"]
        for fld in OBJECTIVE_FIELDS:
            expected = declared.get(fld, getattr(full, fld))
            assert getattr(cfg, fld) == expected, (name, fld)

    # 抽样验证各声明确实生效
    assert build_variant_config("no_mu").lambda_mu == 0.0
    assert build_variant_config("no_var").lambda_var == 0.0
    assert build_variant_config("no_dir_loss").lambda_dir == 0.0
    assert build_variant_config("no_reg").lambda_reg == 0.0
    assert build_variant_config("cosine_match").match_score == "cosine"
    diagonal_only = build_variant_config("diagonal_only")
    assert diagonal_only.cov_rank == 0 and diagonal_only.lambda_dir == 0.0


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


def _fake_eval(variant, mr):
    met = {}
    for k in (1, 5, 10):
        met[f"mc_recall_i2t@{k}"] = mr + 0.01 * k
        met[f"mc_recall_t2i@{k}"] = mr
    return {"variant": variant, "mR": mr, "metrics": met}


def test_report_rows_order_and_delta():
    results = {
        "full": _fake_eval("full", 0.60),
        "no_var": _fake_eval("no_var", 0.55),
    }  # 其余变体缺失 -> 跳过
    rows = build_report_rows(results)
    assert [r["variant"] for r in rows] == ["full", "no_var"]
    assert rows[0]["delta_mR"] is None                     # full 自身无 Δ
    assert abs(rows[1]["delta_mR"] - (-0.05)) < 1e-9       # 0.55 - 0.60
    assert rows[1]["i2t R@1"] == 0.55 + 0.01


def test_markdown_table_full_coverage():
    results = {v: _fake_eval(v, 0.6 - 0.02 * i) for i, v in enumerate(VARIANTS)}
    md = format_markdown_table(build_report_rows(results))
    assert "| variant" in md and "delta_mR" in md
    assert all(f"| {v} " in md for v in VARIANTS)
    assert "i2t R@1" in md and "t2i R@10" in md


def test_markdown_table_empty():
    assert format_markdown_table([]).startswith("（暂无")


def test_write_report_smoke(tmp_path):
    from scripts.run_ablation import write_report

    results = {v: _fake_eval(v, 0.6 - 0.02 * i) for i, v in enumerate(VARIANTS)}
    md_path = write_report(results, out_dir=tmp_path)
    assert md_path.exists() and md_path.name == "report.md"
    csv_path = tmp_path / "report.csv"
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("variant") and "desc" not in lines[0]
    assert len(lines) == 1 + len(VARIANTS)
