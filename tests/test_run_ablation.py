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


def test_variant_set_and_overrides():
    assert list(VARIANTS) == ["full", "no_var", "no_dir", "no_ctr"]

    # full 不覆盖任何字段 = config 默认 = 主实验超参
    full = build_variant_config("full")
    assert full.lambda_var == config.MCDISP_ALIGN_LAMBDA_VAR
    assert full.lambda_dir == config.MCDISP_ALIGN_LAMBDA_DIR
    assert full.lambda_ctr == config.MCDISP_ALIGN_LAMBDA_CTR
    assert full.cov_rank == config.MCDISP_ALIGN_COV_RANK

    assert build_variant_config("no_var").lambda_var == 0.0

    no_dir = build_variant_config("no_dir")
    assert no_dir.cov_rank == 0
    assert no_dir.lambda_dir == 0.0

    assert build_variant_config("no_ctr").lambda_ctr == 0.0


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
    }  # no_dir / no_cover 缺失 -> 跳过
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
