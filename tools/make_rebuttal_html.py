"""Rebuttal 结果 HTML 汇总生成器。

收集本机与服务器的集合级检索评估 JSON，生成单个 HTML 报告存到项目根目录。
可重复运行：跑完的进表，没跑完的标"待补"。

数据源：
  本机  outputs/setlevel_rebuttal/*.json           (论文协议, cos 列)
  服务器 outputs/setlevel_rebuttal/*.json          (论文协议)
  服务器 outputs/ablation_study/<exp>/seed42/eval.json (manifest 协议, set_mc/set_cos)
用法：
  python tools/make_rebuttal_html.py
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOST = "yxzhou@175.102.130.78"
KEY = str(ROOT / "yxzhou.id")
SSH = ["ssh", "-i", KEY, HOST]
SCP = ["scp", "-i", KEY]
REMOTE = "/mydata/home/yxzhou/kxiong/mcdispAlign/DistributionAlignment"
OUT_HTML = ROOT / "rebuttal实验结果汇总_20261116.html"

SERVER_SETLEVEL = {
    "prolip_l14": "outputs/setlevel_rebuttal/prolip_l14_coco.json",
    "clip_setft": "outputs/setlevel_rebuttal/clip_setft_coco.json",
    "mcdisp_full_model": "outputs/setlevel_rebuttal/mcdisp_full_model_coco.json",
    "mcdisp_no_align": "outputs/setlevel_rebuttal/mcdisp_no_align_coco.json",
}
MANIFEST_EVALS = {  # exp 名 -> (显示名, 组别, 结果来源: server|local)
    "full_kl_k1": ("Full-K1（每图 1 条描述）", "exp5", "server"),
    "full_kl_k3": ("Full-K3（每图 3 条描述）", "exp5", "server"),
    "full_kl_k5": ("Full-K5（每图 5 条描述，对照组）", "exp5", "server"),
    "full_model": ("Full MCDisp-Align（消融基准 A0）", "ablation", "server"),
    "no_align": ("w/o L_align（A1，整条对齐损失移除）", "ablation", "server"),
    "no_kl": ("w/o KL Alignment（A2）", "ablation", "server"),
    "no_cover": ("w/o Caption Coverage（A3）", "ablation", "local"),
    "no_direction": ("w/o Direction Alignment（A4）", "ablation", "local"),
    "no_reg": ("w/o Variance Regularization（A5）", "ablation", "server"),
    "diag_only": ("Diagonal-only r=0（A6）", "ablation", "server"),
}
LOCAL_SETLEVEL = {
    "clip_ft": ROOT / "outputs/setlevel_rebuttal/clip_ft_coco.json",
    "probvlm": ROOT / "outputs/setlevel_rebuttal/probvlm_coco.json",
    "pcmepp": ROOT / "outputs/setlevel_rebuttal/pcmepp_coco.json",
}


def fetch_server_files():
    """scp 服务器 JSON 到本地暂存目录；失败返回空（标待补）。"""
    stage = ROOT / "outputs/setlevel_rebuttal/_server"
    stage.mkdir(parents=True, exist_ok=True)
    got = {}
    for name, rel in SERVER_SETLEVEL.items():
        dst = stage / f"{name}.json"
        r = subprocess.run(SCP + [f"{HOST}:{REMOTE}/{rel}", str(dst)],
                           capture_output=True, timeout=120)
        got[name] = dst if r.returncode == 0 and dst.exists() else None
    me = stage / "manifest_evals"
    me.mkdir(exist_ok=True)
    for exp, (_disp, _grp, src) in MANIFEST_EVALS.items():
        if src == "local":
            local = ROOT / "outputs/ablation_study" / exp / "seed42" / "eval.json"
            got[exp] = local if local.exists() else None
            continue
        dst = me / f"{exp}.json"
        rel = f"outputs/ablation_study/{exp}/seed42/eval.json"
        r = subprocess.run(SCP + [f"{HOST}:{REMOTE}/{rel}", str(dst)],
                           capture_output=True, timeout=120)
        if r.returncode != 0 or not dst.exists():
            got[exp] = None
        else:
            got[exp] = dst
    mech = stage / "mechanism_diagnostics_full_model.json"
    r = subprocess.run(
        SCP + [f"{HOST}:{REMOTE}/outputs/ablation_study/mechanism_diagnostics_full_model.json",
               str(mech)], capture_output=True, timeout=120)
    got["mech"] = mech if r.returncode == 0 and mech.exists() else None
    return got


def _read_json(path):
    if path is None:
        return None
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _flat_kv(d, prefix=""):
    """嵌套 dict -> ['path = value'] 列表，只保留标量。"""
    rows = []
    if not isinstance(d, dict):
        return rows
    for k, v in d.items():
        if isinstance(v, bool):
            rows.append((prefix + k, str(v)))
        elif isinstance(v, (int, float)):
            rows.append((prefix + k, f"{v:.4f}" if isinstance(v, float) else str(v)))
        elif isinstance(v, dict):
            rows += _flat_kv(v, prefix + k + ".")
    return rows


def read_records(path):
    """append_eval_results 格式（JSON 数组）取最后一条；dict 直接返回。"""
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        return None
    if isinstance(data, list) and data:
        return data[-1]
    if isinstance(data, dict):
        return data
    return None


def fmt(v, pct=True):
    if v is None:
        return '<span class="pending">待补</span>'
    try:
        return f"{float(v) * 100:.2f}" if pct else f"{v:.4f}"
    except (TypeError, ValueError):
        return str(v)


def cos_row(rec):
    """论文协议记录 -> {i2t@k, t2i@k}。"""
    if not rec:
        return None
    m = rec.get("metrics", {})
    out = {}
    for k in (1, 5, 10):
        out[f"i2t@{k}"] = m.get(f"cos_i2t@{k}")
        out[f"t2i@{k}"] = m.get(f"cos_t2i@{k}")
    return out


def csd_row(rec):
    if not rec:
        return None
    m = rec.get("metrics", {})
    return {f"i2t@{k}": m.get(f"csd_i2t@{k}") for k in (1, 5, 10)} | \
           {f"t2i@{k}": m.get(f"csd_t2i@{k}") for k in (1, 5, 10)}


def manifest_row(path):
    """run.py --phase eval 的 eval.json -> set_mc/set_cos 六个 R + mR。"""
    if path is None:
        return None
    try:
        d = json.loads(Path(path).read_text())
        return d["retrieval"]
    except Exception:
        return None


def mr_of(retr):
    if not retr:
        return None
    vals = []
    for d in ("i2t", "t2i"):
        blk = retr.get(f"set_mc", {})
        for k in (1, 5, 10):
            v = blk.get(f"{d}_r@{k}")
            if isinstance(v, (int, float)):
                vals.append(v)
    return sum(vals) / len(vals) if vals else None


def mr_cell(retr):
    m = mr_of(retr)
    if m is None:
        return '<span class="pending">待补</span>'
    return f"{m * 100:.2f}"


def retr_cells(retr, key, direction, ks=(1, 5, 10)):
    if not retr:
        return ['<td><span class="pending">待补</span></td>'] * len(ks)
    blk = retr.get(key, {})
    out = []
    for k in ks:
        v = blk.get(f"{direction}_r@{k}")
        out.append(f"<td>{v * 100:.2f}</td>" if isinstance(v, (int, float))
                   else '<td><span class="pending">待补</span></td>')
    return out


PAPER_T1 = [  # 论文 Table 1：一对多（集合）检索，各模型用自家聚合方式打分（原表分数）
    ("CLIP (zero-shot)", "0.625", "0.844", "0.910", "0.511", "0.751", "0.839"),
    ("CLIP-FT", "0.825", "0.951", "0.979", "0.761", "0.932", "0.966"),
    ("ProLIP (zero-shot, ViT-H/14)", "0.630", "0.840", "0.902", "0.524", "0.748", "0.833"),
    ("ProLIP-FT (ViT-H/14)", "0.774", "0.932", "0.962", "0.738", "0.917", "0.956"),
    ("GroVE", "0.573", "0.819", "0.893", "0.444", "0.713", "0.805"),
    ("MCDisp-Align（本文）", "0.852", "0.974", "0.992", "0.839", "0.970", "0.988"),
]
PAPER_T2 = [  # 论文 Table 2：标准检索 MSCOCO
    ("CLIP (zero-shot)", "0.560", "0.792", "0.859", "0.359", "0.602", "0.705"),
    ("CLIP-FT", "0.676", "0.881", "0.940", "0.546", "0.803", "0.880"),
    ("ProLIP (zero-shot)", "0.564", "0.786", "0.858", "0.360", "0.591", "0.687"),
    ("ProLIP-FT", "0.731", "0.911", "0.954", "0.565", "0.805", "0.879"),
    ("GroVE", "0.514", "0.756", "0.838", "0.305", "0.559", "0.670"),
    ("MCDisp-Align（本文）", "0.696", "0.906", "0.955", "0.553", "0.815", "0.891"),
]
PAPER_T4 = [  # 论文 Table 4：消融（标准检索）
    ("MSCOCO w/o L_align", "0.608", "0.865", "0.939", "0.481", "0.752", "0.838"),
    ("MSCOCO MCDisp-Align", "0.696", "0.906", "0.955", "0.553", "0.815", "0.891"),
    ("Flickr30K w/o L_align", "0.691", "0.904", "0.964", "0.597", "0.861", "0.924"),
    ("Flickr30K MCDisp-Align", "0.769", "0.941", "0.976", "0.641", "0.886", "0.939"),
]


def _paper_table(rows, title):
    head = ("<tr><th rowspan='2'>模型</th><th colspan='3'>I2T R@1/5/10</th>"
            "<th colspan='3'>T2I R@1/5/10</th></tr>"
            + "<tr>" + "<th>R@1</th><th>R@5</th><th>R@10</th>" * 2 + "</tr>")
    body = "".join(
        f"<tr{' class=paper' if 'MCDisp' in r[0] and 'w/o' not in r[0] else ''}>"
        + f"<td style='text-align:left'>{r[0]}</td>" + "".join(f"<td>{v}</td>" for v in r[1:]) + "</tr>"
        for r in rows)
    return (f"<h3>{title}</h3><table>{head}{body}</table>")


def status_badge(text, cls):
    return f'<span class="badge {cls}">{text}</span>'


def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    got = fetch_server_files()

    local = {n: read_records(p) for n, p in LOCAL_SETLEVEL.items()}
    server = {n: read_records(p) if p else None for n, p in got.items()
              if n in SERVER_SETLEVEL}
    man = {e: manifest_row(got.get(e)) for e in MANIFEST_EVALS}

    css = """
    body{font-family:-apple-system,'Noto Sans SC','Microsoft YaHei',sans-serif;
         margin:2rem auto;max-width:1100px;color:#1a1a1a;line-height:1.5}
    h1{font-size:1.5rem;border-bottom:3px solid #2563eb;padding-bottom:.5rem}
    h2{font-size:1.15rem;margin-top:2rem;color:#1e3a8a}
    table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.9rem}
    th,td{border:1px solid #d1d5db;padding:.35rem .6rem;text-align:center}
    th{background:#eff6ff;font-weight:600}
    td:first-child{text-align:left;white-space:nowrap}
    tr:nth-child(even) td{background:#fafafa}
    .badge{display:inline-block;padding:.1rem .55rem;border-radius:999px;
           font-size:.78rem;font-weight:600}
    .ok{background:#dcfce7;color:#166534}.run{background:#fef9c3;color:#854d0e}
    .pend{background:#e5e7eb;color:#374151}
    .pending{color:#9ca3af}
    .note{color:#6b7280;font-size:.82rem;margin:.3rem 0 1.2rem}
    .meta{color:#6b7280;font-size:.85rem}
    .paper{background:#fff7ed}
    """

    def baseline_row(name, row, note="", badge=None):
        if row is None:
            cells = "".join('<td><span class="pending">待补</span></td>' for _ in range(6))
            b = badge or status_badge("评估中", "run")
        else:
            cells = "".join(f"<td>{fmt(row.get(f'{d}@{k}'))}</td>"
                            for d in ("i2t", "t2i") for k in (1, 5, 10))
            b = badge or status_badge("已完成", "ok")
        return f"<tr><td>{name}{note}</td><td>{b}</td>{cells}</tr>"

    def paper_row(name, vals, note=""):
        return (f'<tr class="paper"><td>{name}'
                + (f'<div class="note">{note}</div>' if note else "")
                + '</td><td><span class="badge pend">论文值</span></td>'
                + "".join(f"<td>{v}</td>" for v in vals) + "</tr>")

    # ---- 表1：消融（manifest 协议） ----
    abl_order = ["full_model", "no_align", "no_kl", "no_cover", "no_direction", "no_reg", "diag_only"]
    abl_names = {
        "full_model": "Full MCDisp-Align（A0 基准）",
        "no_align": "w/o L_align（A1）",
        "no_kl": "w/o KL Alignment（A2）",
        "no_cover": "w/o Caption Coverage（A3）",
        "no_direction": "w/o Direction Alignment（A4）",
        "no_reg": "w/o Variance Regularization（A5）",
        "diag_only": "Diagonal-only r=0（A6）",
    }
    abl_pending_note = {
        "no_cover": "本机训练中，完成后评估",
        "no_direction": "本机排队（no_cover 结束后自动开始）",
        "no_reg": "服务器训练中，完成后评估",
        "diag_only": "服务器训练中，完成后评估",
    }
    t1 = ["<table><tr><th>实验组</th><th>状态</th>"
          + "".join(f'<th>I2T R@{k}</th>' for k in (1, 5, 10))
          + "".join(f'<th>T2I R@{k}</th>' for k in (1, 5, 10))
          + "<th>mR</th></tr>"]
    for g in abl_order:
        if g in man:
            retr = man[g]
            if retr:
                cells = "".join(retr_cells(retr, "set_mc", "i2t") + retr_cells(retr, "set_mc", "t2i"))
                t1.append(f'<tr><td>{abl_names[g]}</td><td>{status_badge("已完成", "ok")}</td>'
                          f'{cells}<td>{mr_cell(retr)}</td></tr>')
            else:
                note = abl_pending_note.get(g, "评估自动接续中")
                t1.append(f'<tr><td>{abl_names[g]}<div class="note">{note}</div></td>'
                          f'<td>{status_badge("进行中", "run")}</td>'
                          + "<td>" + '<span class="pending">待补</span></td>' * 7 + "</tr>")
        else:
            t1.append(f'<tr><td>{abl_names[g]}<div class="note">{abl_pending_note[g]}</div></td>'
                      f'<td>{status_badge("训练中", "pend")}</td>'
                      + "<td>" + '<span class="pending">待补</span>' + "</td>" * 7 + "</tr>")
    t1.append("</table>")

    # ---- 表2：exp5 描述数量 ----
    t2 = ["<table><tr><th>实验组</th><th>状态</th>"
          + "".join(f'<th>I2T R@{k}</th>' for k in (1, 5, 10))
          + "".join(f'<th>T2I R@{k}</th>' for k in (1, 5, 10))
          + "<th>mR</th></tr>"]
    for e in ("full_kl_k1", "full_kl_k3", "full_kl_k5"):
        retr = man.get(e)
        if retr:
            cells = "".join(retr_cells(retr, "set_mc", "i2t") + retr_cells(retr, "set_mc", "t2i"))
            t2.append(f'<tr><td>{MANIFEST_EVALS[e][0]}</td><td>{status_badge("已完成", "ok")}</td>'
                      f'{cells}<td>{mr_cell(retr)}</td></tr>')
        else:
            t2.append(f'<tr><td>{MANIFEST_EVALS[e][0]}</td><td>{status_badge("评估中", "run")}</td>'
                      + "<td>" + '<span class="pending">待补</span></td>' * 7 + "</tr>")
    t2.append("</table>")

    # ---- 表3：基线（论文协议） ----
    t3 = ["<table><tr><th>模型</th><th>状态</th>"
          + "".join(f'<th>I2T R@{k}</th>' for k in (1, 5, 10))
          + "".join(f'<th>T2I R@{k}</th>' for k in (1, 5, 10)) + "</tr>"]
    t3.append(paper_row("CLIP-L/14 FT（论文 Table 1）",
                        ("82.5", "95.1", "97.9", "76.1", "93.2", "96.6"),
                        note="论文原报值，自家聚合打分口径"))
    t3.append(baseline_row("CLIP-L/14 FT（原版，复评·统一余弦）", cos_row(local.get("clip_ft"))))
    t3.append(baseline_row("CLIP-L/14 Set-FT（五句集合微调，复评）", cos_row(server.get("clip_setft"))))
    t3.append(baseline_row("ProbVLM", cos_row(local.get("probvlm"))))
    t3.append(baseline_row("PCME++（本机 11-12 定稿版）", cos_row(local.get("pcmepp"))))
    pr = cos_row(server.get("prolip_l14"))
    t3.append(baseline_row("ProLIP-L/14（自训，复评）", pr,
                           note='<div class="note">CSD 原生打分 R@1：见下方附注</div>'))
    t3.append(paper_row("ProLIP-FT（论文 Table 1，ViT-H/14）",
                        ("77.4", "93.2", "96.2", "73.8", "91.7", "95.6"),
                        note="论文原报值（H/14 骨干，与 L/14 行不可直接比）"))
    t3.append(paper_row("GroVE（论文 Table 1）",
                        ("57.3", "81.9", "89.3", "44.4", "71.3", "80.5")))
    t3.append(paper_row("MCDisp-Align Full（论文 Table 1）",
                        ("85.2", "97.4", "99.2", "83.9", "97.0", "98.8"),
                        note="自家矩匹配聚合打分，口径见第〇节"))
    t3.append(baseline_row("MCDisp-Align w/o L_align（复评·统一余弦）", cos_row(server.get("mcdisp_no_align"))))
    t3.append("</table>")

    prolip_csd = csd_row(server.get("prolip_l14"))
    csd_note = ""
    if prolip_csd:
        csd_note = ("ProLIP CSD 原生打分 R@1/5/10：I2T "
                    + "/".join(fmt(prolip_csd.get(f"i2t@{k}")) for k in (1, 5, 10))
                    + "，T2I "
                    + "/".join(fmt(prolip_csd.get(f"t2i@{k}")) for k in (1, 5, 10)))

    # ---- exp4：分档集合检索 + 泄漏检查（优先用干净重抽的 400 张子集） ----
    exp4 = (_read_json(ROOT / "outputs/setlevel_rebuttal/exp4_results_clean.json")
            or _read_json(ROOT / "outputs/setlevel_rebuttal/exp4_results.json"))
    overlap = (_read_json(ROOT / "outputs/setlevel_rebuttal/exp4_overlap_clean.json")
               or _read_json(ROOT / "outputs/setlevel_rebuttal/exp4_overlap_report.json"))
    t4 = ""
    if exp4:
        model_rows = []
        for fam, disp in (("clip", "CLIP-FT"), ("mcdisp", "MCDisp Full")):
            for tag, r in (exp4.get(fam) or {}).items():
                sr = r.get("set_recall", {})
                name = {"clip_ft": "CLIP-L/14 FT（原版）", "setft": "CLIP-L/14 Set-FT",
                        "full": "MCDisp-Align Full"}.get(tag, tag)
                cells = []
                for b in ("low", "mid", "high", "overall"):
                    blk = sr.get(b, {})
                    cells.append(fmt(blk.get("set_i2t_r1")))
                    cells.append(fmt(blk.get("set_t2i_r1")))
                model_rows.append(f"<tr><td>{name}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        t4 = ('<table><tr><th rowspan="2">模型</th><th colspan="2">low（重复多）</th>'
              '<th colspan="2">mid</th><th colspan="2">high（各说各的）</th>'
              '<th colspan="2">overall</th></tr>'
              '<tr>' + "<th>I2T R@1</th><th>T2I R@1</th>" * 4 + "</tr>"
              + "".join(model_rows) + "</table>"
              + '<p class="note">协议：干净子集（对旧划分模型与消融模型均未训练过的图），'
                '集合中心余弦，N 对 N。分档由 caption 文本统计初标（三分位），已获人工授权采用。'
                '首版 90 张子集经泄漏检查发现 82 张为旧划分训练图，已作废并用干净池重抽。'
                '本次未构造难负例（hard_negatives=0），难负例检查留待补充。</p>')
    else:
        t4 = '<p class="note">exp4 运行中，结果落地后自动填入。</p>'
    overlap_html = ""
    if overlap:
        kvs = _flat_kv(overlap)
        overlap_html = ('<p class="note">泄漏检查（90 图 vs 旧 90/10 训练划分）：'
                        + "；".join(f"{k}={v}" for k, v in kvs[:8]) + "</p>")

    # ---- exp3：机制诊断 ----
    mech = _read_json(got.get("mech"))
    if mech:
        rows = _flat_kv(mech)
        t5 = ("<table><tr><th>指标</th><th>数值</th></tr>"
              + "".join(f"<tr><td style='text-align:left'>{k}</td><td>{v}</td></tr>"
                        for k, v in rows[:60]) + "</table>")
    else:
        t5 = '<p class="note">机制诊断排队中（评估链结束后自动运行），结果落地后自动填入。</p>'

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>MCDisp-Align Rebuttal 实验结果汇总</title><style>{css}</style></head><body>
<h1>MCDisp-Align Rebuttal 实验结果汇总（集合级检索）</h1>
<p class="meta">生成时间：{now}　|　评估协议：集合级 N 图对 N 描述集合，I2T/T2I Recall@1/5/10　|
本页复评数值单位：百分制。本页由 <code>tools/make_rebuttal_html.py</code> 自动生成，可重复运行刷新。</p>

<h2>〇、论文原报值（paper.pdf，未复现，作为 Full 模型的基准口径）</h2>
<p class="note">按 2026-11-16 决定：MCDisp-Align Full 的指标以论文原文为准，不再额外复现。
注意论文 Table 1 的集合检索是各模型用<b>自家聚合方式</b>打分（CLIP 用嵌入平均、ProLIP 用分布聚合、
MCDisp 用矩匹配），与下文复评表的"统一中心余弦"列不是同一口径，不能同表混比；数值为论文原表的分数形式。</p>
{_paper_table(PAPER_T1, "论文 Table 1：一对多（集合）检索 MSCOCO（主实验）")}
{_paper_table(PAPER_T2, "论文 Table 2：标准检索 MSCOCO")}
{_paper_table(PAPER_T4, "论文 Table 4：w/o L_align 消融（标准检索）")}

<h2>一、逐项消融【集合检索·干净协议】（回答 R1-W4 / R2-Q3）</h2>
<p class="note">任务：集合检索。协议：legacy-val 测试清单（旧 90/10 划分的验证集，11814 张，
消融组训练时未见过——修正了此前 manifest 测试集与旧划分训练集重叠的泄漏问题），每图 5 描述，
主列 set_mc = MCDisp-Align 不确定性折扣打分；8 组同批训练同协议复评，组内自洽。
注意：Full（A0）在本表是<b>消融实验的内部基准</b>（与各消融组同批同协议，必须同表可比），
其数字不能与论文原报值互换——论文的 Full 集合检索（Table 1）与标准检索（Table 2）数字见第〇节，
任务与测试集均不同。A7（no_sg）经确认不训练，文字回应。</p>
{''.join(t1)}

<h2>二、每图训练描述数量 1/3/5【集合检索·干净协议】（回答 R3）</h2>
<p class="note">协议同上。K=1 时描述间方差为零，方向项自动关闭；K=5 组为同perm顺序对照。</p>
{''.join(t2)}

<h2>三、同骨干基线与集合微调对照【集合检索】（回答 R1-W2 / R1-W3）</h2>
<p class="note">任务：集合检索（N 图对 N 描述集合）。Full MCDisp-Align 一律以论文 Table 1
原报值为准（自家矩匹配聚合打分）；其余行为本次复评（统一中心余弦，COCO 5000 样本子集，seed 42），
橙底=论文值、白底=复评，两种打分口径不可直接横比。标准检索的论文数字见第〇节 Table 2。
Flickr30K 按决定不做。</p>
{''.join(t3)}
{f'<p class="note">{csd_note}</p>' if csd_note else ''}

<h2>四、描述互补性分档对比【集合检索·干净子集】（回答 R2-W1/W2，exp4）</h2>
{t4}
{overlap_html}

<h2>五、机制诊断（回答 R1-W1，exp3）</h2>
<p class="note">基于 full_model 检查点：覆盖激活率、边际方差、文本方差、常数方差重排、各项梯度路径。</p>
{t5}

<h2>六、进行中任务</h2>
<ul class="meta">
<li>消融 4 组训练：no_cover（本机，进行中）、no_direction（本机，排队自动接续）、
no_reg / diag_only（服务器 GPU0/2）——完成后跑同一 manifest 评估并刷新本页</li>
<li>评估链：服务器 GPU0 串行评估 exp5 三组 + 消融已完成 3 组 + ProLIP/Set-FT/MCDisp 复评</li>
</ul>

<p class="note">权重来源：CLIP-FT=9 月备份（checkpoints 根目录已无 11 月版）；
PCME++=本机 11-12 定稿官方复现；ProLIP/Set-FT/exp5=服务器 11 月版；
消融组=11 月同批（解冻骨干、KL 目标、seed 42、10 epochs）。</p>
</body></html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"written: {OUT_HTML}")
    done = sum(1 for r in list(local.values()) + list(server.values()) if r) \
        + sum(1 for r in man.values() if r)
    print(f"records included: {done}")


if __name__ == "__main__":
    sys.exit(main())
