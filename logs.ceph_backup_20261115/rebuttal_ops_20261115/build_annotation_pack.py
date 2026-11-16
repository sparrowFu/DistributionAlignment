#!/usr/bin/env python3
"""Build an offline, model-blind annotation page from an exp-4 subset template.

Only the Python standard library and Pillow are required. The resulting HTML
contains JPEG thumbnails, the unchanged five captions and a manual annotation
form; it performs no classification or caption generation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

from PIL import Image, ImageOps


PAGE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>实验 4 · 人工核验标注</title>
<style>
:root{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#172536;background:#f2f5f8;line-height:1.6}
*{box-sizing:border-box}body{margin:0}header{background:#fff;border-bottom:1px solid #dce4ec;padding:22px max(20px,calc((100vw - 1260px)/2))}
h1{font-size:24px;margin:0 0 8px}p{margin:8px 0}.intro{max-width:1160px;color:#44556a}.warning{border-left:4px solid #d59c32;padding:8px 12px;background:#fff6e3;color:#67470f}
main{max-width:1260px;margin:18px auto;padding:0 20px 35px}.toolbar,.card,.metadata{background:#fff;border:1px solid #dce4ec;border-radius:10px;padding:16px;margin-bottom:16px}
.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center}.grow{flex:1;min-width:200px}.status{font-weight:600}.counts{font-size:14px;color:#52657a}
button,.file-label,select,input,textarea{font:inherit}button,.file-label{border:1px solid #aebed0;background:#fff;color:#263c55;border-radius:6px;padding:7px 12px;cursor:pointer}
button:hover,.file-label:hover{background:#eaf1f8}button.primary{background:#215f96;color:white;border-color:#215f96}button:disabled{opacity:.45;cursor:default}
button.remove{color:#9c3030;font-size:14px;padding:5px 10px}.file-label input{display:none}.split{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.1fr);gap:22px}
.image-panel{position:sticky;top:18px;align-self:start}img{width:100%;max-height:650px;object-fit:contain;background:#e9eef4;border-radius:6px}
.filename{font-family:ui-monospace,monospace;word-break:break-all;font-size:13px;color:#576b80}h2{font-size:20px;margin:0 0 10px}h3{font-size:17px;margin:20px 0 8px}
ol{padding-left:26px}li{padding-left:4px;margin:8px 0}.caption-index{color:#536c83;font-size:13px}fieldset{border:1px solid #ccd8e4;border-radius:7px;padding:10px 12px}legend{font-weight:600;padding:0 5px}
.radio{display:block;margin:6px 0}.radio input{margin-right:8px}.hint{font-size:14px;color:#52657a}.negative{border:1px solid #d5dfeb;background:#f7f9fc;border-radius:7px;padding:12px;margin:10px 0}
.negative-top{display:flex;align-items:center;justify-content:space-between;gap:10px}label.field{display:block;font-size:14px;margin:8px 0}label.field textarea{display:block;width:100%;min-height:68px;resize:vertical}
textarea,select,input[type=number]{border:1px solid #acbdd0;border-radius:5px;padding:6px;background:white;color:#172536}select{max-width:100%}.confirmation{display:block;padding:12px;border:1px solid #c9d8e5;border-radius:7px;background:#edf5fc;margin-top:16px}
.confirmation input{margin-right:8px}.error{background:#fff1ef;border:1px solid #e6b0a8;padding:10px;border-radius:6px;color:#8e3028;white-space:pre-line;display:none}
.note{padding:10px 0;color:#52657a;font-size:13px}pre{font-family:ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:340px;overflow:auto}
.footer-nav{display:flex;gap:12px;align-items:center;margin-top:18px}.keyboard{font-size:12px;color:#63768a}@media(max-width:850px){.split{grid-template-columns:1fr}.image-panel{position:static}img{max-height:500px}header{padding:18px 20px}}
</style>
</head>
<body>
<header>
<h1>实验 4 · 描述互补性与困难负例人工核验</h1>
<p class="intro">只查看图片和五条原始描述，人工完成分档与单事实修改。页面不展示模型结果，不会自动分类或生成难负例。</p>
<p class="warning">数据来源与划分以本页元数据为准。旧验证集可用于验证集分析，不能称为独立测试集。不要为凑每档 30 张而修改分档。</p>
</header>
<main>
<div class="toolbar">
  <div class="grow"><div id="progress" class="status"></div><div id="counts" class="counts"></div></div>
  <button id="save-draft">保存草稿 JSON</button>
  <label class="file-label">导入 JSON 继续标注<input id="import-file" type="file" accept=".json,application/json"></label>
  <button id="export-verified" class="primary">导出全部人工核验结果</button>
</div>
<div class="toolbar"><button id="prev">上一张</button><select id="jump" aria-label="选择图片"></select><button id="next">下一张</button><span class="hint" id="autosave"></span></div>
<div id="message" class="error" role="alert"></div>
<section class="card split">
  <div class="image-panel"><img id="photo" alt="待人工核验的采样图片"><div id="filename" class="filename"></div><p class="hint">缩略图最大边不超过 960 像素；若细节不清，请先查看独立提供的原图再确认。无法辨认的事实不能标作已核验。</p></div>
  <div>
    <h2 id="entry-title"></h2>
    <h3>五条原始描述（下标 0–4）</h3><ol id="captions" start="0"></ol>
    <fieldset><legend>人工分档</legend>
      <label class="radio"><input type="radio" name="bin" value="low">low · 低互补：五句反复描述相同对象、动作或事实</label>
      <label class="radio"><input type="radio" name="bin" value="mid">mid · 中互补：部分事实重复，部分描述补充信息</label>
      <label class="radio"><input type="radio" name="bin" value="high">high · 高互补：不同描述补充不同对象、属性或关系</label>
    </fieldset>
    <h3>人工构造困难负例</h3><p class="hint">先选择被替换的原句下标；仅修改一个可见事实，并核对原句成立、改写句与图片冲突。每张图至少一条；note 必须写明修改了哪个事实以及看图核验依据。</p>
    <div id="negatives"></div><button id="add-negative">添加一条困难负例</button>
    <label class="confirmation"><input id="confirmed" type="checkbox">我已看图核验分档、五句原文与每条困难负例：原句属实，负例只改一个事实且与图像冲突。</label>
    <p class="note">修改分档或负例后，需重新勾选核验。草稿允许留空；正式导出要求所有图片填写完整且已人工确认。</p>
  </div>
</section>
<div class="footer-nav"><button id="prev-bottom">上一张</button><button id="next-bottom">下一张</button><button id="first-incomplete">跳到首张未完成图片</button></div>
<details class="metadata"><summary>采样来源与划分元数据（原样保留）</summary><pre id="metadata"></pre></details>
<p class="note">页面离线运行。浏览器会尽力保存本机草稿，但 file:// 页面存储可能受浏览器限制；请定期下载“保存草稿 JSON”。导入时会逐张核对图像 ID、顺序和五条原文。导出的核验文件可直接作为 exp4_complementarity.py 的 --subset 输入。</p>
</main>
<script id="pack-data" type="application/json">__PACK_JSON__</script>
<script>
"use strict";
(() => {
  const pack = JSON.parse(document.getElementById("pack-data").textContent);
  const source = pack.subset;
  const entries = source.entries.map(e => ({...e, bin:e.bin || "", hard_negatives:(e.hard_negatives || []).map(h=>({...h})), human_verified:e.human_verified === true}));
  const key = "mcdisp-exp4-human-" + pack.fingerprint;
  let current = 0;
  const $ = id => document.getElementById(id);
  const clone = x => JSON.parse(JSON.stringify(x));
  function message(text) { $("message").textContent=text; $("message").style.display=text ? "block" : "none"; }
  function problems(e) {
    const p=[];
    if(!["low","mid","high"].includes(e.bin))p.push("未选择互补性分档");
    if(!e.hard_negatives.length)p.push("至少填写一条困难负例");
    e.hard_negatives.forEach((h,j)=>{
      if(typeof h.caption!=="string" || !h.caption.trim())p.push("负例 "+(j+1)+" 缺改写描述");
      if(!Number.isInteger(h.replaces) || h.replaces<0 || h.replaces>4)p.push("负例 "+(j+1)+" 缺原句下标 0–4");
      if(typeof h.note!=="string" || !h.note.trim())p.push("负例 "+(j+1)+" 缺单事实修改与核验说明");
    });
    if(!e.human_verified)p.push("未勾选看图核验");
    return p;
  }
  function isDone(e) { return problems(e).length===0; }
  function makeExport(verified) {
    const now=new Date().toISOString();
    return {...clone(source), entries:clone(entries), _annotation_status:verified?"human_verified":"draft", _annotation_tool:{version:1,pack_fingerprint:pack.fingerprint,exported_at:now}};
  }
  function updateProgress() {
    const n=entries.filter(isDone).length;
    $("progress").textContent="已完整核验 "+n+" / "+entries.length+" 张";
    const bins=["low","mid","high"].map(b=>b+": "+entries.filter(e=>e.bin===b).length);
    $("counts").textContent=bins.join("　")+"　未分档: "+entries.filter(e=>!e.bin).length;
    Array.from($("jump").options).forEach((o,i)=>{o.textContent=(i+1)+" / "+entries.length+" · "+(isDone(entries[i])?"✓ ":"○ ")+entries[i].image;});
  }
  function persist() {
    try { localStorage.setItem(key,JSON.stringify(makeExport(false))); $("autosave").textContent="本机草稿已自动保存"; }
    catch(e) { $("autosave").textContent="浏览器未允许本机存储，请下载草稿 JSON"; }
    updateProgress();
  }
  function changed() { entries[current].human_verified=false; $("confirmed").checked=false; persist(); }
  function field(labelText,control) { const label=document.createElement("label");label.className="field";label.append(document.createTextNode(labelText),control);return label; }
  function renderNegatives() {
    const container=$("negatives");container.replaceChildren();
    entries[current].hard_negatives.forEach((h,j)=>{
      const box=document.createElement("div");box.className="negative";
      const top=document.createElement("div");top.className="negative-top";
      const title=document.createElement("strong");title.textContent="困难负例 "+(j+1);
      const remove=document.createElement("button");remove.className="remove";remove.textContent="删除这条";
      remove.addEventListener("click",()=>{entries[current].hard_negatives.splice(j,1);changed();renderNegatives();});
      top.append(title,remove);box.append(top);
      const sel=document.createElement("select");
      const blank=document.createElement("option");blank.value="";blank.textContent="请选择被替换原句";sel.append(blank);
      for(let k=0;k<5;k++){const o=document.createElement("option");o.value=String(k);o.textContent="原句 "+k+" · "+entries[current].captions[k];sel.append(o);}
      sel.value=Number.isInteger(h.replaces)?String(h.replaces):"";
      sel.addEventListener("change",()=>{h.replaces=sel.value===""?null:Number(sel.value);changed();});
      const cap=document.createElement("textarea");cap.value=h.caption || "";cap.placeholder="人工填写：仅修改一个事实后的描述";
      cap.addEventListener("input",()=>{h.caption=cap.value;changed();});
      const note=document.createElement("textarea");note.value=h.note || "";note.placeholder="例如：将红色雨伞改为蓝色；图中雨伞为红色。请根据实际图像填写。";
      note.addEventListener("input",()=>{h.note=note.value;changed();});
      box.append(field("替换原句下标（与上方五句顺序一致）",sel),field("改写后的描述 caption",cap),field("单事实修改与看图核验说明 note",note));container.append(box);
    });
  }
  function render() {
    const e=entries[current];
    $("photo").src=pack.thumbnails[current];$("photo").alt="人工核验图片 "+e.image;
    $("filename").textContent=e.image;$("entry-title").textContent="图片 "+(current+1)+" / "+entries.length;
    $("captions").replaceChildren();e.captions.forEach(c=>{const li=document.createElement("li");li.textContent=c;$("captions").append(li);});
    document.querySelectorAll('input[name="bin"]').forEach(r=>{r.checked=r.value===e.bin;});
    $("confirmed").checked=e.human_verified;$("jump").value=String(current);
    ["prev","prev-bottom"].forEach(id=>{$(id).disabled=current===0;});
    ["next","next-bottom"].forEach(id=>{$(id).disabled=current===entries.length-1;});
    renderNegatives();updateProgress();
  }
  function navigate(n) { current=Math.max(0,Math.min(entries.length-1,n));message("");render(); }
  function download(verified) {
    if(verified){
      const bad=entries.map((e,i)=>({i,problems:problems(e)})).filter(x=>x.problems.length);
      if(bad.length){current=bad[0].i;render();message("还有 "+bad.length+" 张未完成，不能导出人工核验结果。\n当前第 "+(current+1)+" 张："+bad[0].problems.join("；")+"。\n可以先保存草稿。");return;}
    }
    const data=makeExport(verified);const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:"application/json;charset=utf-8"}));
    const link=document.createElement("a");link.href=url;link.download=verified?"subset_verified.json":"subset_draft.json";document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
    persist();message("");
  }
  function importAnnotations(blob) {
    if(!blob || !Array.isArray(blob.entries) || blob.entries.length!==entries.length)throw Error("导入文件图片数量不一致");
    if(blob._annotation_tool && blob._annotation_tool.pack_fingerprint && blob._annotation_tool.pack_fingerprint!==pack.fingerprint)throw Error("导入文件不属于这份采样包");
    const next=blob.entries.map((e,i)=>{
      const original=source.entries[i];
      if(!e || e.image!==original.image)throw Error("第 "+(i+1)+" 张图像 ID 或顺序不一致");
      if(!Array.isArray(e.captions) || e.captions.length!==5 || e.captions.some((c,j)=>c!==original.captions[j]))throw Error("第 "+(i+1)+" 张的五句原文不一致，拒绝导入");
      const bin=e.bin || "";if(!["","low","mid","high"].includes(bin))throw Error("第 "+(i+1)+" 张分档无效");
      const hs=e.hard_negatives===undefined?[]:e.hard_negatives;if(!Array.isArray(hs))throw Error("第 "+(i+1)+" 张 hard_negatives 必须是数组");
      const hard=hs.map((h,j)=>{
        if(!h || typeof h!=="object" || Array.isArray(h))throw Error("负例格式无效");
        const r=h.replaces===undefined || h.replaces===""?null:h.replaces;
        if(r!==null && (!Number.isInteger(r)||r<0||r>4))throw Error("第 "+(i+1)+" 张负例 "+(j+1)+" 的 replaces 必须为空或整数 0–4");
        if(h.caption!==undefined && typeof h.caption!=="string")throw Error("负例 caption 必须是文本");
        if(h.note!==undefined && typeof h.note!=="string")throw Error("负例 note 必须是文本");
        return {...h,caption:h.caption || "",replaces:r,note:h.note || ""};
      });
      return {...clone(original),bin,hard_negatives:hard,human_verified:e.human_verified===true};
    });
    next.forEach((e,i)=>{entries[i]=e;});render();
  }
  entries.forEach((e,i)=>{const o=document.createElement("option");o.value=String(i);$("jump").append(o);});
  $("jump").addEventListener("change",()=>navigate(Number($("jump").value)));
  ["prev","prev-bottom"].forEach(id=>{$(id).addEventListener("click",()=>navigate(current-1));});
  ["next","next-bottom"].forEach(id=>{$(id).addEventListener("click",()=>navigate(current+1));});
  $("first-incomplete").addEventListener("click",()=>{const i=entries.findIndex(e=>!isDone(e));if(i>=0)navigate(i);else message("全部图片已填写完整并确认，可导出核验结果。");});
  document.querySelectorAll('input[name="bin"]').forEach(r=>{r.addEventListener("change",()=>{entries[current].bin=r.value;changed();});});
  $("confirmed").addEventListener("change",()=>{entries[current].human_verified=$("confirmed").checked;persist();});
  $("add-negative").addEventListener("click",()=>{entries[current].hard_negatives.push({caption:"",replaces:null,note:""});changed();renderNegatives();});
  $("save-draft").addEventListener("click",()=>download(false));$("export-verified").addEventListener("click",()=>download(true));
  $("import-file").addEventListener("change",async()=>{
    const file=$("import-file").files[0];if(!file)return;
    try{const blob=JSON.parse(await file.text());importAnnotations(blob);persist();message("");}
    catch(e){message("导入失败："+e.message);}finally{$("import-file").value="";}
  });
  const meta={...source};delete meta.entries;$("metadata").textContent=JSON.stringify({...meta,_offline_pack:{fingerprint:pack.fingerprint,image_count:entries.length,max_thumbnail_edge:pack.max_edge}},null,2);
  try{const draft=localStorage.getItem(key);if(draft){importAnnotations(JSON.parse(draft));$("autosave").textContent="已恢复本机草稿，请定期下载备份";}}
  catch(e){$("autosave").textContent="本机草稿未恢复："+e.message;}
  render();
})();
</script>
</body>
</html>
'''


def _load_subset(path: Path) -> dict:
    blob = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(blob, dict) or not isinstance(blob.get("entries"), list) or not blob["entries"]:
        raise ValueError("subset must be an object containing a non-empty entries list")
    seen = set()
    for i, entry in enumerate(blob["entries"]):
        if not isinstance(entry, dict):
            raise ValueError(f"entry {i}: must be an object")
        name = entry.get("image")
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError(f"entry {i}: image ID must be non-empty and unique")
        seen.add(name)
        caps = entry.get("captions")
        if not isinstance(caps, list) or len(caps) != 5 or not all(
            isinstance(c, str) and c.strip() for c in caps
        ):
            raise ValueError(f"entry {i} ({name}): exactly five non-empty captions are required")
        if entry.get("bin", "") not in ("", "low", "mid", "high"):
            raise ValueError(f"entry {i} ({name}): invalid bin")
        hs = entry.get("hard_negatives", [])
        if not isinstance(hs, list):
            raise ValueError(f"entry {i} ({name}): hard_negatives must be a list")
        for j, hard in enumerate(hs):
            if not isinstance(hard, dict):
                raise ValueError(f"entry {i} negative {j}: expected an object")
            if not isinstance(hard.get("caption", ""), str) or not isinstance(hard.get("note", ""), str):
                raise ValueError(f"entry {i} negative {j}: caption/note must be strings")
            replaces = hard.get("replaces")
            if replaces not in (None, "") and (type(replaces) is not int or not 0 <= replaces <= 4):
                raise ValueError(f"entry {i} negative {j}: replaces must be blank or an integer in [0,4]")
    return blob


def _image_path(images_dir: Path, name: str) -> Path:
    # Resolve symlinks as well; never package a file outside the supplied root.
    root = images_dir.resolve(strict=True)
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"image must be a regular file inside --images-dir: {name}")
    return path


def _thumbnail(path: Path, max_edge: int, quality: int) -> str:
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original)
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, (255, 255, 255))
            rgb.paste(rgba, mask=rgba.getchannel("A"))
            image = rgb
        else:
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _safe_json(data: dict) -> str:
    # Escape script terminators and HTML syntax even inside a non-JS JSON tag.
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return text.replace("&", "\\u0026").replace("<", "\\u003c").replace(
        ">", "\\u003e"
    ).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def build_pack(subset_path: Path, images_dir: Path, output: Path,
               max_edge: int = 960, quality: int = 82,
               originals_zip: Path | None = None) -> dict:
    if not 1 <= max_edge <= 960:
        raise ValueError("--max-edge must be between 1 and 960")
    if not 30 <= quality <= 95:
        raise ValueError("--quality must be between 30 and 95")
    if output.exists():
        raise FileExistsError(f"refusing to replace an existing annotation page: {output}")
    if originals_zip is not None and originals_zip.exists():
        raise FileExistsError(f"refusing to replace an existing originals archive: {originals_zip}")
    if originals_zip is not None and originals_zip.resolve() == output.resolve():
        raise ValueError("--output and --originals-zip must be different paths")
    subset = _load_subset(subset_path)
    identity = [{"image": e["image"], "captions": e["captions"]} for e in subset["entries"]]
    fingerprint = hashlib.sha256(json.dumps(
        identity, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    paths = [_image_path(images_dir, e["image"]) for e in subset["entries"]]
    thumbnails = [_thumbnail(p, max_edge, quality) for p in paths]
    payload = {"subset": subset, "fingerprint": fingerprint,
               "thumbnails": thumbnails, "max_edge": max_edge}
    page = PAGE.replace("__PACK_JSON__", _safe_json(payload))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as f:
        f.write(page)
    if originals_zip is not None:
        originals_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(originals_zip, "x", compression=zipfile.ZIP_STORED) as archive:
            for e, path in zip(subset["entries"], paths):
                archive.write(path, arcname=path.relative_to(images_dir.resolve()).as_posix())
    return {"output": str(output), "image_count": len(paths), "fingerprint": fingerprint,
            "html_bytes": output.stat().st_size,
            "originals_zip": str(originals_zip) if originals_zip else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", required=True, type=Path,
                        help="existing subset_template.json or saved draft")
    parser.add_argument("--images-dir", required=True, type=Path,
                        help="root containing the subset's image filenames")
    parser.add_argument("--output", type=Path, default=Path("annotation.html"))
    parser.add_argument("--max-edge", type=int, default=960,
                        help="thumbnail maximum dimension; at most 960")
    parser.add_argument("--quality", type=int, default=82)
    parser.add_argument("--originals-zip", type=Path, default=None,
                        help="optional separate archive of original images; never embedded in HTML")
    args = parser.parse_args()
    print(json.dumps(build_pack(args.subset, args.images_dir, args.output,
                                args.max_edge, args.quality, args.originals_zip),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
