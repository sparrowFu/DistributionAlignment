# 实验 4 离线人工标注

`build_annotation_pack.py` 只使用 Python 标准库和 Pillow。它将已经抽样的模板和图片打包成单个离线 HTML，不调用模型，不自动分档或改写事实，不展示模型结果。

```bash
python build_annotation_pack.py \
  --subset outputs/rebuttal_exp4_20261115/subset_template.json \
  --images-dir /实际图片根目录 \
  --output outputs/rebuttal_exp4_20261115/annotation.html
```

可选 `--originals-zip outputs/rebuttal_exp4_20261115/originals.zip`，另存原图压缩包用于核对细节。HTML 仅嵌入最大边 960 像素的 JPEG 缩略图，不包含原始大图。脚本拒绝覆盖已有 HTML 或原图压缩包；修改输出名保留旧材料。

将 `annotation.html` 拷到本机并用浏览器打开。每张图的五条描述固定编号 0–4，填写 `low/mid/high`、至少一条困难负例及其 `replaces`、`note`，并勾选看图核验。困难负例只改一个事实；note 写明改了什么及图像核验依据。不要为凑每档 30 张而改变分档。

- “保存草稿 JSON”：允许缺项，下载 `subset_draft.json`。
- “导入 JSON 继续标注”：逐张核对图像 ID、顺序、五句原文，拒绝不匹配的文件。
- “导出全部人工核验结果”：所有图片完整填写、至少一条负例、说明非空且已勾选核验后才导出 `subset_verified.json`。修改分档或负例后需要重新确认。

浏览器会尽力自动保存本机草稿，但本地页面存储可能受限制，请定期下载草稿文件。导出的核验 JSON 保留模板元数据，可直接传给 `scripts/exp4_complementarity.py --subset`。

采样来源和划分由模板元数据提供，页面原样展示。旧验证集分析不能称为独立测试集。缩略图无法辨认的事实需查看原图，不应凭推测标为已核验。
