const fs = require('fs');
const path = require('path');
const vm = require('vm');
const crypto = require('crypto');

const root = path.resolve(__dirname, '..');
const source = path.join(root, 'MCDisp-Align_Rebuttal_No_New_Experiments.html');
const output = path.join(root, 'MCDisp-Align_Rebuttal_No_New_Experiments_中文版.html');
const original = fs.readFileSync(source, 'utf8');
const sourceHash = crypto.createHash('sha256').update(original).digest('hex');
function readJson(filename) {
  return JSON.parse(fs.readFileSync(path.join(__dirname, filename), 'utf8').replace(/^\uFEFF/, ''));
}
const translations = {
  ...readJson('rebuttal_math_zh.json'),
  ...readJson('rebuttal_protocol_zh.json'),
};

let cardCount = 0;
let html = original.replace(/<article class="card" id="([^"]+)">([\s\S]*?)<\/article>/g, (article, id) => {
  const translated = translations[id];
  if (!translated || !translated.title || !translated.html) throw new Error('Missing translation: ' + id);
  if (/<script\b|<article\b|<\/article>/i.test(translated.html)) throw new Error('Unexpected translation markup: ' + id);
  let changed = article.replace(/<h3>[\s\S]*?<\/h3>/, '<h3>' + translated.title + '</h3>');
  const response = /<div class="response" lang="en">[\s\S]*?<\/div>\s*(?=<div class="editor-note)/;
  if (!response.test(changed)) throw new Error('Missing response container: ' + id);
  changed = changed.replace(response, '<div class="response" lang="zh-CN">\n' + translated.html.trim() + '\n        </div>\n        ');
  cardCount++;
  return changed;
});
if (cardCount !== 10 || Object.keys(translations).length !== 10) throw new Error('Expected exactly ten translated cards');

const substitutions = [
  ['包含英文回应、中文说明和证据边界。', '包含中文回应正文、原文依据和证据边界。'],
  ['<title>MCDisp-Align | 无新增实验 Rebuttal 初稿</title>', '<title>MCDisp-Align | 无新增实验 Rebuttal 中文初稿</title>'],
  ['REBUTTAL · WORKING DRAFT', 'REBUTTAL · 中文初稿'],
  ['REVIEWER 1', '审稿人 1'], ['REVIEWER 2', '审稿人 2'], ['REVIEWER 3', '审稿人 3'],
  ['>COVERAGE<', '>回应范围<'],
  ['AUTHOR RESPONSE · NO NEW EXPERIMENTS', '作者回复 · 无新增实验'],
  ['<h1>无需新增实验的 Rebuttal 初稿<span>英文候选正文 / 中文使用说明 / 审稿问题逐条定位</span></h1>', '<h1>无需新增实验的 Rebuttal 中文初稿<span>中文回应正文 / 审稿问题逐条定位 / 原文依据与回应边界</span></h1>'],
  ['版本：初稿 v1', '版本：中文初稿 v1'],
  ['只看英文正文', '只看回应正文'],
  ['复制全部英文', '复制全部中文回应'],
  ['>复制英文<', '>复制中文<'],
  ['英文段落可单独复制，中文说明不属于拟提交正文。', '中文回应段落可单独复制；“依据”“使用边界”等编辑说明不属于拟提交正文。'],
  ['<h2>Reviewer 1</h2>', '<h2>审稿人 1</h2>'],
  ['<h2>Reviewer 2</h2>', '<h2>审稿人 2</h2>'],
  ['<h2>Reviewer 3</h2>', '<h2>审稿人 3</h2>'],
  ['<span>Draft response</span>', '<span>中文回应初稿</span>'],
  ['<span>Conditional draft response</span>', '<span>中文条件稿 · 核对后使用</span>'],
  ['下面英文依据当前仓库实现撰写。', '以下回应根据当前仓库实现撰写。'],
  ['复制功能只复制英文候选正文；', '复制功能只复制中文回应正文；'],
  ["font-family:Georgia,\"Times New Roman\",serif;font-size:16px", "font-family:\"Segoe UI\",\"Microsoft YaHei\",sans-serif;font-size:16px"],
  ['.english-only', '.response-only'],
  ["classList.toggle('english-only')", "classList.toggle('response-only')"],
  ['已复制英文候选正文', '已复制中文回应正文'],
  ['[Conditional draft: verify the submitted checkpoint configuration before use]', '【条件稿：使用前请核对投稿 checkpoint 的实际训练配置】'],
  ['Working draft. R1/R2/R3 denote review order, not official reviewer IDs. Some responses clarify only part of a concern. The R3-C1 paragraph requires checkpoint verification.', '中文初稿。R1/R2/R3 按审稿意见出现顺序编号，并非正式审稿人 ID。部分回应仅澄清某一子问题。R3-C1 须先核对投稿 checkpoint 的实际训练配置。'],
];
for (const [before, after] of substitutions) {
  if (!html.includes(before)) throw new Error('Missing expected source text: ' + before);
  html = html.split(before).join(after);
}

const formulaLabels = {
  'gradient of coverage loss with respect to U': '覆盖损失关于低秩矩阵 U 的梯度',
  'caption set marginal variance': '描述集分布的边际方差',
  'matching score derivative with respect to a text variance coordinate': '匹配分数关于文本方差分量的导数',
  'expected normalized Mahalanobis distance': '马氏距离平方的条件期望',
};
for (const [before, after] of Object.entries(formulaLabels)) html = html.split(before).join(after);

const errors = [];
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
const links = [...html.matchAll(/href="([^"]+)"/g)].map(m => m[1]);
if (ids.length !== new Set(ids).size) errors.push('Duplicate IDs');
for (const href of links) {
  if (href.startsWith('#')) {
    if (!ids.includes(href.slice(1))) errors.push('Broken anchor: ' + href);
  } else if (!fs.existsSync(path.resolve(root, href.split('#')[0]))) errors.push('Missing source: ' + href);
}
const scripts = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)];
scripts.forEach((s, i) => new vm.Script(s[1], { filename: 'inline-' + i + '.js' }));
const cleaned = html.replace(/<(script|style)\b[^>]*>[\s\S]*?<\/\1>/g, '');
const stack = [];
const voids = new Set(['area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr']);
for (const m of cleaned.matchAll(/<(\/?)([a-z][a-z0-9]*)\b[^>]*>/gi)) {
  const tag = m[2].toLowerCase();
  if (voids.has(tag)) continue;
  if (m[1]) {
    const open = stack.pop();
    if (open !== tag) errors.push('Unbalanced tags: ' + open + ' / ' + tag);
  } else stack.push(tag);
}
if (stack.length) errors.push('Unclosed tags: ' + stack.join(','));
if ((html.match(/class="response" lang="zh-CN"/g) || []).length !== 10) errors.push('Missing Chinese response');
if (/lang="en"|Draft response|复制英文|英文候选正文|下面英文/.test(html)) errors.push('Untranslated response UI');
if (crypto.createHash('sha256').update(fs.readFileSync(source)).digest('hex') !== sourceHash) errors.push('Source changed');
if (errors.length) throw new Error(errors.join('\n'));
fs.writeFileSync(output, html, 'utf8');
console.log(JSON.stringify({output, cards:cardCount, sourceUnchanged:true, linkCount:links.length, bytes:Buffer.byteLength(html), errors}, null, 2));
