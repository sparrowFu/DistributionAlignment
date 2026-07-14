# ProLIP Baseline 重构 — 会话交接文档

> 写给一个**完全没有上下文**的新会话。读完这一份就能接着干。
> 最后更新:2026-07-14(微调训练进行中,epoch 2/5)

---

## 0. 一句话总结

把 ProLIP baseline(B3)从「假 ProLIP」(CLIP-ViT-L/14 + 从零训的 MLP 头)改成**真 ProLIP ViT-H/14**(通过 `prolip` 库加载),支持 `zero_shot`(冻结预训练)和 `fine_tuning`(全参微调 + ProLIP inclusion loss)两种模式,下游任务是**图文检索**(I2T + T2I),脚本风格对齐 CLIP 三件套。**代码已完成、已验证、已提交。微调正在跑。**

---

## 1. 任务背景

- 项目:`DistributionAlignment`(方法名 MSDA / dist_align)。ProLIP 是 baseline B3。
- 旧 `models/prolip_model.py` 名字叫 ProLIP,但其实是 CLIP ViT-L/14 + 4 个 MLP 头从零训,**从没加载过真 ProLIP 权重**。
- 真模型权重 `PreTrainedModels/prolip` 是 `SanghyukChun/ProLIP-ViT-H-14-FT-DC-1B-1_28M`(OpenCLIP 架构 + ProLIP 的 uncertainty 头,**不能**用 HuggingFace `CLIPModel.from_pretrained` 加载)。
- 用户已把 `prolip` 库及其依赖装进 conda env `CudaVersion128Fuxp`;三个本地 artifact 已下载:`PreTrainedModels/prolip`(权重)、`prolipProcessor`(图像)、`prolipTokenizer`(文本)。
- 用户要求:zero_shot + fine_tuning 两种模式,风格对齐 `evaluate_clip_zero_shot.py` / `evaluate_clip_baseline.py` / `train_clip_baseline.py`;**VQA 任务已废弃,不要动 vqa 相关代码**。

---

## 2. 已完成(均已提交到 commit `425c0ee`,工作区干净)

| 文件 | 操作 | 说明 |
|---|---|---|
| `models/prolip_model.py` | **重写** | `ProLIPModel` 包裹 `prolip.model.ProLIPHF`;`process_images/process_text/forward/encode_images/encode_texts/save/load/freeze/trainable_parameters`;2D/3D input_ids 兼容;transformers≥5 兼容补丁 |
| `scripts/evaluate_prolip_zero_shot.py` | **新建** | 冻结预训练,无 checkpoint,cosine+CSD 的 I2T+T2I |
| `scripts/evaluate_prolip.py` | **重写** | 加载 `prolip_best.pt` 后评估 |
| `scripts/train_prolip.py` | **重写** | 全参微调 + `prolip.loss.ProLIPLoss`(PPCL+inclusion+VIB),早停/cosine LR+warmup/resume/best+last |
| `utils/retrieval_metrics.py` | **新建** | 分块计算 I2T/T2I × cosine/CSD 的 Recall@K |
| `config.py` | 改 | ProLIP 三路径、`PROLIP_EMBED_DIM=1024`、微调超参 + inclusion loss 权重、zero-shot 结果/日志路径 |
| `main.py` | 改 | 新增 task `eval_prolip_zero_shot` |
| `scripts/eval_flickr30k.py` | 改 | ProLIP 分支 `ProLIPModel(freeze=True)`(wrapper 向后兼容,抽取逻辑没动) |
| `scripts/visualize_modality_gap.py` | 改 | 同上 + 修了过时 docstring |

最新 commit:`425c0ee 修改prolip模型的训练和评估代码`(代码)+ `7da9c29 添加prolip和grove依赖`(env yml)。当前分支 `test`。

---

## 3. 当前状态(微调进行中)

- **训练正在外部用户终端跑**(不是 Claude 启的),日志:`logs/train_prolip.log`。
- 配置:5 epochs,B=16,LR=1e-6,106459 训练 / 11828 验证,**每 epoch ≈ 2.5h**(ViT-H/14 全参微调很慢)。
- 进度:epoch 1 已完成(train loss 0.479 / acc 0.979,val loss 0.325 / acc 0.9794),**当前 epoch 2**。
- 已有产物:
  - `checkpoints/prolip_best.pt`(3.95 GB,epoch 1 best)✅
  - `outputs/prolip_zero_shot_eval_results.json`(5000 样本 zero-shot 结果)✅
- **尚未生成**:`checkpoints/prolip_last.pt`(训练结束才写)、`outputs/prolip_eval_results.json`(微调后评估)。

### 已有的 zero-shot 结果(5000 样本,真 ProLIP 冻结)
| 方向 | 度量 | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| i2t | cosine | 0.3670 | 0.6104 | 0.7108 |
| i2t | csd | 0.3828 | 0.6318 | 0.7292 |
| t2i | cosine | 0.3512 | 0.5784 | 0.6738 |
| t2i | csd | 0.3420 | 0.5718 | 0.6672 |

读数:CSD 在 i2t 上比 cosine 好约 +0.016(σ 在文本侧有用),在 t2i 上略差(图像 σ² 更多反映场景复杂度,惩罚它帮倒忙)。这是真 ProLIP 的 σ²;对比本项目 dist_align 的 σ²(见 memory `sigma-retrieval-finding`)对检索无增益——**σ 有没有用取决于它怎么训出来的**。

---

## 4. 下一步计划

1. **等微调跑完**(还剩 epoch 2–5,约 7–8 小时)。确认方式:`tail -5 logs/train_prolip.log`,看到 `Training completed!` 即结束。
2. **跑微调后评估**:`python main.py --task eval_prolip`(默认读 `checkpoints/prolip_best.pt`,评估 5000 样本,输出到 `outputs/prolip_eval_results.json`)。
3. **对比 zero-shot vs fine-tuned** 的 I2T/T2I × cosine/CSD,看微调是否提升。
4. (可选)如果数值不理想,可调 `--inclusion-alpha` / `--ppcl-lambda` / `--vib-beta` / LR / epochs 重训。
5. 代码已提交,无需再 commit(除非有新改动)。

---

## 5. 如何运行

```bash
# 关键:conda env python 路径(见坑#3)
PY=/home/xpfu/.conda/envs/CudaVersion128Fuxp/bin/python

$PY main.py --task eval_prolip_zero_shot   # zero-shot 检索(冻结,无 checkpoint)
$PY main.py --task train_prolip            # 微调(慢,~2.5h/epoch)
$PY main.py --task eval_prolip             # 微调后检索(需先有 prolip_best.pt)

# 也可直接带参数,例如小样本快速验证:
$PY scripts/evaluate_prolip_zero_shot.py --num-samples 64 --batch-size 16
```

---

## 6. ⚠️ 绝对不要再踩的坑(重点)

1. **transformers 5.x 删了 `PreTrainedTokenizer.batch_encode_plus`**,而 `prolip.tokenizer.HFTokenizer.__call__` 依赖它。`models/prolip_model.py` 的 `__init__` 里有一段把 `batch_encode_plus` 别名到 `__call__` 的兼容补丁——**千万别删**,删了 `process_text` 立刻崩。

2. **`prolip` 库自带(vendor)了它自己的 open_clip**,所以外部 `open_clip` 包**没装也不需要装**。别去 `pip install open_clip`,也别以为 `import open_clip` 失败是个问题。

3. **conda env python 路径是 `/home/xpfu/.conda/envs/CudaVersion128Fuxp/bin/python`**,不是 `/home/xpfu/miniconda3/...`(那个路径不存在)。激活需要先 `conda init`;直接用上面的绝对路径 python 最稳。env 名 `CudaVersion128Fuxp`。

4. **三个 artifact 必须从本地路径加载**,且模块顶部设了 `HF_HUB_OFFLINE=1`。别把 `PreTrainedModels/prolip` 换成 HF hub id(如 `"SanghyukChun/..."`),这服务器 GitHub/HF 访问不稳。

5. **CUDA 上 forward 即使 eval 也有 ~2e-3 的不确定性**(CPU 上完全精确)。这是 fp32 matmul/attention 原子累加导致,**不是 bug**,对 R@K 排序无影响。别花时间追这个。

6. **`save()` 只存 `prolip.state_dict()`(权重)**;processor/tokenizer 每次从 config 里的固定路径重新加载,**不进 checkpoint**。`prolip_last.pt` 额外含 optimizer/epoch/best_val_loss/base_lrs 用于 resume;`prolip_best.pt` 只有权重,被 `evaluate_prolip` 的 `model.load()` 读。

7. **`forward` 忽略 `attention_mask`**(ProLIP 文本塔用 pad_id=0 内部处理 padding)。`process_text` 返回 `{input_ids, attention_mask}` 纯粹是为了对齐 CLIP 脚本风格,mask 是"装饰"。**别"修复"成真去 mask。**

8. **`forward` 同时吃 2D `(B,L)` 和 3D `(B,K,L)` 的 input_ids**;3D 会触发 K 个 caption 的 moment-matching 合并。`eval_flickr30k.py` / `visualize_modality_gap.py` 依赖 3D 路径。**改 forward 时别破坏这个向后兼容**,否则这俩消费者会崩。

9. **flat 别名 `img_mu/text_mu` 是未归一化的**;`image_features/text_features` 字典里的 `mean` 是**已 L2 归一化**的(喂给 ProLIPLoss 用)。两套别混。

10. **ProLIP 的 `std` / `logvar` = log(σ²)**,不是 log(σ) 也不是 σ。`exp(std) = σ²`。CSD 里 `exp(logvar).sum(-1)` 才是 σ² 之和。

11. **训练日志里的 `acc` 是批内(B=16)对比 top-1,不是大 gallery 的 R@K**。acc≈0.98 是因为 16 选 1 简单(随机基线 6.25%),**绝不代表检索强**。真正信号是 loss 下降 + 训完后的 eval R@K。

12. **VQA 任务已废弃**。`models/vqa_model.py` 里 ProLIP 分支是**死代码**(还在用旧接口),本次**故意没改**。除非 VQA 复活,别动它;若要复活,注意 embed_dim 从 768 变成 1024,VQA 头要跟着改。

13. **embed_dim = 1024(ViT-H/14)**。旧 wrapper 是 768(CLLIP ViT-L/14)。任何依赖 ProLIP 特征维度的下游都要按 1024 来。

14. **微调用的是 ProLIP inclusion loss(`prolip.loss.ProLIPLoss`),不是 CLIP 对比 loss**——这是用户明确选的。`ProLIPLoss` = PPCL(常开)+ inclusion_alpha + vib_beta。**别自作主张换成 `clip_contrastive_loss`。**

15. **zero-shot 评估必须同时报 cosine 和 CSD、I2T 和 T2I**(用户硬需求)。CLIP 脚本只做 I2T cosine,但 ProLIP 这边**四个组合都要**。别为了"对齐 CLIP"砍掉 CSD/T2I。

16. **训练很慢**(~2.5h/epoch)。调参/重训前先想清楚;`--num-samples` / 减 epoch 只能用于冒烟测试,正式检索评估要用全量(默认 5000)。

17. **代码已提交**(`425c0ee`),工作区干净。本任务开始前 `config.py` 和 `build_vqa_expansions.py` 各自有一些**与本任务无关的预存改动**,已被用户一起提交了,不用管。

---

## 7. 关键接口速查(`models/prolip_model.py`)

```python
m = ProLIPModel(freeze=False)          # 微调;freeze=True 为 zero-shot(0 可训参数)
m = m.to("cuda")

pv   = m.process_images(list_of_PIL)   # -> (B,3,224,224) 已在 model device 上
ti   = m.process_text(list_of_str)     # -> {"input_ids":(B,77), "attention_mask":(B,77)}
out  = m(pv, ti["input_ids"])          # input_ids 可 2D 或 3D (B,K,77)

# out 关键键:
#   out["image_features"]["mean"]  (B,1024) 已归一化 —— 喂 ProLIPLoss
#   out["text_features"]["mean"]   (B,1024) 已归一化 —— 喂 ProLIPLoss
#   out["logit_scale"], out["logit_bias"]               —— 喂 ProLIPLoss
#   out["img_mu"]/["text_mu"]      (B,1024) 未归一化     —— 检索/消费者用
#   out["img_logvar"]/["text_logvar"] (B,1024) = log σ²  —— CSD 用

m.save(path); m.load(path)
m.trainable_parameters(); m.num_trainable_parameters()
```

**微调 loss 用法**(`scripts/train_prolip.py` 的 `_forward_loss`):
```python
from prolip.loss import ProLIPLoss
crit = ProLIPLoss(ppcl_lambda=1.0, inclusion_alpha=1.0, vib_beta=1e-5)
loss = crit(out["image_features"], out["text_features"],
            logit_scale=out["logit_scale"], logit_bias=out["logit_bias"])
```

**检索指标**(`utils/retrieval_metrics.py`):
```python
compute_retrieval_metrics(img_mu, img_logvar, text_mu, text_logvar, [1,5,10])
# -> {"i2t":{"cosine":{...},"csd":{...}}, "t2i":{"cosine":{...},"csd":{...}}}
```

---

## 8. 相关记忆(已存)

- `prolip-real-baseline.md` — 本任务的持久记录
- `conda-env.md` — env 用法
- `sigma-retrieval-finding.md` — dist_align 的 σ 对检索无增益(对照)
- `msda-methodology.md` — 项目方法
