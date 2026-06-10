# 代码逻辑分析报告

## 项目架构概览

本项目实现 **UC-CL (Uncertainty-Calibrated Distributional Contrastive Learning)** 及 7 个对比基线（B1-B8），用于图像-文本表示学习。

### 核心方法
- **UC-CL**: 冻结 CLIP ViT-L/14 + 4 MLP 头 → 高斯分布 N(μ, σ²I)
- **关键约束**: σ²_img ≈ Var(μ_captions)，使 σ 具有语义意义

### 基线对比

| ID | 方法 | 模型文件 | 训练脚本 | σ 来源 |
|----|------|---------|---------|--------|
| B1 | CLIP Zero-Shot | `clip_zero_shot.py` | 无需训练 | 无 |
| B2 | CLIP Fine-Tune | `clip_baseline.py` | `train_clip_baseline.py` | 无 |
| B3 | ProLIP | `prolip_model.py` | `train_prolip.py` | 隐式(inclusion loss) |
| B4 | GroVE | `grove_model.py` | `train_grove.py` | GP 后验方差 |
| B5 | ICPE | `icpe_model.py` | 无需训练 | k-NN 协方差 |
| B6 | D2P | `d2p_model.py` | `train_d2p.py` | 仅文本端 |
| Ours | UC-CL | `dist_align_model.py` | `train_dist_align.py` | 显式约束 |

---

## 模块详细分析

### 1. 数据层 (`data/`)

#### `caption_dataset.py` — ImageCaptionDataset (MSCOCO)
- 输入: Parquet (url, caption, image_file_name)
- 输出: `{"image": PIL.Image, "captions": List[str] (K=5)}`
- 关键: `_get_captions()` 自动填充/截断到 5 个; `filter_none_collate` 过滤加载失败的样本

#### `flickr30k_dataset.py` — Flickr30KDataset (Exp6)
- 加载 Flickr30K 图像-文本对，按图像分组 captions
- `get_flickr30k_test_loader()` 便捷函数

#### `vqa_dataset.py` — VQADataset (Stage 2)
- 加载 VQA 问题-答案对，构建 answer vocabulary
- 返回: `{"images", "questions", "answer_indices", "question_types"}`

---

### 2. 模型层 (`models/`)

#### 统一接口

所有模型都实现:
- `encode_image(pixel_values) → Tensor(B, 768)`: 提取图像特征
- `encode_text(input_ids, attention_mask) → Tensor(B, 768)`: 提取文本特征
- `process_images(images) → pixel_values`: PIL → Tensor
- `process_text(texts) → {input_ids, attention_mask}`: str → Token IDs
- `save(path)` / `load(path)`: 模型序列化

#### `dist_align_model.py` — DistributionAlignmentModel (Ours)

```
CLIP ViT-L/14 (frozen)
├── Image: clip_feat → img_mu_head → μ_img
│                   → img_logvar_head → logvar_img
└── Text K captions: clip_feat_k → text_mu_head → μ_k
                                      → text_logvar_head → logvar_k
                   → merge_distributions(moment_matching) → μ_text, logvar_text
```

- `forward(pixel_values, input_ids_3d, attention_mask_3d)` → Dict(img_mu, img_logvar, text_mu, text_logvar, text_mus, ...)
- `encode_image()` / `encode_text()`: 返回确定性 μ

#### `prolip_model.py` — ProLIPModel (B3)
- 同架构（4 MLP 头），但 σ 无显式语义约束
- 使用 `baseline_utils.merge_distributions_moment_matching` 合并

#### `grove_model.py` — GroVEModel (B4)
- GP 后验: 可学习 inducing points + attention 加权
- `_compute_gp_posterior()`: 计算后验 μ 和距离依赖的 logvar

#### `icpe_model.py` — ICPEModel (B5)
- Training-free: 冻结 CLIP + k-NN 协方差
- `compute_icpe_covariance()`: 在完整数据集上计算每样本方差
- `trainable_parameters()` 返回空列表

#### `d2p_model.py` — D2PModel (B6)
- 图像: 确定性点嵌入 (img_projection)
- 文本: 分布嵌入 (text_mu_head + text_logvar_head)
- `d2p_loss()`: MC 采样的 distribution-to-point 对比损失

#### `vqa_model.py` — VQAModel (Stage 2 统一封装)
- 冻结 backbone + 可训练分类头: `Linear(1536→512) → ReLU → Dropout → Linear(512→num_classes)`
- 对 dist_align: 训练时采样 z=μ+εσ，评估时用 μ
- 对其他基线: 委托 `base_model.encode_image()` / `encode_text()`
- clip_zero_shot 由 `train_vqa.py` 独立处理，不经过 VQAModel

#### `baseline_utils.py` — 共享工具
- `merge_distributions_moment_matching(mus, logvars)`: 矩匹配合并
- `encode_clip_features(clip_model, ...)`: 统一 CLIP 编码
- `init_heads_xavier(heads)`: Xavier 初始化

---

### 3. 损失函数层 (`losses/`)

#### `clip_losses.py`
- `clip_contrastive_loss(img_feat, text_feat, temperature)`: 双向交叉熵

#### `dist_align_losses.py` — UC-CL 损失
- **UncertaintyCalibratedContrastiveLoss**: L = λ_cl·L_calibrated_CL + λ_consist·L_consistency + λ_var·L_variance
  - L_calibrated_CL: `sim = μ_x·μ_y / (τ·√(1+var_x)·√(1+var_y))`
  - L_consistency: `||σ²_img - Var(μ_captions)||²`
  - L_variance: `||σ² - target||²`
- **DistributionAlignmentLoss**: 标准 KL + 对比损失（旧版，ProLIP 使用）

---

### 4. 训练/评估脚本 (`scripts/`)

#### 通用训练流程
```
Dataset → DataLoader (shuffle=True, collate_fn=filter_none_collate)
    → train/val split (90%/10%, seed=42)
    → 每个 epoch: train → validate → save best if improved
    → 早停 (patience=3)
```

#### 数据处理流 (分布模型)
```
batch = {"image": List[PIL], "captions": List[List[str]]}
→ pixel_values = model.process_images(pil_images)     # [B, 3, 224, 224]
→ all_captions = [c for cs in caption_lists for c in cs]  # [B*K]
→ text_inputs = model.process_text(all_captions)        # {input_ids: [B*K, 77]}
→ input_ids.view(B, K, -1)                              # [B, K, 77]
→ model(pixel_values, input_ids, attention_mask)        # Dict with distributions
→ loss → backward → optimizer.step()
```

#### 评估流程
```
加载 checkpoint → 遍历 DataLoader
    → 提取 img_mu, text_mu (分布模型) 或 features (点模型)
    → compute_recall_chunked() → R@1, R@5, R@10
    → (可选) compute_recall_uc_chunked() → UC-Recall@K
```

---

### 5. 实验脚本

| 实验 | 脚本 | 指标 |
|------|------|------|
| Exp3 Calibration | `eval_calibration.py` | ECE, NLL, Brier, AUROC |
| Exp4 OOD | `eval_ood.py` | AUROC, FPR@95TPR (sigma-based) |
| Exp5 Ablation | `run_ablation.py` | R@K (6 配置 + λ/τ 敏感性分析) |
| Exp6 Flickr30K | `eval_flickr30k.py` | R@K (跨数据集泛化) |
| Exp7 σ Analysis | `eval_sigma_analysis.py` | Pearson/Spearman 相关系数 |
| Exp8 Modality Gap | `visualize_modality_gap.py` | t-SNE, gap distance, cosine sim 分布 |

---

### 6. 工具层 (`utils/`)

| 模块 | 功能 |
|------|------|
| `seed.py` | 设置 Python/NumPy/PyTorch/CUDA 随机种子 |
| `io_utils.py` | JSON/Parquet/Checkpoint/Pickle 读写 |
| `image_utils.py` | 图像加载、变换、格式转换 |
| `metrics.py` | Recall@K 计算和格式化 |
| `logger.py` | 文件+控制台双输出日志系统 |
| `calibration.py` | ECE/MCE/NLL/Brier/AUROC/FPR@95TPR |

---

## 数据对应关系验证

### 图文对应（正确）
```
pixel_values[i] ↔ input_ids[i, :, :]   # 第 i 张图 ↔ 第 i 组描述
```

### VQA 对应（正确）
```
images[i] ↔ questions[i] ↔ answer_indices[i]   # 图-问题-答案严格对应
```

### 跨数据集（Exp6）
```
Flickr30K 图像与 5 个 caption 按图像名分组，保持对应关系
```

## 配置管理

所有配置集中在 `config.py`，按功能分区:
- 路径配置 (数据集、模型、输出)
- Ours 超参 (UC-CL 损失权重、温度、分布配置)
- B2-B6 基线超参
- VQA 配置
- Exp3-8 实验配置
- Flickr30K 路径
