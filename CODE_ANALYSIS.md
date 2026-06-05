# 代码逻辑分析报告

## 项目架构概览

本项目实现两种图文表示学习方法：

1. **CLIP Baseline**: 标准 CLIP 微调 + 双向对比损失
2. **Distribution Alignment**: 高斯分布建模 + 对比损失 + KL 散度

---

## 模块详细分析

### 1. 数据层 (`data/`)

#### `caption_dataset.py` - ImageCaptionDataset

**职责**: 加载 MS-COCO 格式的图像-文本对

**数据流**:
```
Parquet 文件 (url, caption, image_file_name)
    ↓ pd.read_parquet()
DataFrame
    ↓ __getitem__()
{
    "image": PIL.Image (RGB),
    "image_path": str,
    "image_name": str,
    "captions": List[str] (固定 NUM_CAPTIONS=5 个)
}
```

**关键逻辑**:
- `_get_captions()`: 自动填充/截断描述数量到 5 个
- `collate_fn()`: 过滤掉加载失败的 None 样本
- `_validate_images()`: 快速检查前 100 个样本的图片是否存在

**数据对应关系验证**:
- 每个样本的 `image` 和 `captions` 严格一一对应
- Parquet 中的 `image_file_name` 关联到 `IMAGES_DIR` 中的实际图片

---

### 2. 模型层 (`models/`)

#### `clip_baseline.py` - CLIPFineTuneBaseline

**架构**:
```
CLIP ViT-Large-Patch14 (本地加载, local_files_only=True)
├── vision_model → image features [B, projection_dim]
├── text_model → text features [B, projection_dim]
├── visual_projection → 投影到统一维度
└── text_projection → 投影到统一维度
```

**关键方法**:
- `encode_image(images, normalize=True)` → `[B, 512]`
- `encode_text(input_ids, attention_mask, normalize=True)` → `[B, 512]`
- `forward(images, input_ids, attention_mask)` → `(image_features, text_features)`
- `process_images(images: List[PIL.Image])` → `pixel_values` tensor
- `process_text(texts: List[str])` → `{input_ids, attention_mask}`

**冻结机制**:
- `freeze_image=True`: 冻结 `vision_model` + `visual_projection`
- `freeze_text=True`: 冻结 `text_model` + `text_projection`

#### `dist_align_model.py` - DistributionAlignmentModel

**架构**:
```
CLIP ViT-Large-Patch14 (可冻结)
├── Image Branch
│   ├── CLIP vision_model → features [B, 768]
│   └── MLP head → μ_img [B, 768], logvar_img [B, 768]
└── Text Branch
    ├── CLIP text_model → features [B, K, 768]
    ├── MLP head → K 组 (μ, logvar) [B, K, 768]
    └── merge_distributions() → μ_text, logvar_text [B, 768]
```

**分布合并方法**:
1. **moment_matching** (默认): 矩匹配，最小化 KL 散度
   - μ = Σwᵢμᵢ
   - σ² = Σwᵢ(σᵢ² + μᵢ²) - μ²

2. **poe** (Product of Experts): 乘积合并
   - τ = Στᵢ
   - μ = (Στᵢμᵢ) / τ

3. **simple**: 简单平均

**关键方法**:
- `encode_image_distribution(pixel_values)` → `(img_features, img_mu, img_logvar)`
- `encode_text_distribution(input_ids, attention_mask)` → `(text_features, text_mu, text_logvar)`
- `merge_distributions(mus, logvars)` → `(merged_mu, merged_logvar)`
- `forward(pixel_values, input_ids, attention_mask)` → `(img_features, text_features, img_mu, img_logvar, text_mu, text_logvar)`
- `process_images()` / `process_text()`: 同 CLIPBaseline

---

### 3. 损失函数层 (`losses/`)

#### `clip_losses.py`

**函数**:
- `compute_similarity_matrix(image_features, text_features, temperature)` → `[B, B]` 相似度矩阵
- `clip_contrastive_loss(image_features, text_features, temperature)` → `(loss, info_dict)`
  - 双向交叉熵: `loss = (loss_i2t + loss_t2i) / 2`
  - info_dict 包含: loss, loss_i2t, loss_t2i, acc, acc_i2t, acc_t2i
- `clip_contrastive_loss_with_hard_negatives()` → 带硬负采样的对比损失
- `CLIPLoss` (nn.Module): 封装对比损失为 PyTorch Module

#### `dist_align_losses.py`

**类**:
1. **DistributionAlignmentLoss**: 对比损失 + KL 散度
   - 支持 KL 类型: symmetric, forward, reverse, wasserstein
   - KL 对称版本: `KL(P||Q) + KL(Q||P)`
   - Wasserstein: `||μ₁ - μ₂||² + ||σ₁ - σ₂||²`

2. **VarianceRegularizationLoss**: 方差正则化
   - 防止分布退化（方差为 0 或过大）
   - `L_var = ||σ² - target_variance||²`

3. **CombinedDistributionLoss**: 完整损失组合
   - `L_total = λ_contrastive × L_contrastive + λ_kl × L_kl + λ_var × L_var`

---

### 4. 训练/评估脚本 (`scripts/`)

#### `train_clip_baseline.py`

**数据集划分**:
```
Full Dataset (118K samples)
    → random_split(90% train / 10% val, seed=42)
    → train_dataloader (shuffle=True)
    → val_dataloader (shuffle=False)
```

**训练流程** (与 train_dist_align.py 风格一致):
```
每个 epoch:
1. train_epoch() → 在训练集上训练
2. evaluate() → 在验证集上评估
3. 如果 val_loss < best_val_loss → 保存 clip_baseline_best.pt, 重置 patience
4. 否则 patience_counter += 1
5. 如果 patience >= early_stop_patience (3) → 早停
```

**数据流**:
```
DataLoader(ImageCaptionDataset, collate_fn=filter_none_collate)
    ↓ batch = {"image": List[PIL.Image], "captions": List[List[str]], ...}
    ↓
1. 提取 PIL 图像和描述列表
2. 随机选择每个图像的 1 个描述: random.choice(captions)
3. model.process_images(pil_images) → pixel_values
4. model.process_text(selected_captions) → input_ids, attention_mask
5. model(pixel_values, input_ids, attention_mask) → features
6. clip_contrastive_loss(image_features, text_features) → loss
7. backprop + optimizer.step()
```

**检查点管理**:
- `clip_baseline_best.pt`: 验证 loss 最低时保存
- `clip_baseline_last.pt`: 训练结束时保存

**早停机制**:
- 默认 patience=3: 连续 3 个 epoch 验证 loss 无改善则停止
- 可通过 `--early-stop-patience` 调整，`--no-early-stop` 禁用
- 可通过 `--val-split` 调整验证集比例（默认 0.1）

#### `evaluate_clip_baseline.py`

**评估流程**:
```
加载 checkpoint → 遍历 DataLoader
    → 提取所有 image_features 和 text_features (使用第 1 个描述)
    → 计算完整相似度矩阵
    → compute_bidirectional_recall() → Recall@K
```

#### `train_dist_align.py`

**数据集划分**:
```
Full Dataset (118K samples)
    → random_split(90% train / 10% val, seed=42)
    → train_dataloader (shuffle=True)
    → val_dataloader (shuffle=False)
```

**训练流程**:
```
每个 epoch:
1. train_epoch() → 在训练集上训练
2. evaluate() → 在验证集上评估
3. 如果 val_loss < best_val_loss → 保存 dist_align_best.pt, 重置 patience
4. 否则 patience_counter += 1
5. 如果 patience >= early_stop_patience (3) → 早停
```

**数据流**:
```
DataLoader(ImageCaptionDataset, collate_fn=collate_fn)
    ↓ batch = {"image": List[PIL.Image], "captions": List[List[str]], ...}
    ↓
1. 提取 PIL 图像: pil_images = batch["image"]
2. model.process_images(pil_images) → pixel_values [B, 3, 224, 224]
3. 展平所有描述: all_captions = [c for cs in batch["captions"] for c in cs]  # [B*K]
4. model.process_text(all_captions) → input_ids, attention_mask [B*K, 77]
5. 重塑: input_ids.view(B, K, -1) → [B, K, 77]
6. model(pixel_values, input_ids, attention_mask) → features + distributions
7. DistributionAlignmentLoss → loss
8. backprop (CLIP 和 MLP 使用不同学习率)
```

**检查点管理**:
- `dist_align_best.pt`: 验证 loss 最低时保存（基于验证集自动选择）
- `dist_align_last.pt`: 训练结束时保存

**早停机制**:
- 默认 patience=3: 连续 3 个 epoch 验证 loss 无改善则停止
- 可通过 `--early-stop-patience` 调整，`--no-early-stop` 禁用
- 可通过 `--val-split` 调整验证集比例（默认 0.1）

**双学习率**:
- CLIP 参数: `DIST_ALIGN_CLIP_LR` (1e-6)
- MLP 参数: `DIST_ALIGN_MLP_LR` (1e-4)

#### `evaluate_dist_align.py`

**评估流程**:
```
加载 checkpoint → 遍历 DataLoader
    → 提取 img_mu 和 text_mu (使用分布均值作为特征)
    → 计算相似度矩阵
    → compute_bidirectional_recall() → Recall@K
```

---

### 5. 工具层 (`utils/`)

#### `seed.py`
- `set_seed(seed)`: 设置 Python, NumPy, PyTorch, CUDA 的随机种子
- `get_seed()`: 获取当前种子值

#### `io_utils.py`
- `load_json()` / `save_json()`: JSON 文件读写
- `load_parquet()`: Parquet 文件读取
- `save_checkpoint()` / `load_checkpoint()`: 模型检查点管理
- `save_pickle()` / `load_pickle()`: Pickle 序列化

#### `image_utils.py`
- `load_image()` / `load_images()`: 图像加载和验证
- `resize_image()` / `center_crop()`: 图像变换
- `image_to_numpy()` / `numpy_to_image()`: 格式转换
- `validate_image_format()`: 格式校验

#### `metrics.py`
- `compute_recall_at_k()`: 计算 Recall@K 指标
- `compute_bidirectional_recall()`: 双向（i2t + t2i）召回率
- `format_recall_results()`: 格式化输出结果

#### `logger.py`
- `setup_logger()`: 配置日志器（文件 + 控制台）
- `get_logger()`: 获取命名日志器
- `log_exception()`: 异常日志记录（含堆栈跟踪）

---

## 数据对应关系验证

### 数据集层面 (正确)
```
dataset[0] = {
    "image": PIL.Image (000000000009.jpg),
    "captions": ["desc1", "desc2", "desc3", "desc4", "desc5"],
    "image_path": ".../images/000000000009.jpg",
    "image_name": "000000000009.jpg"
}
→ 图像和描述严格对应
```

### CLIP Baseline 训练层面 (正确)
```
batch["image"][i] ↔ batch["captions"][i][random_idx]
→ 通过 process_images/process_text 编码为 tensor
→ 保持对应关系
```

### Distribution Alignment 训练层面 (正确)
```
pixel_values[i] ↔ input_ids[i, 0:K, :]
→ 第 i 张图像 ↔ 第 i 组的 K 个描述
→ 保持对应关系
```

### 评估层面 (正确)
```
image_features[i] ↔ text_features[i]
→ 使用对角线作为正样本对
```

---

## 配置管理验证

所有配置集中在 `config.py`:

| 类别 | 配置项 | 验证状态 |
|------|--------|----------|
| 路径 | PROJECT_ROOT, CAPTIONS_PATH, IMAGES_DIR | 正确 |
| 模型 | CLIP_VIT_L_14_PATH | 正确 |
| 输出 | CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR | 正确 |
| Baseline 超参 | EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, TEMPERATURE | 正确 |
| DistAlign 超参 | EPOCHS, BATCH_SIZE, CLIP_LR, MLP_LR, LAMBDA_* | 正确 |
| 分布配置 | MERGING, KL_TYPE, DROPOUT, TARGET_VARIANCE | 正确 |

---

## 总结

### 整体代码状态

| 模块 | 状态 | 说明 |
|------|------|------|
| data/caption_dataset.py | 正确 | 数据加载、图文对应正确 |
| models/clip_baseline.py | 正确 | CLIP 微调模型，含 processor |
| models/dist_align_model.py | 正确 | 分布对齐模型，含 processor 和分布合并 |
| losses/clip_losses.py | 正确 | 标准对比损失 |
| losses/dist_align_losses.py | 正确 | 分布对齐损失（KL + 对比 + 方差正则） |
| scripts/train_clip_baseline.py | 正确 | 验证集划分 + 早停 + best checkpoint |
| scripts/evaluate_clip_baseline.py | 正确 | 正确的评估流程 |
| scripts/train_dist_align.py | 正确 | 验证集划分 + 早停 + best checkpoint + 双学习率 |
| scripts/evaluate_dist_align.py | 正确 | 使用分布均值评估 |
| utils/* | 正确 | 工具函数完善 |
| config.py | 正确 | 集中配置管理 |

### 数据处理统一性

所有训练/评估脚本统一通过 `model.process_images()` 和 `model.process_text()` 处理数据，而非直接使用 `torch.stack()`。这两个方法内部使用 CLIPProcessor 完成 PIL Image → tensor 和 str → input_ids 的转换。
