# 方法文档 · Methods

> 本文件描述本项目当前采用的图文分布对齐方法 **MSDA（Multi-caption Semantic Distribution Alignment）**，
> 以及在引入协方差项 `L_cov` 后出现的训练稳定性问题与其修复方案（§6）。
> 项目总览与使用方式见 `README.md`；实验结果与复现命令见 `experiments.md`。

---

## 1. 概述

MSDA 是 UC-CL 的演进版本，核心思想不变：**把图像与文本建模为高斯分布，并显式约束图像方差 σ² 等于多描述的语义散度**。相对 UC-CL，MSDA 做了三点升级：

1. **一般高斯**：将 UC-CL 的对角高斯 `N(μ, diag(σ²))` 升级为一般高斯 `N(μ, Σ)`，其中 `Σ = diag(σ²) + UUᵀ`，`U ∈ ℝ^{D×r}` 为低秩协方差因子（`r` 为协方差秩，`r=0` 退化为对角版）。
2. **协方差方向对齐损失 `L_cov`**：监督图像协方差子空间去对齐多描述的偏离子空间。
3. **3 段式分阶段训练调度**：Warmup → Main → Full，逐步激活损失项。

---

## 2. 记号

| 符号 | 含义 |
|---|---|
| `x` / `{c^(k)}_{k=1..K}` | 一张图像 / 其 K 条描述（MSCOCO 默认 `K=5`） |
| `h ∈ ℝ^768` | 冻结 CLIP ViT-L/14 输出特征 |
| `μ_x, logσ²_x ∈ ℝ^768` | 图像分布均值 / 对数方差 |
| `U_x ∈ ℝ^{768×r}` | 图像低秩协方差因子（`r=0` 时为 `None`） |
| `μ_c^(k), logσ²_c^(k), U_c^(k)` | 第 k 条描述的分布参数 |
| `(μ_c, logσ²_c)` | K 条描述经 moment matching 融合后的集合分布 |

---

## 3. 模型结构

冻结 CLIP 之上接 **6 个互相独立的 MLP/Linear 头**（无参数共享）：

```
img_mu_head, img_logvar_head, img_cov_head   （图像端：μ, logσ², U）
text_mu_head, text_logvar_head, text_cov_head（文本端：μ, logσ², U，逐 caption 编码后融合）
```

- 每个头输入均为冻结的 CLIP 特征，CLIP 不参与训练（`Freeze CLIP = True`）。
- 可训练参数：`r=4` 时约 **9.45M**；`r=0`（对角版）约 **4.72M**（与 UC-CL 持平）。
- **方差数值下限**：`σ² = softplus(x) + VAR_FLOOR(1e-4)`。相比 UC-CL 旧的硬下限 `0.1`，此处改为软下限，使 σ² 能向下拟合真实描述散度（语义范围由 `L_var` / `L_reg` 学习决定，而非人为钉死）。
- **文本分布融合（Moment Matching）**：
  - `μ_c = (1/K) Σ_k μ_c^(k)`
  - `diag(Σ_c) = (1/K) Σ_k [ σ²_c^(k) + diag(U_c^(k) U_c^(k)ᵀ) + μ_c^(k)² ] − μ_c²`

---

## 4. 损失函数

```
L = λ_ctr · L_set-NCE + λ_mu · L_mu + λ_var · L_var
  + λ_cover · L_cover + λ_cov · L_cov + λ_reg · L_reg
```

| 项 | 公式 | 作用 |
|---|---|---|
| `L_set-NCE` | 不确定性折扣双向 InfoNCE：`sim = μ̂_x·μ̂_c / (τ·√(1+meanσ²_x)·√(1+meanσ²_c))` | 集合级检索对齐（主驱动） |
| `L_mu` | `1 − cos(μ_x, μ_c)` | 均值中心对齐 |
| `L_var`（核心） | `MSE(σ²_x, sg[Var_k(μ_c^(k))])`，对描述散度 stop-gradient | **σ² 拟合多描述语义散度** |
| `L_cover` | `mean(relu(dM/D − m_pos))`，`dM` 为 Mahalanobis 距离 | 多描述覆盖约束 |
| `L_cov` | `2r − 2‖Q_vᵀ Q_t‖²`（子空间 Frobenius 距离） | **协方差方向对齐** |
| `L_reg` | `MSE(logσ², log σ₀²)`（图像+文本对称） | 防止方差坍塌/爆炸 |

其中 `Q_v = orth(U_x)`（QR 得到），`Q_t` 由描述偏差矩阵 `dev = μ_c^(k) − μ_c` 经小 Gram 矩阵特征分解得到（**对 `dev` 做 stop-gradient**，理由见 §6）。

**默认权重**：`λ_ctr/mu/var/cover/cov/reg = 1.0 / 0.5 / 1.0 / 0.5 / 0.01 / 0.01`
（`λ_cov` 在 §6 修复中由 `0.1` 下调至 `0.01`）。其余超参：`τ=0.07, m_pos=1.0, σ₀²=0.5, r=4`。

---

## 5. 3 段式分阶段训练调度

按总轮数 `T` 的比例划分（`MSDA_STAGE_WARMUP_FRAC/MAIN_FRAC/FULL_FRAC = 0.2/0.6/0.2`）：

| 阶段 | 区间（占总轮数） | 激活的损失项 | cov 系数 |
|---|---|---|---|
| Warmup | 前 20% | `L_set-NCE + L_mu` | 0 |
| Main | 中间 60% | `+ L_var + L_cover` | 0 |
| Full | 末 20% | `+ L_cov` | **线性 ramp 0→1**（见 §6.3） |

> 说明：旧实现中 Full 阶段 `cov` 系数是 **0→1 硬切**；本次修复改为 **线性 ramp**。

---

## 6. 稳定性设计与修复（重要）

### 6.1 问题现象

首次 MSDA 实验（10 轮）中，`L_cov` 在 Full 阶段（第 9 轮）一激活，训练 loss 立刻从 0.16 暴涨到 0.52，验证 Recall@1 从 **0.465 崩到 0.265**，第 10 轮继续恶化到 0.240。详见 `experiments.md` §4。

### 6.2 根因分析

1. **`λ_cov=0.1` 对该 loss 量级过大**。`L_cov = 2r − 2‖Q_vᵀ Q_t‖²`，最大值 `2r = 8`，实测约 3。`0.1 × 3 ≈ 0.30`，占当轮 total loss（0.52）的 **58%**，瞬间主导整张计算图的梯度。
2. **`img_cov_head` 在 cov 激活前几乎未训练**。它在此之前只收到极弱的 `L_cover` 梯度，故 `U_x ≈ 小随机初始化（std=1e-2）`，其列空间 `Q_v` 基本是随机 r 维子空间，与描述偏离子空间 `Q_t` 近似正交 → `L_cov` 一激活就近上限，梯度极陡。
3. **无梯度裁剪** → 上述巨大 cov 梯度直接转化为参数大更新。
4. **检索均值被污染的回流路径**：`img_cov_head` 与 `img_mu_head` 虽参数独立，但 `L_cover` 同时依赖 `U_x` 与 `μ_x`（Mahalanobis），不稳定的 `U_x` 经 cover 路径把噪声梯度灌入 `img_mu_head`；叠加 `qr/eigh/solve` 反向在近退化点可能产生 Inf/NaN。

> 代码注释中作者已知 "`L_cov` 会 crash Recall@1 the moment it activates"，并用 `detach()` 挡住了文本端；但 image 端经 `L_cover` 的回流、以及权重/裁剪问题此前未解决。

### 6.3 P0 修复（已实现并验证）

| # | 位置 | 改动 |
|---|---|---|
| 1 | `scripts/train_dist_align.py` `train_epoch` | 反向后、step 前加 `torch.nn.utils.clip_grad_norm_(model.parameters(), config.MSDA_GRAD_CLIP_NORM)` |
| 1b | `config.py` | 新增 `MSDA_GRAD_CLIP_NORM = 1.0` |
| 2 | `config.py` | `MSDA_LAMBDA_COV`: **0.1 → 0.01** |
| 2b | `config.py` 消融块 | 7 处 `"lambda_cov": 0.1` 同步改为 `0.01`（`no_cov`/`diagonal_only`/`k1` 的 `0.0` 不动） |
| 3 | `scripts/train_dist_align.py` `stage_multipliers` | Full 阶段 `cov` 系数改为 **线性 ramp** `0→1`，取代硬切 |
| 4 | `losses/dist_align_losses.py` `MSDALoss.forward` | `L_cov` 非有限值时置零（`torch.zeros_like(cov_loss)`） |

**设计要点**：
- 梯度裁剪 + 降权 + ramp 三者共同保证：`L_cov` 即便在冷启动近上限时，其梯度贡献也不再主导、不再瞬时打飞 `img_cov_head`。
- NaN 保护用 `torch.zeros_like` 而非 `nan*0`（IEEE-754 下 `nan*0=nan`，无法归零——这是实现中验证发现并修正的坑）。
- 这些仅保证 **不再崩**；要追平/超过 UC-CL 的 0.5622，还需配合训练轮数等（见 `experiments.md` §7）。

---

## 7. 推理

- **图文检索**：用均值 `μ`（确定性）计算相似度；可选 UC 相似度（σ 参与排序）。
- **VQA**：`μ` 确定性推理；或 Monte-Carlo 采样 `z ~ N(μ, σ²)` 多次平均以估计不确定性。

---

## 8. 关键代码位置

| 模块 | 文件 |
|---|---|
| 模型（分布头、融合、cov head） | `models/dist_align_model.py` |
| 损失（含 `L_cov` 与 NaN 保护） | `losses/dist_align_losses.py` |
| 训练循环（梯度裁剪、分阶段调度） | `scripts/train_dist_align.py` |
| 超参与调度比例 | `config.py`（`MSDA_*`） |
