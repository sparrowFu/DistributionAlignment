# 方法文档 · Methods

> 本文件描述本项目当前采用的图文分布对齐方法 **MCDisp_Align（Multi-Caption Semantic Dispersion Guided Distribution Alignment）**，
> 以及在引入协方差项 `L_cov` 后出现的训练稳定性问题与其修复方案（§6）。
> 项目总览与使用方式见 `README.md`；实验结果与复现命令见 `experiments.md`。

---

## 1. 概述

MCDisp_Align 的核心思想：**把图像与文本建模为高斯分布，并显式约束图像方差 σ² 等于多描述的语义散度**。它支持两种模式——**对角模式**（`r=0`，`Σ = diag(σ²)`）与**全协方差模式**（`r>0`，默认）。全协方差模式相对对角模式做了三点升级：

1. **一般高斯**：将对角高斯 `N(μ, diag(σ²))` 升级为一般高斯 `N(μ, Σ)`，其中 `Σ = diag(σ²) + UUᵀ`，`U ∈ ℝ^{D×r}` 为低秩协方差因子（`r` 为协方差秩，`r=0` 退化为对角模式）。
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
- 可训练参数：`r=4` 时约 **9.45M**；`r=0`（对角模式）约 **4.72M**。
- **方差数值下限**：`σ² = softplus(x) + VAR_FLOOR(1e-4)`。相比对角模式旧的硬下限 `0.1`，此处改为软下限，使 σ² 能向下拟合真实描述散度（语义范围由 `L_var` / `L_reg` 学习决定，而非人为钉死）。
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
| `L_var`（核心） | `MSE(log σ²_x, log sg[Var_k(μ_c^(k))])`（log 空间匹配，最小值仍 σ²=s²；见 §6.4） | **σ² 拟合多描述语义散度** |
| `L_cover` | 正项 `mean(relu(dM/D − m_pos))` + 可选负排斥 `λ_neg·mean(relu(m_neg − dM/D))`（§5.4 "可以再加"；`λ_neg=0` 关闭，= 方法论 canonical 正项 only） | 多描述覆盖约束 |
| `L_cov` | `2r − 2‖Q_vᵀ Q_t‖²`（子空间 Frobenius 距离） | **协方差方向对齐** |
| `L_reg` | `MSE(logσ², log σ₀²)`（图像+文本对称） | 防止方差坍塌/爆炸 |

其中 `Q_v = orth(U_x)`（QR 得到），`Q_t` 由描述偏差矩阵 `dev = μ_c^(k) − μ_c` 经小 Gram 矩阵特征分解得到（**对 `dev` 做 stop-gradient**，理由见 §6）。

**默认权重**：`λ_ctr/mu/var/cover/cov/reg = 1.0 / 0.5 / 1.0 / 0.5 / 0.2 / 0.01`
（`λ_cov=0.2` 取 HTML 方法论建议值；`L_cov` 量级较大，稳定性依赖 grad-clip + 线性 ramp + NaN-guard，详见 §6）。其余超参：`τ=0.07, m_pos=1.0, σ₀²=0.04（≈实测 MSCOCO caption 散度，见 §6.4）, r=4`。

---

## 5. 3 段式分阶段训练调度

按总轮数 `T` 的比例划分（`MCDISP_ALIGN_STAGE_WARMUP_FRAC/MAIN_FRAC/FULL_FRAC = 0.2/0.6/0.2`）：

| 阶段 | 区间（占总轮数） | 激活的损失项 | cov 系数 |
|---|---|---|---|
| Warmup | 前 20% | `L_set-NCE + L_mu` | 0 |
| Main | 中间 60% | `+ L_var + L_cover` | 0 |
| Full | 末 20% | `+ L_cov` | **线性 ramp 0→1**（见 §6.3） |

> 说明：旧实现中 Full 阶段 `cov` 系数是 **0→1 硬切**；本次修复改为 **线性 ramp**。

---

## 6. 稳定性设计与修复（重要）

### 6.1 问题现象

首次 MCDisp_Align 实验（10 轮）中，`L_cov` 在 Full 阶段（第 9 轮）一激活，训练 loss 立刻从 0.16 暴涨到 0.52，验证 Recall@1 从 **0.465 崩到 0.265**，第 10 轮继续恶化到 0.240。详见 `experiments.md` §4。

### 6.2 根因分析

1. **`λ_cov=0.1` 对该 loss 量级过大**。`L_cov = 2r − 2‖Q_vᵀ Q_t‖²`，最大值 `2r = 8`，实测约 3。`0.1 × 3 ≈ 0.30`，占当轮 total loss（0.52）的 **58%**，瞬间主导整张计算图的梯度。
2. **`img_cov_head` 在 cov 激活前几乎未训练**。它在此之前只收到极弱的 `L_cover` 梯度，故 `U_x ≈ 小随机初始化（std=1e-2）`，其列空间 `Q_v` 基本是随机 r 维子空间，与描述偏离子空间 `Q_t` 近似正交 → `L_cov` 一激活就近上限，梯度极陡。
3. **无梯度裁剪** → 上述巨大 cov 梯度直接转化为参数大更新。
4. **检索均值被污染的回流路径**：`img_cov_head` 与 `img_mu_head` 虽参数独立，但 `L_cover` 同时依赖 `U_x` 与 `μ_x`（Mahalanobis），不稳定的 `U_x` 经 cover 路径把噪声梯度灌入 `img_mu_head`；叠加 `qr/eigh/solve` 反向在近退化点可能产生 Inf/NaN。

> 代码注释中作者已知 "`L_cov` 会 crash Recall@1 the moment it activates"，并用 `detach()` 挡住了文本端；但 image 端经 `L_cover` 的回流、以及权重/裁剪问题此前未解决。

### 6.3 稳定性保障（当前正式方法）

`L_cov` 量级大（max `2r=8`、实测~3），在 `λ_cov=0.1/0.2` 时易占 total loss 过半并主导梯度（§6.1–6.2 的崩溃即发生在 `λ_cov=0.1`）。当前正式方法按 HTML 方法论取 **`λ_cov=0.2`**，并通过下列三重保障吸收其量级，不再下调权重：

| # | 位置 | 改动 |
|---|---|---|
| 1 | `scripts/train_mcdisp_align.py` `train_epoch` | 反向后、step 前加 `torch.nn.utils.clip_grad_norm_(model.parameters(), config.MCDISP_ALIGN_GRAD_CLIP_NORM)` |
| 1b | `config.py` | `MCDISP_ALIGN_GRAD_CLIP_NORM = 1.0` |
| 2 | `scripts/train_mcdisp_align.py` `stage_multipliers` | Full 阶段 `cov` 系数改为 **线性 ramp** `0→1`（让 `img_cov_head` 先热身，`Q_v` 不再是纯随机 → `L_cov` 不再一激活就近上限）|
| 3 | `losses/mcdisp_align_losses.py` `MCDispAlignLoss.forward` | `L_cov` 非有限值时置零（用 `torch.zeros_like`；IEEE-754 下 `nan*0=nan`，必须 `zeros_like` 才能真正归零）|

**设计要点**：
- 梯度裁剪 + 线性 ramp + NaN-guard 三者共同保证：`L_cov` 即便在冷启动近上限时，其梯度贡献也不再瞬时打飞 `img_cov_head` 与（经 `L_cover` 回流的）检索均值。
- 历史上的 P0 修复曾把 `λ_cov` 从 `0.1` 下调到 `0.01`；现已按 HTML 方法论回退为 `0.2`，改由上述三重保障维持稳定（`L_cov` 量级与描述偏差 target 的具体形式无关——正交基重叠公式对输入尺度不变）。
- **重训监控建议**：若 Full 阶段 `L_cov` 占 total loss 比例持续过高或 R@1 再次下滑，可回退 `λ_cov` 至 `0.1`（HTML 下限）乃至 `0.01`（已验证稳定）。
- 这些保障仅保证 **不再崩**；要追平/超过对角模式（`r=0`）的 0.5622，还需配合训练轮数等（见 `experiments.md` §7）。

### 6.4 σ² 坍塌修复（σ₀² 尺度 + L_var log 空间）

`scripts/diagnostics/eval_sigma_diagnostic.py` 实测（MSCOCO，`mcdisp_align_coco_best.pt`）：caption 散度 `s²≈0.038`（CV 41%），但模型 σ² 坍塌成常数（mean 0.13，CV 7.2%，Pearson(σ², s²)=0.04）——**核心创新（σ²=多描述散度）未生效**。根因：

1. **σ₀² 尺度失配**：旧 `MCDISP_ALIGN_TARGET_VAR=1.0` 比 s²≈0.038 大 ~26×，`L_reg`（log 空间）把 σ² 强行拉向常数 1.0，压垮 `L_var`。
2. **L_var 线性 vs L_reg log 空间梯度失衡**：s² 很小时线性 MSE 梯度极小（~0.18/dim），弱于 `L_reg` 的 log 空间梯度（~0.31/dim），方差头取"输出常数"的捷径。

修复（P0+P1）：

- **P0**：`MCDISP_ALIGN_TARGET_VAR = 0.04`（≈实测散度），让 `L_reg` 与 `L_var` 目标同量级、不再对拉。
- **P1**：`L_var = MSE(log σ², log s²)`（s² stop-gradient）。最小值仍 `σ²=s²`，但梯度尺度无关，小 σ² 也能拿到足够跟踪梯度。

**验证标准**（重训后复跑 `eval_sigma_diagnostic.py`）：σ² CV 从 7% 升向 ~41%、Pearson(σ², s²) 从 0.04 升向显著正相关、σ²/s² → ~1×。注：`L_var` 改 log 空间是对方法论 §5.4 线性 MSE 的**有据偏离**（同最小值、仅优化 landscape 更良态）；σ₀² 需在重训后复测校准，flickr 量级可能不同。

---

### 6.5 训练稳定性大修（cover 拆分 · L_var ramp · 阻断 L_set→σ² 梯度 · 5 段调度 · 坍塌监控）

σ² 在 Main 阶段激活瞬间坍塌到 floor（训练日志：σ²img 0.037→1e-4，Reg→36，R@1 0.395→0.046）。根因：L_var 在 σ² 高于**未成熟**的 caption 散度 s²（~0.015，文本头尚未区分 K 条 caption）时**硬激活**（0→1），log 空间 L_var（§6.4）把 σ² 猛拉向 s²，叠加 L_set 的不确定性折扣把 σ² 压向 0，弱化的 L_reg（σ₀²=0.04）拉不住，floor 把 σ² 困死。

修复（5 项联合）：

1. **拆分 L_cover**：`L_cover_pos`（正覆盖，§5.4 canonical）与 `L_cover_neg`（可选负排斥）**分离加权**（`λ_cover_pos=0.5`、`λ_cover_neg=0.0`），不再共享一个权重。便于"有/无负排斥"消融。
2. **阻断 L_set→σ² 梯度（Warmup）**：`uncertainty_grad_alpha` 用 straight-through 缩放 `eff_logvar = logvar.detach() + α·(logvar − logvar.detach())`——前向分数不变，只缩放 L_set 对方差的梯度。Warmup α=0（L_set 不再压低 σ²），Var-Bootstrap 渐增到 1。
3. **L_var 按 optimizer step 线性 ramp**：Var-Bootstrap 阶段 `λ_var·L_var` 从 0.05 线性增到 1.0（`var_ramp`），避免 epoch 边界 0→1 硬切的梯度方向突变。验收：刚启用时 `λ_var·L_var ≪ L_set + λ_mu·L_mu`。
4. **5 段调度**：Mean-Warmup → Var-Bootstrap（渐增 L_var）→ Pos-Coverage（渐增正覆盖）→ Neg-Repulsion（最后加负排斥）→ Full-Cov（最后加 L_cov）。验收：正覆盖启用前 σ² 已能跟踪 caption 散度。
5. **坍塌监控**：每 batch 记 `variance_floor_ratio`（σ² 接近 floor 的维度占比）；连续多 batch `ratio>0.5 且 mean σ²<2·floor` 时发 SEVERE 警告。**不提高 floor**（只会掩盖坍塌）。

**首轮稳定性配置**：`λ_cover_neg=0`、`λ_cov=0.01`（Full 阶段 cov 仍有再崩风险；官方目标仍 0.2，稳定后再测 0.05/0.1/0.2）、`λ_cover_neg` 稳定后扫 0.05/0.1/0.25/0.5。验收：σ²img 全程 >~0.02（不触 floor）、`Var` 随 s² 成熟而下降、R@1 超过 0.395、无 SEVERE 警告。

## 7. 推理

- **图文检索**：用均值 `μ`（确定性）计算相似度；可选 UC 相似度（σ 参与排序）。
- **VQA**：`μ` 确定性推理；或 Monte-Carlo 采样 `z ~ N(μ, σ²)` 多次平均以估计不确定性。

---

## 8. 关键代码位置

| 模块 | 文件 |
|---|---|
| 模型（分布头、融合、cov head） | `models/mcdisp_align_model.py` |
| 损失（含 `L_cov` 与 NaN 保护） | `losses/mcdisp_align_losses.py` |
| 训练循环（梯度裁剪、分阶段调度） | `scripts/train_mcdisp_align.py` |
| 超参与调度比例 | `config.py`（`MCDISP_ALIGN_*`） |
