# MCDisp-Align 审查分诊：哪些需要修改、哪些不需要

分诊日期：2026-08-31
依据：[MODEL_PAPER_CONSISTENCY_REVIEW.md](MODEL_PAPER_CONSISTENCY_REVIEW.md)（M01-M18）、[ALIGNMENT_CONSISTENCY_TODO.md](ALIGNMENT_CONSISTENCY_TODO.md)（A01-A16、O01、R01-R05）、本仓库当前代码逐项核实。

每条判定都在本仓库代码上重新验证过，不是照抄审查报告。判定分四档：

- **【必改】** 正确性/安全性/协议缺陷，不修会影响结果合法性或训练稳定性
- **【建议改】** 配套工程，随必改项一起做才完整
- **【决策点】** 研究方向问题，需要用户拍板，不该由实现方单方面定
- **【不需要】** 无代码问题（论文措辞/图层面），或经核实当前数据下不触发

---

## 0. 最紧急：R01 属实，且比报告写的更具体

**核实结果**：`utils/eval_common.py:63` 的 COCO 评测分支用 `config.CAPTIONS_PATH`
构建数据集——即 `TrainDatasets/mscoco_captions/captions/train-00000-of-00001.parquet`
（**训练 parquet**），再 `randperm(seed=42)[:5000]` 随机抽 5000 张。三个模型
（clip_baseline / prolip / mcdisp_align）的训练集都来自**同一个 parquet 的 90%**
（train 脚本同样 seed=42 做 random_split）。

**后果**：
1. 今天的 E0 统一口径 COCO 基线（clip 0.6088 / prolip 0.6415）与历史 MCDisp COCO
   数字（0.4972 等）**全部是"训练池评测"**——三个模型互相可比（同池同样本），
   但绝对值虚高，且不能作为论文的 COCO 测试数字。
2. **Flickr 不受影响**：registry 里 `flickr` 的 train_kind/eval_kind 是真分开的
   train/test split（`dataset_registry.py:55-58`）。
3. 本地**没有独立 COCO 验证/测试 parquet**（captions 目录只有 train 一个文件），
   唯一现成的 held-out 是三个模型都没训过的 **seed-42 random_split 的 10% val
   部分（11,828 张）**。

**修复选项**（按干净程度）：
- **A（立即可用）**：COCO 评测改为从 seed-42 random_split 的 **val 11,828 张**
  里取固定 5000 子集。三模型统一、真 held-out。注意：MCDisp 的检查点选择用了
  val（select_by=mc_recall@1），val 当测试有轻微选择泄漏，论文需注明，或后续
  把选择集与测试集再分开。
- **B（论文标准口径）**：取得 Karpathy 5k test split 的 image ID 清单
  （karpathy_split json，公开可得），按 ID 从本地图片目录构造测试集。最干净，
  但要新增数据准备步骤。
- **C（最彻底）**：训练改三段切分（train/val/test），重训三个模型。成本最高。

**判定：【必改】，优先级最高。** E0 的 COCO 表在修复后需重跑（每模型 ~10 分钟）。

---

## 1. 逐项判定总表

| 审查条目 | 核实结论 | 判定 | 依据/理由 |
|---|---|---|---|
| **R01 评测划分** | 属实（见第 0 节） | 【必改】 | COCO 评测在训练池取样 |
| **A02 完整方差** | 属实 | 【必改】 | 见 2.1 |
| **A05 方向退化** | 属实 | 【必改】 | 见 2.2 |
| **A06 梯度守卫** | 属实 | 【必改】 | 见 2.3 |
| **A07 中心化矩匹配** | 属实 | 【必改】 | 见 2.4 |
| A08 日志拆分 | 随 A02 | 【建议改】 | 见 2.5 |
| A12 消融拆分 | 属实（`run_ablation.py:49` 确认 no_dir 同时关损失+移除头） | 【建议改】 | 见 2.6 |
| A13 版本记录 | 有必要（马上又要改目标） | 【建议改】 | 见 2.7 |
| R03 mu 头 dropout | 机制属实、量级未测 | 【建议改】（先诊断） | 见 2.8 |
| **A16 L_match 高斯重叠对比** | 需求成立、路线存疑 | 【决策点】 | 见 3.1（四个硬冲突） |
| **A01 显式 L_mu 中心 MSE** | 数学事实成立 | 【决策点】 | 见 3.2 |
| A11 新评分器 | 仅当 A16 采纳才需要 | 【决策点·附属】 | 随 A16 |
| A09/A11 旧诊断评分器 | 属实但只影响辅助路径 | 【不需要（暂）】 | 见 4.1 |
| M15 caption 补齐 mask | 代码属实、数据不触发 | 【不需要】 | 见 4.2 |
| A10 分布对齐评估指标 | 好东西但属实验增值 | 【不需要（暂）】 | 论文分析章要用再做 |
| M13 框架图 / M10 措辞 / M02 M03 M07 论文补写 | 论文/图层面 | 【不需要（代码）】 | 归入论文修改批次 |
| A14 回归测试 / A15 文档同步 | 收尾工作 | 【不需要（暂）】 | 随最终采纳的改动一起 |
| O01 谱匹配扩展 | 超出最小目标 | 【不需要】 | 仅当用户扩展研究目标 |
| M01 M04 M06 M14 M16 M17 M18 | 分别并入上面对应项 | — | — |

## 2. 必改与建议改的核实细节

### 2.1 A02 —— L_var 图像侧漏掉低秩贡献【必改】

核实：`losses/mcdisp_align_losses.py` 的 `_var_loss(img_logvar, text_logvar)` 只用
`exp(img_logvar)`（对角分量 d_v），而文本目标 `text_logvar` 是**完整**集合方差
`log(s² + mean(σ_k²) + eps)`（`models/mcdisp_align_model.py` 矩匹配含 caption
方差）。图像侧真实逐维方差是 `d_v + Σ_r U²`（模型协方差定义 `Σ=diag(d)+UUᵀ`）。
审查报告的反例成立：U 贡献 0.04 时 `L_var=0` 但真实方差超目标 60%。

修复（比 TODO 写的小）：只需改图像侧——加共享函数
`image_marginal_variance(img_logvar, img_U)`，`_var_loss` 改为
`MSE(log(d+ΣU²+eps), sg(text_logvar))`，U 不 detach。文本侧**已经是完整目标，
不用动**（TODO A02 第 6 点描述的改动当前代码已满足）。

附带收益：U 的幅度从此有数据监督（同时解决 A03 的主体）。

### 2.2 A05 —— 方向项对退化 caption 集没有护栏【必改】

核实：`_dir_loss` 的 `r_eff=min(r, K-1, D)` 只是理论上限，不查实际谱秩。全部
caption 均值相同（或从头头部初始化期的近似相同——**这是训练早期必然出现的
状态**：随机初始化的头对 5 条不同 caption 的输出几乎一样）时，`G≈eps·I`，
方向损失退化为常数 2r、梯度为零，且 Qt 由数值噪声归一化而来。

修复（首版保守版即可）：在 detach 的偏差矩阵上按阈值估计实际秩；秩为 0 的
样本跳过方向监督并计数（`dir_skipped_zero_spread`）；实际秩 < r_req 的样本首版
也跳过并报告比例。热身 ramp（L_dir 前 10% 步从 0 起）已部分缓解时序问题，
但护栏本身仍需要。

### 2.3 A06 —— 前向有限 ≠ 反向有限【必改】

核实：`U=0` 时 `torch.linalg.qr` 前向正常、反向梯度含 Inf/NaN（QR 反向对
R 对角元求逆）。trainer 现有检查只看 loss 有限性（`train_epoch` 的
`torch.isfinite(loss)`），`backward` 后到 `optimizer.step()` 之间无梯度检查；
NaN 梯度进 Adam 会永久污染动量。U 初始化 std=1e-2 非零，自然触发概率低，
但 2.2 的跳过逻辑配合后这是廉价的保险。

修复：`backward` 后检查全局梯度范数有限性（`clip_grad_norm_` 返回值即可），
非有限则 `zero_grad` 跳过该步并计数（`nonfinite_grad_steps`）；按 2.2 跳过
奇异子批后再调 QR。

### 2.4 A07 —— 矩匹配用二阶矩相减，大中心小离散度时抵消【必改】

核实：`_moment_matching` 现为 `mean(var + mu²) − mean(mu)²`。数学正确、float32
下当坐标量级 >> 离散度时有效数字被吃掉。且损失端 `_dir_loss` 与诊断
`caption_spread` **已经在用中心化偏差**（`dev = mus − center`）——模型端改中心化
还能让三处统计口径统一。

修复：`combined_var = mean(caption_var) + mean(dev²)`（dev 中心化）。K=1 时两
种写法都退化为 caption 自身方差，行为不变。审查报告的手算例
（[9999,10000,10001]，var 0.04）从 ~1e-6 修复到 ~0.7067，可作回归测试。

### 2.5 A08 —— 日志字段与 A02 后的语义对齐【建议改，随 A02】

`img_var_*`/`floor_ratio` 目前只看对角分量。A02 落地后拆三组字段
（`img_diag_var_*` / `img_lowrank_var_*` / `img_marginal_var_*`），旧字段保留
原语义并标注兼容。不拆会把"残差分量贴地板"误读成"整个分布塌缩"。

### 2.6 A12 —— no_dir 混淆"关损失"与"移除模块"【建议改】

核实：`run_ablation.py:49` 确认 `no_dir: {cov_rank: 0, lambda_dir: 0}` ——同时
删头又关损失，消融结论无法归因。拆成 `no_dir_loss`（保 U 只关损失）与
`no_lowrank/diagonal_only`（结构消融）。**具体变体集等 3.1/3.2 决策后定**
（若加 L_mu 则补 no_mu；若采纳 L_match 则加 cosine_match 对照）。

### 2.7 A13 —— 检查点缺目标版本标识【建议改】

马上要再次修改训练目标（A02/A07 至少），新旧 checkpoint 权重形状完全兼容、
语义不同。轻量方案即可：best/last 存 `objective_version` 字符串 + 全部损失权重
+ 关键开关；恢复时校验，不匹配则明确降级为 legacy 初始化而非静默续训。

### 2.8 R03/M14 —— mu 头 dropout 污染离散度统计【建议改，先诊断】

核实：四个头都有 `Dropout(0.1)`；统计（μ̄、s²、S_t）来自 `text_mu_head` 的
train 态输出，dropout 噪声会混入 s² 并经 L_var 进入图像方差。审查的合成例
（train 态离散度 5.27e-5 vs eval 0）量级很小，真实影响未测。

建议顺序：先加一个一次性诊断（同批 caption 在 train/eval 两态下的 s² 差），
若差值相对真实 s² 不可忽略，去掉 `text_mu_head`（及对称地 `img_mu_head`）的
dropout——统计两端都干净；logvar 头的 dropout 保留。这属于训练细节而非架构
变更（层数/维度不变）。

## 3. 两个决策点（需要你拍板）

### 3.1 A16 —— 高斯重叠对比 L_match：需求成立，路线有四个硬冲突

审查报告转述的**需求**（"每条 caption 的方差由模型学习、表示自身语义范围、
有数据监督"）是合理的研究目标，当前实现确实只有常数先验（L_cal）监督 caption
方差——这个缺口是真的（M05 成立）。

但审查提议的**路线**（把对比分数换成高斯重叠 `s_ij = log∫p_v·p_t`）我建议缓行：

1. **与论文自己的定位冲突**：`.tex` 引言（L72）批评现有概率方法的正是
   "uncertainty typically learned **indirectly through contrastive** …
   objectives, has no explicit connection to the actual semantic variance among
   ground-truth captions"。L_match 恰好让方差经对比目标隐式学习——等于把
   MCDisp-Align 的差异化卖点（σ² 由 caption 离散度**显式**监督）往被批评的
   PCME/ProLIP 模式上靠。
2. **项目史实**：2026-07 的 likelihood 式重写把检索 R@1 从 ~0.5 打到 0.03
   （温度塌缩 + 从头头部不稳，教训在案）。高斯重叠对比属于同一家族，且
   logdet/负例对方差塌缩没有硬保证（审查报告自己也承认，A16 风险节）。
3. **训练-评测打分再次失配**：检索主指标是均值 cosine（今天刚把三个方法
   统一到这个口径）。用重叠分数训练，要么评测也换分数（三方法又要重新对齐），
   要么重演刚修掉的失配。
4. **损失项数约束**：你本会话明确"损失项不需要特别多"。四组五原子项是膨胀。

**替代的诚实路线**（不写代码，只改论文表述，成本零）：如实写"caption 级方差由
先验校准，作用是把集合方差的第二组成部分保持在合理量级，使数据驱动的离散度
s² 保持主导"——这已经是我昨天写的 §3.3 续稿的原话；集合方差（真正监督图像
方差的对象）的主项 s² 完全来自数据。"caption 方差学到语义范围"的主张本来也
没有真值可验证（审查 R05 自己承认）。

**我的推荐**：暂不采纳 L_match；把它降级为**受控消融候选**——在损失重构
（第 2 节必改项）落地、基线重训稳定后，用小预算做一次 `overlap_match` vs
`cosine_match` 短程对照，稳定且检索不掉再全面换。若你坚持现在就上，我按
审查报告的 A11 分块评分器方案实现，但建议先接受"可能需要回滚"的预期。

### 3.2 A01 —— 显式中心 MSE（L_mu）：可加可不加，倾向不加

数学事实成立：cosine 对比对均值缩放不敏感，不约束"高斯中心位置"。但：
检索打分就是 cosine——**对检索目标而言方向即全部**；位置约束服务的是高斯
叙事。加第 5 项违反项数约束，且两个从头训练的均值头坐标系是自由学习的，
MSE 硬绑会压掉均值头的自由度（现在 i2t 0.5618 的成绩就是这自由度学出来的）。

**我的推荐**：不加 L_mu；论文措辞采用"均值对齐到 caption 集合、以共享中心
为锚"（§3.3 续稿已如此写，审查 4.1 节建议的措辞修改方向一致但保留中心直配
——两种措辞二选一即可，关键是全文统一）。若你想验证，做成 `mu_mse` 消融
变体小预算试跑，有效再转正。

## 4. 判定为"不需要"的说明

### 4.1 A09/A11 旧似然评分器：真实缺陷，但只影响辅助路径

`utils/distribution_score.py` 的坐标混用（归一化均值 × 原始方差）与
`eval_sigma_diagnostic.py` 的目标口径错误（把 s² 当完整目标）都属实，但**主
检索、训练、选点都不经过它们**。只有论文要报 likelihood 分数或 σ 语义分析
（Exp7）时才需要修。挂起，随实验章需要再修。

### 4.2 M15 caption 补齐/截断：防御性代码，当前数据不触发

`caption_dataset.py` 的 pad/truncate 逻辑属实，但 COCO parquet 每图恰好 5 条
caption、Flickr 恒 5 条，两条路径都不执行。有效 caption mask 的工程量不匹配
收益。**建议只加一步验证**（启动时断言或日志统计一次触发率）而非全面改造。

### 4.3 其余

- M13（框架图文图不符）、M10（Σ̄_t 主轴措辞）、M02/M03/M07（论文补写冻结策略/
  低秩参数化/矩匹配公式）：论文与图层面的修改，无代码问题。
- A10（分布对齐评估指标）、R05（语义范围分析）：实验增值项，论文分析章要用
  时再做。
- O01（谱匹配）：超出当前最小目标，单列。
- A14/A15（测试与文档同步）：随最终采纳的改动集合收尾，现在做会白做。

## 5. 建议的实施顺序（若按本分诊执行）

1. **R01 评测口径**（选项 A 先落地，Karpathy test 选项 B 并行准备）→ 重跑 E0
   COCO 基线（Flickr 不用动）
2. **损失正确性三件套**：A02（完整方差）→ A07（中心化矩匹配）→ A05+A06
   （方向退化护栏 + 梯度守卫），配 A08 日志与 A13 版本字段
3. **R03 诊断**（train/eval 离散度差）→ 视结果去 mu 头 dropout
4. **A12 消融拆分**（变体集按 3.1/3.2 的决策补全）
5. 全量单测（A14 相应部分）→ 短程验证 → 重训
6. 论文/图同步（A15、M13 等）

第 2 步改动全部落在 `losses/mcdisp_align_losses.py`、
`models/mcdisp_align_model.py` 的矩匹配、`utils/mcdisp_align_trainer.py` 的
步级守卫——不动模型架构、不加超参（A02/A05/A06/A07 均无新权重）。
