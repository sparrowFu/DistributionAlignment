# MCDisp-Align 训练与实现诊断（2026-09-23）

修订约束：以 [论文正文](D:/code/paper/iclr2027_conference.tex:118) 为唯一方法依据。已删除更换模型、损失、梯度路径与检索打分的修复建议。下文保留的公式性质与日志现象属于诊断观测，不等于论文实现 bug；最终修复范围见 [修复方案](D:/code/DistributionAlignment/MCDisp-Align修复方案.md)。论文明确采用集合级匹配训练和均值 cosine 标准检索，并已报告 Flickr 标准检索落后 CLIP-FT，不能以更改方法来保证所有指标超过 baseline。

结论：这批实验包含两个不同问题。最佳模型在 KL 启用之前就已保存，因此“best 不如 baseline”首先应检查随机 MLP 重建对齐空间、集合中心训练与单句检索不一致；第 3 轮以后的进一步下降，则与 KL 和不确定性梯度同时启用、原始特征尺度收缩、覆盖约束缺少排他性高度一致。日志不能把两者全部归因于 KL，更不能据此断言整个分布建模思路无效。

本次只检查代码和日志，新增本报告及独立审计脚本，没有修改训练实现或超参数。当前工作区没有训练 checkpoint、数据集或完整统一测试集评估结果；本机 Python 也没有 torch/transformers。数值验证使用 NumPy，验证公式的性质，不冒充重训练或真实模型梯度实验。

## 1. 以日志为准的实际运行

来源：`logs/train_mcdisp_align.log`、`logs/train_clip_baseline.log`，并交叉检查 `logs/seed42_full` 的控制台记录。

| 项目 | CLIP baseline | MCDisp-Align |
|---|---|---|
| 数据集 | COCO、Flickr | COCO、Flickr |
| seed / batch | 42 / 32 | 42 / 32 |
| 骨干 | 图文均微调 | Freeze CLIP=False，骨干微调 |
| 学习率 | 1e-6 | 骨干 1e-6，头 5e-5 |
| epoch | 5，完整完成 | 计划 10，两组均在第 5 轮 early stop |
| 实际损失 | 随机单 caption 双向 NCE | kl，lambda_kl=1，集合中心 NCE |
| 其他权重 | — | cover_pos=.5，cover_neg=0，cov=.01，reg=.01 |
| 选模 | 验证 batch loss | 全验证集单句 cosine_mr |
| 最佳 checkpoint | 第 5 轮 | 两组都是第 2 轮 |

COCO train/val=106459/11828；Flickr=25744/2860。两个训练器均使用独立、同 seed 的 `random_split` generator，当前代码未发现因 MLP 初始化消耗随机数而改变划分的问题。相同划分并不意味着训练顺序和随机 caption 采样完全相同。

注意：日志打印 mu=.5、var=1，但 kl 分支用的是 lambda_kl 替代这两个权重；调整 `--lambda-mu`、`--lambda-var` 不会改变当前 KL 目标。`config.py` 的冻结默认值也不代表此次实际运行。

## 2. 最直接的退化证据

以下 mR 为日志中的 **mR(mc)**，即不确定性打分的单句多 caption 双向 Recall@1/5/10 均值；不是 R@1，也不是 cosine_mr。各轮 cosine_mr 未完整打印，只能拿到最终最佳值。

| 数据集 / epoch | Train NCE | mR(mc) | 验证文本原始坐标离散度 between | 负样本落入球形近似区域 |
|---|---:|---:|---:|---:|
| COCO / 2 | .0246 | 63.5% | .0265 | 0.0% |
| COCO / 3 | .0209 | 60.6% | .0207 | 8.1% |
| COCO / 5 | .0160 | 60.7% | .0141 | 45.3% |
| Flickr / 2 | .0175 | 88.5% | .0321 | 0.0% |
| Flickr / 3 | .0147 | 86.7% | .0262 | 0.6% |
| Flickr / 5 | .0106 | 85.9% | .0179 | 5.9% |

两个数据集在同一个阶段边界变差，而集合 NCE 持续改善。第 2→5 轮，COCO 的 between 减少约 47%，Flickr 减少约 44%。第 5 轮 KL 加权贡献分别约 .079 和 .108，NCE 仅约 .016 和 .011。损失数值比不能替代梯度比，但它明确说明训练已强烈强调分布对齐而非仅优化排名。

第 3 轮同时开启 lambda_kl ramp 和 uncertainty_grad_alpha ramp，不能只根据时间关联把损失全部归咎于 KL。当前仅记录这一因果判断限制，不再将分别关闭这些项列为修复方案。

`neg_in_ball` 只是以平均对角方差构造的欧氏球代理（trainer:499 起），没有使用每样本完整 Mahalanobis 椭球和 U，因此 45.3% 不能解读成“真实模型有 45.3% 负例被覆盖”或误分类率。它是几何退化的警讯。

## 3. 为什么加 MLP 后 best 也可能不如直接微调

### 3.1 预训练对齐被两个独立随机头改写

`models/mcdisp_align_model.py:125` 起，图文均值头是独立的 Linear→Dropout→ReLU→Linear；`:183` 使用 Xavier 初始化。均值直接取 MLP 输出，没有保留原始 CLIP 特征的恒等通路。

因此并不是在 baseline 的检索能力上自然叠加一个可选分布模块：图文原有公共坐标系先被分别变换，再由小数据、batch=32 的训练重新对齐。头学习率还是骨干的 50 倍。这个结构具备退化条件，但“50 倍”本身不能单独证明学习率过大，也不能从没有 epoch-0 评估的日志量化初始化损失。

**确定的实现 bug：残差头开关没有接通。**

`scripts/train_mcdisp_align.py` 把 `--mu-residual` 传进 `MCDispAlignTrainConfig`，配置类也有该字段（trainer:551）；但 trainer:717 的 `MCDispAlignModel(...)` 没有 `mu_residual=cfg.mu_residual`。模型只能回退到全局配置；当前 `config.py:493` 为 False。即便 CLI 加了 `--mu-residual`，在此代码和全局配置下仍然是普通随机头。日志没记录该值，无法证明服务器曾另行修改过全局配置。

该扩展开关的问题不作为本次论文修复的必做项。论文未明确采用残差头，不据此新增残差结构或改变初始化；只核实论文实验实际使用的头并记录结构。

### 3.2 集合中心 NCE 不监督每句 caption 的排名

baseline 在 `scripts/train_clip_baseline.py:129` 随机采样一条 caption 做 NCE；MCDisp 在模型 `:419` 合并五条文本分布，再在 `losses/mcdisp_align_losses_kl.py:248` 对 B 个图像与 B 个集合中心做 NCE。

评估和选模却在 B/N 个图像与 B*K/N*K 条单句之间检索（trainer:454、utils/retrieval.py:202）。中心靠近图像，并不保证偏离中心的每条句子靠近图像或能压过负例。小 batch 上辨认经过平均的中心更容易，NCE 很低不能当作比 baseline 更好的证据。

项目已有 `kl_capnce` 分支，在 `losses/mcdisp_align_losses_kl_capnce.py:46` 添加逐 caption 对比监督；**此次日志是 kl，没有使用它**。该分支对图搜文实际使用均匀多正例交叉熵，不宜误判为每个正例互相被当成独立负例；其理想 I2T 下限包含 log(K)，总 loss 不宜直接与单正例 NCE 的绝对值比较。

## 4. 为什么分布目标启用后进一步退化

### 4.1 余弦目标与原始坐标高斯目标的尺度没有固定

NCE 内部对均值做 L2 normalize（KL loss:251），KL 和 coverage 却使用未经归一化的均值（:299、:355）。文本集合方差由下面两项组成（model:310）：

`v_C = mean(v_caption) + Var(mu_caption)`

于是均值整体缩小可以保持全部余弦相似度不变，却降低原始空间均值距离和 between-caption 方差。第 3 轮以后，NCE 还能通过 `text_logvar → moment_matching → text_mus` 反传，奖励较小的集合方差。在正样本 logit 已高于负样本的情形下，缩小该方差相当于降低有效温度，可继续压低 NCE。

KL 的目标 detach 只阻断当次 KL 到文本的梯度，并没有冻结文本编码器，也没有阻断上述 NCE 路径；文本仍是随训练移动的目标。

NumPy 检查构造了符合当前公式的 B=32、K=5 输入，把图文均值缩放为原来的 1/4：

- 余弦 NCE 完全相同，约 .000232554。
- 不确定性 NCE 从 .000393522 降至 .000350052。
- KL 均值项从 .08129 降至 .00758，方差项从 .05775 降至 .00058。
- 文本尺度到不确定性 NCE 的有限差分导数为正，说明该合成样例中梯度下降确实奖励缩小文本尺度。

这是“损失改善可以不对应检索改善”的反例，不是该真实训练的逐步回放。联合缩放路径比较的是前向目标值，KL 目标 detach 下的实际更新不等于沿联合缩放路径的全导数。原始 between 降低也可能仅是范数变化；要判定语义坍塌，必须同时记录归一化后的句间角距离、均值范数和跨图间隔。

### 4.2 学到的主要是接近常数的方差，而非强判别语义范围

`L_reg` 将图像和每句文本方差拉向 .04（KL loss:417）。KL 对齐的是包含该文本方差的整个 v_C，不是纯 caption between spread。COCO 第 5 轮训练日志中合并文本方差 .0552，其中 between .0151，剩余约 .0401 来自逐句方差。这表明先验仍占很大部分。

验证图像方差的跨样本 CV 只有约 .04，样本差异很小。不能因为 corr(v_v, between) 提升，就推断范围建模能提升检索：相关性提升与检索下降在此同时发生。

### 4.3 覆盖可以通过膨胀 U 完成，缺少排他性

日志 cover_neg=0。`L_cover_pos = relu(d_M² / D - 1)` 只要求正例位于区域内；放大 U 沿残差方向扩张区域，也可降低它。当前 KL、方差 prior 不约束 U，L_cov 只比较 QR 后的方向、不约束幅度（KL loss:400）。

第 5 轮 U/diag 已达 COCO .83、Flickr 2.08，覆盖损失接近零，但检索未恢复。这支持“覆盖目标很容易完成，却不等于学会区分负例”的解释。它不能解释第 3 轮的初次下降，因为那时 coverage 尚未启用。

## 5. 协方差没有有效进入当前检索打分

实际 `_sim_matrix` 和 `utils/retrieval.py` 使用：

`score(i,j) = cos(mu_i, mu_j) / [tau * sqrt(1+mean(v_i)) * sqrt(1+mean(v_j))]`

直接后果：

1. U 完全不参与分数，方差向量也只保留一个平均值；模型无法直接利用学到的覆盖方向检索。
2. 固定一个图像查询时，其图像方差项是所有候选共有的正数，不改变排序；图像方差不会使该图像的图搜文结果体现更宽的语义范围。
3. 图像方差会影响文搜图的候选惩罚，但同样没有方向信息；若所有样本方差近似相同，分数与普通余弦更接近。
4. 这不意味着分布损失完全无作用，它仍会通过训练改变均值；只是额外参数没有对应的直接检索表达力。日志中 cosine 与 MCDisp 的中心 Recall 接近也符合这一点。

按论文 236、248 行，标准检索明确使用均值 cosine；以上不确定性分数用于论文训练及现有额外诊断。U 不进入标准检索是论文设计，不能作为引入 likelihood/coverage scorer 或协方差体积约束的理由。`utils/distribution_score.py` 的旧说明不是此次论文训练的依据。

## 6. 分阶段训练和记录问题

- 总计 10 epoch 时，KL 第 3 轮开始，coverage 第 5 轮开始，cov 第 9 轮开始；patience=3 使两组都在第 5 轮退出。协方差方向监督从未启用。
- best 均在第 2 轮，实际加权 KL、coverage、cov 均为 0。best 是随机均值头 + 集合 NCE + 方差 prior 的 warmup 模型，不是完整方法。要解释 best 与 baseline 的差距，不能使用第 5 轮的 KL 退化作为直接原因。
- `--no-staged` 也有实现缺口：它只影响 `stage_multipliers`，`train_epoch` 仍在 :261 无条件调用 `var_ramp/alpha_schedule`，不接收 no_staged。用它做“全部损失从第一轮开启”的消融会得到错误实验。
- `Best model saved (val_loss: inf)` 是打印分支问题：cosine_mr 模式仍打印未更新的 best_val_loss。最终标成 recall@1 的 .6364/.8858 实际是最佳 cosine_mr。并不是出现 infinite loss，也不是实际 R@1。
- baseline 的 Val Acc=.9662/.9863 是 batch 内约 32 个候选的准确率；MCDisp 的 Recall/mR 是整个验证集的检索。两者不能直接比较。当前日志不能给出严格的 baseline-vs-MCDisp 同协议测试差值。

## 7. 仅保留论文一致性与工程修复

以 [修复方案](D:/code/DistributionAlignment/MCDisp-Align修复方案.md) 为准，按以下顺序开展：

1. 对照论文核实 kl 分支、四项对齐损失、detach 和阶段梯度调度，区分论文参数与本批日志参数。
2. 修复有效配置记录、选模指标标签、checkpoint 元数据与精确恢复。
3. 对 moment matching、KL、Mahalanobis、方向对齐和 prior 做公式等价验证，不替换任何目标。
4. 按论文的标准 cosine 检索、one-to-one 候选构造与 WSD/BSD/RSD 定义重新核对结果。
5. 核查本批日志与论文表格是否对应同一配置、数据及 checkpoint；保持论文的三轮早停规定，不擅自延迟。

已移除残差/Identity 头改造、逐句 NCE、单位球 Gaussian、固定或 EMA 文本目标、新协方差损失、分布 scorer、额外协方差正则和强制完整训练阶段等建议。上述删除项不再属于后续执行计划。

可复核产物：`tmp/mcdisp_audit_20260923/audit.py` 和 `audit_results.json`。它们只保留源码/日志检查和公式性质的数值验证，不改变模型，也不授权引入新方法。
