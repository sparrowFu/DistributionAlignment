# MCDisp-Align 修复方案：严格遵循论文

依据：[iclr2027_conference.tex](D:/code/paper/iclr2027_conference.tex)。本方案取代此前版本；唯一目标是修复工程实现并核对其与论文的一致性，不重新设计模型、损失、梯度路径或检索方法。

本次仅修订方案与诊断文档，未修改训练代码、模型代码、损失代码或论文。论文没有写明的细节，核查原始实验配置和 checkpoint 元数据，不自行补成另一套方法。

## 1. 论文规定的内容保持不变

| 项目 | 论文规定 | 对应正文位置 |
|---|---|---|
| 表示 | 图像 Gaussian，协方差 diag(v)+UUᵀ；文本 Gaussian，协方差 diag(v_t) | 118–142 行 |
| 文本集合 | 等权 moment matching；均值平均，协方差含平均逐句方差与句间离散度 | 146–167 行 |
| 匹配分数 | 均值 cosine 除以固定温度及图像 base variance、集合 marginal variance 的折扣 | 179–191 行 |
| 匹配目标 | 图像与 caption 集合之间的双向对比损失 | 192–205 行 |
| 对齐目标 | 对角 KL、正覆盖、低秩方向对齐、图文 log-variance prior 四项 | 207–234 行 |
| KL | 方向为图像到集合；使用图像 base variance；目标 detach；目标方差下界 .01 | 228 行 |
| 覆盖 | 使用图像完整协方差的 Mahalanobis 距离；阈值 d²/D=1；文本中心 detach | 230 行 |
| 方向 | QR 与 caption 主方向匹配，r₀=min(r,K−1,D) | 232 行 |
| 方差先验 | v₀=.04；不额外约束 U 的幅度 | 234 行 |
| 训练 | 两个模态的 heads/backbones 由 matching 和 prior 更新；依次启用 KL、coverage、direction；方差 matching 梯度渐进开启 | 234 行 |
| 标准检索 | 独立编码图像和每句文本，按均值 cosine 排序 | 236、248 行 |
| heads LR / batch | 5e-5 / 32 | 245 行 |
| 早停 | 连续三轮验证无提升则停止，保留验证最优 checkpoint | 245 行 |

因此，不把训练中使用集合匹配、标准检索使用单句 cosine 视为需要替换目标的代码 bug；也不把 U 不直接进入标准检索视为必须改成概率打分的缺陷。这些都是论文明确规定的设计。

论文对投影头只写了 lightweight projection heads，未在所读正文明确规定残差结构。保留现有论文实验对应的头，不再建议新增 Identity/residual 结构或改变初始化方式。

## 2. 保留的修复：配置、执行路径与可追溯性

### 2.1 固定论文运行入口

核对并显式记录论文使用的 kl 分支。正式复现实验不切换到 standard 或 kl_capnce，不加入逐句 NCE、negative coverage 或其他论文外目标。

将论文权重与代码字段清晰对应：

| 论文符号 | 当前字段 |
|---|---|
| KL 权重 α | lambda_kl |
| coverage 权重 β | lambda_cover_pos |
| direction 权重 γ | lambda_cov |
| prior 权重 η | lambda_reg |

论文 α 是 KL 系数，代码 uncertainty_grad_alpha 是方差匹配梯度的调度系数，日志必须分别命名，避免混淆。

日志已给出本批运行 lambda_kl=1、cover_pos=.5、cover_neg=0、cov=.01、reg=.01、r=4、骨干 LR=1e-6；这些标记为“本批日志配置”，不伪称全部在论文中明文给出。论文未写明的数值，应追查产生论文表格的配置、代码版本和 checkpoint，不自行改权重。

### 2.2 记录实际生效配置

启动时输出完整解析后的配置，以及实际模型结构、有效参数组、学习率和可训练参数数量。kl 分支应注明 lambda_mu/lambda_var 不直接参与该目标，不能让打印出来的历史字段误导读者。

每轮记录基础权重和调度后的实际权重、uncertainty_grad_alpha、当前阶段、成功/跳过更新数。修复调度错误只以恢复原定论文执行路径为限，不重排阶段、不永久切断梯度、不自行增加冻结阶段。

此前的 mu_residual、no_staged 扩展路径不作为本次论文修复的必做项；不得借修复扩展开关引入残差模型或无分阶段训练。

### 2.3 统一 checkpoint 信息

best 与 last 都保存并核验：
- 模型结构配置、损失标识与版本、实际训练配置。
- epoch、global_step、阶段、验证指标名和值、seed、数据划分标识、代码版本。
- 用于精确恢复的 optimizer、调度和随机状态。

同目标同结构才能 resume；结构或损失不匹配时不自动恢复优化器。加载顺序应为确认结构→加载参数→建立/恢复匹配的优化器，保证参数引用正确。

保存再加载应得到相同推理结果；恢复训练应在确定性条件下复现下一步更新。旧文件与新记录分开保存，不覆盖历史实验。

## 3. 保留的核查：逐项验证实现是否等价于论文公式

只修“公式写的是 A，代码却算了 B”的实现偏差，不改公式本身。

| 核查项 | 验证方式 |
|---|---|
| moment matching | 小张量手算 mean 与 covariance diagonal；分母为 K，保留逐句方差与句间方差两部分 |
| set matching | 手算 B×B 正负配对、两个方向和温度；确认图像折扣使用 base variance |
| KL | 与小维度 Gaussian KL 参照计算比较；一致地处理 .01 target floor；确认方向和 /D |
| coverage | 小维度稠密求逆与 Woodbury 对照；覆盖区域内为零，边界外产生损失 |
| direction | 核对 QR/PCA、r₀ 和系数；K=1、重复 caption、退化方向时检查数值行为 |
| prior | 确认图像与每句文本 log-variance 的平均方式及 v₀=.04 |
| detach | 单独反向 KL/coverage/direction 时，caption 目标无梯度；matching 和 prior 保留论文要求的更新路径 |
| schedule | 实际逐步权重和方差梯度 ramp 与已核实的原始配置一致 |

诊断梯度时各项损失分别计算范数/夹角，只读记录，不使用梯度投影、重加权或改变 backward 路径。

可增加非有限值、方差极值、U 能量、均值范数和有效 caption 数的监控。数值实现优化必须通过公式等价验证，不能借“稳定性修复”新增协方差体积惩罚、修改 floor/prior 或换 KL。

原始离散度下降和代理球覆盖率上升仅作为观测；不能仅据此认定论文方法实现错误，更不能据此自动更换目标。

## 4. 保留的修复：评估与论文口径一致

### 4.1 标准图文检索

严格按论文 248 行：
- 图像与每条 caption 独立编码。
- 使用分布均值 cosine 排序，不使用 uncertainty score、likelihood、CSLS 或混合分数作为主表结果。
- I2T 为任一正确 caption 命中；T2I 为每条 caption 找对应图像。
- 分别报告两向 R@1/5/10；mR 可作明确标注的汇总，不冒充 R@1。

核查 baseline 和 MCDisp 使用相同 image pool、caption gallery、有效样本数和 ID 映射。保留 baseline 原方法，不把其训练目标替换为 all-caption NCE。

### 4.2 One-to-one 检索

论文 281 行明确：MCDisp 对五条 caption 分布做 moment matching，CLIP/ProLIP 使用单条 caption。按此复现，清楚记录不同方法的文本构造方式，不擅自改成所有模型都聚合五条。

该段对具体单 caption 选择和 one-to-one scorer 未完全展开；需核对生成论文表格的原评估配置，不将自己的推断写成论文要求。将该协议与标准检索分别报告。

### 4.3 几何指标与辅助指标

WSD/BSD/RSD 按论文 316 行的 cosine distance 定义计算，核查图像/文本采样、随机 seed 和权重。

AllHit@5 不是所读正文中的主表指标；若保留辅助输出，修复零基 rank 的边界为 <5，单独标注，不用它替换论文指标。

训练日志里的 batch Val Acc、集合 Recall、单句 Recall、mR 分开命名。已有 cosine_mr 选模时的 val_loss=inf/recall@1 标签错误应修复；论文只写 best validation score，没有明确是哪一项，因此选模指标须标记为原日志/实验配置来源。

## 5. 保留原训练规定，调查早停现象

继续采用论文规定的 heads LR=5e-5、batch=32、连续三轮无提升早停。删除此前建议的 heads LR=1e-5、关闭 dropout、固定表示训练、推迟 early stop 及强制跑完整阶段等方案。

本批日志显示 best 在第 2 轮、在第 5 轮早停、direction 阶段未启动。对此应核查：
1. 这批运行是否就是论文表格对应的代码、配置、数据划分及 checkpoint。
2. 验证指标是否算对、选对、记录对。
3. 阶段实际执行是否与原实验配置一致。
4. 是否存在参数恢复、数据过滤、数值异常或 checkpoint 选择错误。

“必须跑到最后阶段”不是论文给出的 early-stop 例外，不能为让训练继续而自行改规则。若原论文配置与当前日志不同，明确记录差异并复现真实配置；没有记录支持时，不编造一个配置以追求表格数值。

## 6. 实施顺序与交付

| 顺序 | 工作 | 完成标准 |
|---|---|---|
| 1 | 建立论文—配置—代码映射 | 明确哪些来自论文、哪些来自日志、哪些仍无证据 |
| 2 | 修复日志、选模标签、保存/恢复 | 能准确复核每个 checkpoint 及其训练阶段 |
| 3 | 公式、detach、schedule 等价测试 | 结果符合论文公式，排除实现偏差 |
| 4 | 按论文协议重评现有 checkpoint | 主表 scorer 和候选构造准确，结果可追溯 |
| 5 | 使用已核实的原配置复现训练 | 仅包含工程修复，模型/损失/训练规则不变 |

涉及文件主要为训练器、日志与 checkpoint 逻辑、评估脚本和一致性测试。模型/损失文件仅在证实其实现违背论文公式时做等价纠错，不能主动改结构或换目标；本次请求只更新文档，并未执行任何此类代码改动。

论文 276 行本身已报告 Flickr 标准检索落后 CLIP-FT。成功标准是忠实复现与可靠定位实现偏差，不是通过改变方法保证所有数据集都超过 baseline。
