# MCDisp-Align 审稿回复初稿（中文版）

初稿参照 ICLR 模板逐条回应。每条回复下方附有证据示例和实验步骤，供作者组核对与回填。R1–R3 对应 [Review.md](D:/code/DistributionAlignment/Review.md) 中的三组意见。章节、公式和表格编号均指向 [投稿稿件](D:/code/DistributionAlignment/paper.pdf)。新实验数值标为 **待测**。新增训练和语义诊断集中在 MSCOCO 图文检索。Flickr30K 的标准检索补表使用可核实的已有权重。 [ICLR 回复模板](C:/Users/sparr/Desktop/rebuttal/rebuttal_iclr/4XWw.md)

## 审稿人 1

感谢审稿人仔细检查模型的梯度路径、检索设置和基线对照。

### 1. 梯度路径与协方差参数化（W1）

**回复。** 低秩因子 $U_i$ 的直接作用在训练阶段。标准检索按图文分布中心的余弦相似度排序，式（4）的图像端使用基础方差，式（6）的对角 KL 使用基础对角高斯分布。这三处计算均不直接依赖 $U_i$。当覆盖项权重非零且马氏距离超过阈值时，覆盖项会通过完整协方差向 $U_i$ 回传梯度。子空间对齐项在其权重非零时也会向 $U_i$ 回传梯度。该项使用 QR 分解得到的正交基来约束方向，对 $U_i$ 整体乘以正标量保持不变，因此无法单独约束低秩方差的大小。设置较小的 $\gamma$ 是为了平衡不同损失项的量级。鉴于附录 A 的方向项到第 9 轮才启用，我们会核实投稿所用检查点的保存轮次，并报告各损失项实际产生的梯度。

式（2）按全协方差公式合并单条描述的平均方差与描述中心之间的方差。审稿人指出的偏差可能出现在式（6）的图像端。对角 KL 将基础方差向描述集合的边际方差对齐，低秩项 $U_iU_i^\top$ 还会增加图像的边际方差。因此，图像的**总边际方差**可能超过对齐目标。我们会分别测量基础方差和总边际方差相对于目标的偏差，并报告真实批次中 $d^2/D>1$ 的样本比例。增大 $\beta$ 只能放大已经激活的覆盖梯度。

停止梯度操作切断了 KL、覆盖和方向对齐三项到文本侧目标的直接梯度。方差正则仍会更新文本方差预测头，匹配损失在方差梯度预热后也可以更新它。我们会测量不同描述的预测方差及相应梯度，并用固定方差 $v^t=v_0=0.04$ 的对照检验预测方差对检索的影响。

**证据示例与获取方式（E1）**

| 质疑 | 可展示的证据 | 怎么获得 |
| --- | --- | --- |
| $U_i$ 实际是否更新，$\gamma$ 是否太弱 | 逐轮报告有效权重、原始及加权损失、各项对 $U$ 和低秩头参数的梯度范数、最佳检查点的保存轮次。示例字段为 `epoch, best, weighted_dir, grad_U_cover, grad_U_dir` | 先核对表 1/4 的原始权重和训练日志。对固定的真实 COCO batches 分项求梯度并区分「无计算路径」和「有路径、数值为 0」。在同一完整日程下用 E4 的 Full 与 no-direction 比较。需要检验权重量级时，只增加一个 $\gamma=0.05$ 对照，保持启动轮次不变。 |
| Coverage 是否激活 | `p50/p90/p99(d²/D), Pr(d²/D>1), β·Lcover, grad_U_cover` | 用原始坐标和 Woodbury 距离，在覆盖项启用前后抽相同图像计算。以真实日志的非零权重为准。若实际活跃率接近零，如实报告。改阈值属于另一个修订设定。仅在原 hinge 有信号时，才用 E4 的 $\beta=0$ 和可选 $\beta=1$ 检验权重。 |
| 总边际方差是否偏大 | $\operatorname{diag}(\Sigma^v)-\operatorname{diag}(\Sigma^C)$ 的中位数和分位数，以及 $\lVert U\rVert_F^2/\sum_d v_d^v$ | 从同一 checkpoint 导出逐图 $v^v,U,\Sigma^C$，分别比较 base 和 total 的边际偏差，并与 diagonal-only 对照。如果要验证修改公式，把 KL 的图像端替换为 $v^v+\operatorname{diag}(UU^\top)$，以单独的修订变体报告。 |
| $v^t$ 是否退化为常数 | $v^t$ 的分位数、同一维度在不同描述间的方差、文本方差头的分项梯度，以及固定 $v^t$ 后的 I2T 排名变化 | 先在早期/后期各取固定 batch，分开记录 $\mathcal L_{\rm match}$ 与方差正则的梯度，确认运行没有额外启用 `sg_set_variance`。固定 checkpoint，将每条 caption 的 $v^t$ 设为 0.04 后重算 Eq. (2)，保留各 caption 均值及其 描述间方差。Eq. (4) 中该项可能改变集合 I2T 排名。在集合 T2I 中它是同一个查询对所有图像的公因子，标准中心余弦也不受影响。若需要评估训练贡献，再加固定 $v^t$ 的重训组。 |

**可直接引用的数学证据。** 记 $z=(\Sigma^v+\epsilon I)^{-1}(\mu^t-\mu^v)$，则 $\nabla_U d^2=-2zz^\top U$。活跃 caption 的 coverage 梯度还要乘 $\beta/(KD)$。阈值以内梯度为零。方向项的未归一化值介于 $0$ 和 $2\gamma r_0$ 之间，原稿最终权重下上界为 $0.08$。这是损失值的上界，梯度量级仍须分项测量。

**已有的诊断线索。** 仓库一组 9 月 23 日的 COCO/Flickr 日志显示最佳权重在第 2 轮、训练在第 5 轮早停。这批运行没有进入方向监督阶段。该 COCO 运行第 5 轮的未加权 Cover+ 为 0.0002，说明该次运行的 hinge 平均值很小但非零。它们与投稿表格权重的对应关系待核对。如果表格所用权重也在第 9 轮前选出，回信应明确该 checkpoint 未接受方向监督，并把共同跑满日程的实验单独报告。 [旧运行审计](D:/code/DistributionAlignment/tmp/mcdisp_audit_20260923/audit_results.json) · [COCO 旧运行日志](D:/code/DistributionAlignment/logs/train_mcdisp_align.log:44) · [KL、覆盖与方向实现](D:/code/DistributionAlignment/losses/mcdisp_align_losses_kl.py)

### 2. 原版基线的微调目标与同目标对照（W2）

**回复。** 我们有意沿用两篇原论文各自的图文配对训练目标来微调基线。**CLIP-FT 沿用 CLIP 的双向图文对比损失。ProLIP-FT 沿用 ProLIP 的成对概率训练目标，包括概率对比匹配、跨模态包含损失和方差正则。两组基线在每次更新时都从每张图像的标注中随机采样一条描述。MCDisp-Align 则在每次更新时联合使用每图的五条描述，并根据描述集合的统计量对齐图像表示。** 我们会在修订稿第 3.1 节明确说明各模型的训练目标和每次更新使用的描述数量。 [CLIP 原论文第 2.3 节](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf) · [ProLIP 原论文第 3.2–3.3 节](https://proceedings.iclr.cc/paper_files/paper/2025/file/9032e5c9ec394ce768a2fa9bdc56af6c-Paper-Conference.pdf)

各方法的训练标注来自相同的基准数据集。表 1 以同一批人工标注的五句集合评估所有方法，适用于描述已分组的场景。原有对比确实存在训练监督差异。表 2 另外评估了独立编码描述的标准检索。在 MSCOCO 上，MCDisp-Align 的图搜文和文搜图 R@1 分别为 69.6% 和 55.3%，CLIP-FT 分别为 67.6% 和 54.6%。我们将补充五句微调对照，同时评估集合检索和标准检索，以估计集合训练本身对增益的贡献。

**证据示例与获取方式（E2）**

| MSCOCO 对照组 | 训练目标 | 集合 I2T/T2I R@1 | 独立 I2T/T2I R@1 |
| --- | --- | --- | --- |
| CLIP-FT 原版 | 单句，CLIP 成对对比 | 82.5%/76.1%（表 1） | 67.6%/54.6%（表 2） |
| ProLIP-FT 原版 | 单句，ProLIP 成对概率目标 | 77.4%/73.8%（表 1） | 73.1%/56.5%（表 2） |
| CLIP-FT + Pair-5 | 五句分别配对，平均五个 CLIP 成对损失 | 待测 | 待测 |
| CLIP-FT + Set | 五句集合中心，双向集合余弦交叉熵 | 待测 | 待测 |
| ProLIP-FT + Set | 五句集合中心，双向集合余弦交叉熵 | 待测 | 待测 |
| MCDisp-Align w/o $\mathcal L_{\rm align}$ | 原式 (4)–(5) 的方差加权集合匹配 | 待测 | 60.8%/48.1%（表 4） |
| MCDisp-Align Full | 同上，加式 (6) | 85.2%/83.9%（表 1） | 69.6%/55.3%（表 2） |

原稿数字用于定位参考行。采用新图库或共同评分时，先统一复评原版，再填写正式比较表。

1. 先从同一 CLIP ViT-L/14 初始化比较 **CLIP-FT 原版 → CLIP-FT + Set**。Set 组将五个句向量分别编码后求集合中心，再计算 $B\times B$ 余弦 logits 及双向交叉熵。固定图像 ID、训练更新数、优化器和验证选模规则。为了隔离聚合和文本曝光量，再加一个 **CLIP-FT + Pair-5** 对照。每步读取同样的五条 caption，按 caption 槽位计算五个 $B\times B$ 的原版成对损失，再平均后更新一次。这样两组都看相同五句，同图的其他描述也不会被当成负例。
2. 若可复用原 ProLIP ViT-H/14 初始化，用同一 checkpoint 新训 **ProLIP-FT + Set**，保留编码器结构，将训练目标替换为与 CLIP-FT + Set 一致的集合中心对比。以 H/14 内的 pair→set 差值回答训练目标问题，并单列该组。原版 ProLIP 保留原生配方，作为原方法参考。
3. 为严格匹配目标，用同一个双向集合中心余弦损失跑 CLIP-FT + Set 与 MCDisp-Align 的控制组。MCDisp 控制组在两行都移除式 (4) 的方差折扣，分别令 $\mathcal L_{\rm align}=0$ 和保留 $\mathcal L_{\rm align}$，从而在同一概率架构内隔离式 (6) 的贡献。表格扩充这两行时标明「共享余弦训练」。原式 (4)–(6) 的 Full 和原生基线另行保留。若预算只够首批，先报告 CLIP Set−Pair-5 和原式 Full−w/o 两个对比，将共享目标下的跨架构比较列为待补。
4. 集合评估固定同一个五句图库，并行报告共享中心余弦和原稿各方法打分。独立评估逐句编码，以中心余弦排名。用同一批图的配对差值和置信区间展示 pair→set 和 no-align→align。若匹配后差距收窄，就报告收窄的幅度。若基本消失，则将主要增益归于当前实验中的集合监督。

仓库训练脚本能直接查到两组原版基线的 `random.choice`。本地 ProLIP 微调实现使用原论文的跨模态配对项和 VIB，因此正文表述为「沿用原论文的成对训练目标」。完整预训练配方还包含针对遮掩输入的额外损失项。集合训练目标需要补充实现，现有集合评测可复用。 [CLIP-FT 训练](D:/code/DistributionAlignment/scripts/train_clip_baseline.py:129) · [ProLIP-FT 训练](D:/code/DistributionAlignment/scripts/train_prolip.py:167) · [集合评测](D:/code/DistributionAlignment/scripts/eval_setlevel_all.py)

### 3. 标准检索、仅匹配消融与概率基线（W3）

**回复。** 表 2 的标准检索主对比只报告了 MSCOCO。表 4 已包含 Flickr30K 的消融结果，仍需补充 Flickr30K 的标准检索基线。在 MSCOCO 上，移除 $\mathcal L_{\rm align}$ 后，图搜文和文搜图 R@1 从 69.6% 和 55.3% 降至 60.8% 和 48.1%，低于 CLIP-FT 的 67.6% 和 54.6%。这一结果表明，在我们的架构中，仅用集合匹配训练时表现较弱。与 CLIP-FT 比较时，预测头结构和训练目标也存在差异。我们将用 W2 的同目标对照和 W4 的逐项消融检验这些因素。

ProLIP 的 ViT-H/14 结果同时反映了骨干、预训练和对齐方法的差异。我们会保留其骨干标注，并在 MSCOCO 上补充使用 CLIP ViT-L/14 初始化的 PCME++ 和 ProbVLM。对照使用同一测试图像和描述，并说明各方法的可训练参数范围和训练配置。Flickr30K 补表将使用可获得的、在 Flickr30K 上微调的检查点。

**证据示例与获取方式（E3）**

- **MSCOCO 标准检索补表。** 原稿表 2/4 的六个 Recall 作历史参考。在同一候选库上评估 CLIP-FT、Full、w/o alignment、PCME++、ProbVLM，主列统一中心余弦，附列可给各方法原生分数，例如 PCME++ 的 CSD。每行写骨干、初始化、冻结范围、更新预算和 checkpoint 来源。ProbVLM 使用冻结 ViT-L/14 加适配器。PCME++ 使用 ViT-L/14 初始化和自己的训练配方。若加入本地 ProLIP-L/14，应标为「CLIP 初始化的 ProLIP 目标微调」，注明缺少原版概率预训练并先验证初始化及训练是否正常。已有的冒烟或中途中断权重不填入主结果表。 [PCME++ 原论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ad9d6ab10446114cf5482d5e1f971a84-Abstract-Conference.html) · [ProbVLM 原论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Upadhyay_ProbVLM_Probabilistic_Adapter_for_Frozen_Vison-Language_Models_ICCV_2023_paper.pdf) · [本地基线配方与历史状态](D:/code/DistributionAlignment/experiments.md)
- **Flickr30K 缺失对照。** 表 4 已有 Full 76.9%/64.1%、w/o 69.1%/59.7%（标准 I2T/T2I R@1）。核对原表划分、checkpoint 和打分。先找现有 Flickr30K CLIP-FT、MCDisp-Align 权重，再找 ProLIP-FT 等已有模型，在同一 Flickr30K 测试图库补六项 Recall，按实测报告。`scripts/eval_flickr30k.py` 默认用于跨数据集迁移。可改用 `scripts/eval_full_metrics.py --dataset flickr --model … --checkpoint … --tag …`，取普通余弦列。需要核对 ProLIP 的加载配置。只找到 COCO 权重的行只能写「COCO→Flickr30K 迁移」。 [标准评测](D:/code/DistributionAlignment/scripts/eval_full_metrics.py) · [跨数据集评测](D:/code/DistributionAlignment/scripts/eval_flickr30k.py)

### 4. 逐项损失消融与描述集合检索（W4）

**回复。** 表 4 只报告了移除整个对齐目标后标准检索的变化。我们将补充 MSCOCO 的描述集合检索消融，分别移除对角 KL、描述中心覆盖、子空间对齐和方差正则，并比较低秩加对角协方差与纯对角协方差。另设去除三项描述目标的停止梯度和改变低秩因子秩 $r$ 的对照。去除方向监督时保留低秩分支，以便单独测量方向监督的贡献。如果某项消融的变化很小，我们会结合 W1 中的启用轮次、激活比例和梯度量级解释结果。

**证据示例与获取方式（E4）**

| 配置 | 只改这一项 | 集合 I2T R@1 | 集合 T2I R@1 | 集合 mR |
| --- | --- | --- | --- | --- |
| `full_model` | 原稿 Full，$\alpha=1,\ \beta=0.5,\ \gamma=0.01,\ \eta=0.01,\ r=4$ | 待测 | 待测 | 待测 |
| `no_align` | $\alpha=\beta=\gamma=\eta=0$ | 待测 | 待测 | 待测 |
| `no_kl` | $\alpha=0$ | 待测 | 待测 | 待测 |
| `no_cover` | $\beta=0$ | 待测 | 待测 | 待测 |
| `no_direction` | $\gamma=0$，保留 $r=4$ | 待测 | 待测 | 待测 |
| `no_reg` | $\eta=0$ | 待测 | 待测 | 待测 |
| `diag_only` | $r=0$，方向项无定义而跳过 | 待测 | 待测 | 待测 |
| `no_sg` | 三个对齐目标取消 detach，保留匹配方差梯度日程 | 待测 | 待测 | 待测 |
| 秩的敏感性 | 同 Full，单独跑 $r=2$，有余量再补 $r=1$ | 待测 | 待测 | 待测 |

已有八组可通过 `python scripts/run_ablation.py --config <组名> --dataset coco --loss kl --no-freeze-clip` 进入共享训练器。命令入口已存在，需先接上固定且互斥的划分、共同的验证选模指标与输出路径。当前 runner 默认冻结骨干，内置 summary 采用自己的 scorer 与抽样口径。表中应另用同一集合 evaluator 重评各组 checkpoint，确认 `diag_only` 按其实际 rank 加载并保留所有原稿 Recall@1/5/10。mR 为双向六个 Recall 的平均值。用相同 checkpoint 可低成本补标准余弦指标，与表 4 桥接。 [消融配置](D:/code/DistributionAlignment/config.py:561) · [消融入口](D:/code/DistributionAlignment/scripts/run_ablation.py)

Full↔no-direction 检验方向监督，no-direction↔diag-only 检验无方向监督时低秩分支在覆盖中的贡献，Full↔diag-only 检验整条低秩设计。做方向项机制对照时，两组按相同预算跑过第 9–10 轮，报告末轮和验证选出的最佳轮次。若需关闭早停，给所有相关组使用同一规则，并标注这是共同完整日程的补充实验。各项使用同一初始化与图像批次，先一颗种子出表，资源允许再优先复跑关键差异。

## 审稿人 2

感谢审稿人指出，需要把表示空间的分析与具体检索错误联系起来。

### 1. 属性与关系检索错误的直接证据（W1、Q1）

**回复。** 表 3 和表 5 的 WSD/BSD 描述了文本表示的平均几何特征。具体哪些属性或关系仍可区分，需要直接的检索证据。我们会收窄原稿对语义失真和语义坍缩的论断，并用同一批图像和经过人工核验的属性、关系对照，比较 CLIP-FT、采用同一集合训练目标的 CLIP 和 MCDisp-Align。我们会同时报告完整模型纠正了多少基线排序错误，以及出现了多少新的排序错误。对于概率模型，还会检查图像表示覆盖了哪些经核验的描述事实，以及困难负例与这些事实的分离程度。

**证据示例与获取方式（E5，与 E6 共用样本）**

表 5 已报告 MSCOCO 的 WSD 在 w/o alignment 与 Full 间为 0.402→0.408。这是整体几何变化的对照。用下面的事实级实验检验较小的总体变化是否掩盖了具体属性或关系错误。

1. 从 MSCOCO 未见图中先抽样、后看模型结果。每张图标注清楚可见的对象、属性、动作或关系，构造单事实替换对。以虚构示例说明标注格式。同一图上「a person holding a red umbrella」与「a person holding a blue umbrella」构成属性对。关系对可交换 `in front of` 与 `behind`，保留其他词。两位标注者独立核验正句可见、负句确实与图像冲突，排除颜色不清、视角歧义和真假均成立的样本。
2. 对每个模型记录共同余弦分差 $s(i,c^+)-s(i,c^-)$、正句胜率、硬负例排在正例前的比例，以及配对错→对和对→错次数。若能找到属性或关系相反的真实图，也做逐句 T2I 的成对排序。可加一个文本三元组诊断，以真实描述为 anchor、保留同一事实的释义为正、单事实替换为负，报告语义区分的分差。对照 zero-shot CLIP、原版 CLIP-FT、Set-CLIP 与 Full，避免从单一 checkpoint 的错例反推训练造成了坍缩。
3. 复用 E6 的正负覆盖差。概率模型内先比较 Full、w/o align、diag-only。若 PCME++ 或 $K_{\rm train}=1$ 的 MCDisp 权重合格，则加入单句概率对照，注明训练配方变化。CLIP 无原生协方差，使用共同余弦排名。出一张真实案例图，标出具体事实和最难负例，并列 CLIP-FT/Set-CLIP/Full 的候选排名。若做二维投影，将其作为案例说明，结论依据完整空间的配对统计。

**示例表头。** `图像ID | 属性/关系 | 正句与硬负例 | CLIP-FT分差 | Set-CLIP分差 | Full分差 | 错→对/对→错`。以图像为单位做配对自助重采样，分别报告属性、关系样本数和置信区间。若只见 WSD 增大或个别错例改善，正文将结论收窄至观察到的排名或几何变化。

### 2. 描述互补性与困难负例（W2、Q2）

**回复。** 五条描述可能反复表达同一组视觉事实。我们会在现有 MSCOCO 测试图像中标注一个子集，按每组描述提供的可核实视觉事实划分互补性。我们将分档报告检索结果、真实描述覆盖率与困难负例的分离程度，检验模型增益是否主要出现在描述互补性较高的图像上。

**证据示例与获取方式（E6）**

1. 在 E5 的 MSCOCO 图像池为各 caption 标注事实集合 $F_k$，同义表达归一到同一事实。用 $1-\operatorname{mean}_{a<b}\operatorname{Jaccard}(F_a,F_b)$ 衡量互补性，剔除无可靠事实的描述。固定五句，预先划分低/中/高三档，尽量平衡句长、对象数和场景类型。预算紧可从每档约 30 张起，保留有效样本数和分档阈值。分组过程与被评估模型的得分无关。
2. 每图从相似场景选候选硬负例，保持主对象或背景相同、至少一个可核验事实冲突。候选可用一个固定检索器预筛，再由标注者确认，所有模型共享同一组硬负例。按档统计查询结果，共用原候选图库。主列为集合 I2T/T2I R@1 及 Full−CLIP-FT、Full−Set-CLIP 差值。可附独立 caption 的最差正例排名或 AllHit@5，检验集合正确是否伴随各语义均能被检索。
3. 对概率模型，以式 (6) 的 $d^2/D$ 记录最远真描述和最近硬负描述，报告「最近负例距离 − 最远正例距离」、正例覆盖率、硬负误覆盖率及 $\operatorname{tr}(\Sigma^v)$。用原阈值 1 判断覆盖项是否激活。跨概率模型比较覆盖率时，可在独立 dev 上校准到同一硬负误覆盖率。共同余弦分差可用于包含 CLIP 的全模型比较。若正例与硬负同时被覆盖，分别报告两个比例，避免把整体方差变大解释为选择性覆盖。

| 互补性分组 | 图像数 | 集合 I2T/T2I R@1 | Full−Set-CLIP | 正/负覆盖率 | 最近负−最远正 |
| --- | --- | --- | --- | --- | --- |
| 低 / 中 / 高（每档一行） | 待标注 | 待测 | 待测及区间 | 待测 | 待测 |

可复用仓库 Mahalanobis 与独立排名逻辑。分档和硬负清单需新增，使用同一份事实标注。 [覆盖计算](D:/code/DistributionAlignment/scripts/ablation_study/h2_coverage.py) · [逐 caption 排名](D:/code/DistributionAlignment/scripts/eval_allhit.py)

### 3. 各对齐项的贡献（Q3）

**回复。** 我们会在回复审稿人 1 所述的 MSCOCO 描述集合检索表中，报告 KL、覆盖、子空间对齐和方差正则的逐项消融。低秩加对角与纯对角两个协方差变体会使用相同的检索、真实描述覆盖和困难负例指标。完整模型与去除方向监督的模型之间的差值用于衡量方向监督。去除方向监督后保留低秩分支的模型，与纯对角模型之间的差值用于评估低秩分支在覆盖项中的作用。

**证据示例与获取方式。** 复用 E4 全部逐项训练。证据表增加 Full/no-direction/diag-only 的「真描述覆盖、硬负误覆盖、最近负−最远正」三列，复用 E6 的同一批图和负例。方向项如未生效或召回差值与零相容，则报告实际轮次、差值及不确定性。`no_direction` 与 `diag_only` 的比较要确认覆盖项有活跃样本。

## 审稿人 3

感谢审稿人询问基线微调目标和描述数量的作用。

### 1. CLIP-FT 是否使用相同的描述集合训练目标？

**回复。** 投稿稿件中的 **CLIP-FT 使用 CLIP 原论文的图文成对对比损失，ProLIP-FT 使用 ProLIP 原论文的成对概率训练目标**。两组基线每次更新都为每张图采样一条描述。MCDisp-Align 联合使用每图的五条描述。我们会在第 3.1 节明确说明这一区别。补充实验将比较使用相同五条描述的 CLIP 成对训练和集合训练，以衡量分组监督的作用。随后在相同的集合匹配目标下，比较仅保留匹配损失和加入分布对齐损失的 MCDisp-Align，以估计分布对齐的额外贡献。

**证据示例与获取方式。** 复用 E2 的原版 CLIP-FT、Set-CLIP 和同架构的 alignment 开关对照。可给一行简表显示每组「每图每步 caption 数 / 损失 / 集合 R@1 / 标准 R@1」，把主要差值写成「pair→set 提升［待测］pp，匹配目标下增加 $\mathcal L_{\rm align}$ 提升［待测］pp」。符号和叙述随实测调整。训练脚本和两篇原论文的引用见 R1 W2。

### 2. 对训练描述数量的敏感性

**回复。** 原稿实验使用每图五条参考描述。我们将在 MSCOCO 上分别用每图 1、3、5 条描述训练 MCDisp-Align，三组使用同一测试图库与五句评估设置，以测量训练描述数量对检索的影响。每图只有一条描述时，描述中心之间的协方差为零，子空间项也没有可对齐的方向。我们会报告三个设置下的实测变化和不确定性。

**证据示例与获取方式（E7）**

| 训练 caption 数 | 测试 caption 数 | 集合 I2T R@1 | 集合 T2I R@1 | 集合 mR |
| --- | --- | --- | --- | --- |
| 1 | 固定 5 | 待测 | 待测 | 待测 |
| 3 | 固定 5 | 待测 | 待测 | 待测 |
| 5 | 固定 5 | 待测 | 待测 | 待测 |

1. 固定每图五条真实描述的随机顺序和 ID，按嵌套前缀取 1/3/5 条，各组用同样的图像、初始化、每批图像数、优化更新数和日程。记录文本曝光数随 $K$ 的变化。先复用满足同一配方的 E4 Full，优先跑 3 对 5，再补 1。所有组用独立固定的五句验证及测试清单。不要使用自动继承训练 $K$ 的训练器内置的最终评估。
2. 核对 $r_0=\min(r,K-1,D)$。$K=1$ 跳过方向项，$K=3$ 的有效秩至多 2。主实验报告这一随 $K$ 发生的变化。如果希望区分 caption 数量和可用方向数，可在 $K=3,5$ 用相同 $r=2$ 补做一对控制，复用 E4 的 $K=5,r=2$。
3. 若更高 $K$ 有稳定增益，可写「在 MSCOCO 的 1–5 条范围内，更多真实 caption 有益」，并给配对差值。持平或非单调时直接报告结果。如果还需回答**推理**时多给查询描述是否有效，可固定同一 Full checkpoint，只在 T2I 查询端下采样 1/3/5 条并保持图像库不变，单列该结果。 [现有 caption 清单加载](D:/code/DistributionAlignment/data/manifest_caption_dataset.py) · [训练器评估入口](D:/code/DistributionAlignment/utils/mcdisp_align_trainer.py:1028)

## 开跑与回填

1. **固定评估口径并核对旧结果。** 保存 train/dev/test 图像 ID、五条 caption ID、随机种子、有效样本数及 checkpoint 配置。按图像 ID 检查三个划分互斥。当前 COCO 默认评测路径可能与训练注释路径相同，需显式提供未参与训练或选模的清单或 test-only 注释，并核对现有评测代码读取的实际 ID。能确认原稿图库时优先复用。若采用新图库，所有行同场复评并注明，避免把原表数字和新协议结果放在同一比较列。 [评估数据入口](D:/code/DistributionAlignment/utils/eval_common.py) · [训练数据入口](D:/code/DistributionAlignment/utils/dataset_factory.py)
2. **先拿无需训练的证据。** 核对原权重和早停轮次，做 E1 的梯度与 coverage 诊断、E3 已有权重的复评、E5/E6 人工核验与检索分析。其后优先训练 E2 的 Set-CLIP 和 E4 的四项损失消融，复用 Full 与 w/o。补 diag-only/no-sg、同骨干概率基线、ProLIP-set、rank 和 caption 数对照时沿用同一清单。
3. **只回填已核验结果。** 标准检索用 $N$ 张图和 $5N$ 条独立描述，I2T 以任一参考句命中为正，T2I 每句独立查询。集合检索以 $N$ 个预定义集合与 $N$ 张图匹配。每张结果表注明图库大小、打分、实际种子数、选模轮次，并用百分点和配对区间报告差值。待完成的实验用将来时。核验结果后再填写数值和差值。
