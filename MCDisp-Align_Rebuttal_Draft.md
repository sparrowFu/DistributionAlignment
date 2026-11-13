# MCDisp-Align rebuttal 初稿

英文段落参照 ICLR 模板，按审稿意见逐条回应。每条下方附中文证据示例和实验步骤，供作者组使用。R1–R3 对应 [Review.md](D:/code/DistributionAlignment/Review.md) 中的三组意见。§、式号和表号均指向 [投稿稿件](D:/code/DistributionAlignment/paper.pdf)。表中 **待测** 是填数位置。新增训练和语义诊断集中在 MSCOCO 图文检索。Flickr30K 只考虑用已有同数据集权重补一次标准检索评估。 [ICLR 回复模板](C:/Users/sparr/Desktop/rebuttal/rebuttal_iclr/4XWw.md)

## Reviewer 1

We thank the reviewer for examining the gradient paths, the retrieval protocols, and the baseline comparisons.

### **1. Gradient paths and covariance parameterization (W1)**

**Response.** The standard retrieval score uses the cosine similarity of the distribution centers. Equation (4) uses the image base variance, and the diagonal KL in Eq. (6) uses the image base Gaussian. Consequently, these three computations have no direct dependence on $U_i$. During training, $U_i$ receives a gradient from the full-covariance Mahalanobis coverage term when the hinge is active, and from the subspace term when its weight is nonzero. The QR-based subspace term controls orientation and is invariant to a positive rescaling of $U_i$. Its coefficient was chosen to balance the scales of the loss terms. The magnitude and timing of the resulting gradients require an empirical check, especially because Appendix A enables the direction term only in epochs 9–10 and allows early stopping.

Equation (2) correctly adds within-caption variance and between-caption variance by the law of total covariance. We agree that Eq. (6) can still overestimate the image's **total marginal variance**. It fits the image base variance to the caption-set marginal variance while $U_iU_i^\top$ adds another marginal contribution along the learned directions. We will clarify this distinction and measure the mismatch. We will also report how often $d^2/D>1$ on real batches. Increasing $\beta$ can only amplify the gradients of active hinges.

The stop-gradient blocks direct updates to caption targets from KL, coverage, and direction alignment. The variance regularizer updates the text variance head, and the matching loss can update it after the variance-gradient warm-up described in Appendix A. The existence of these paths leaves open whether the learned text variances vary meaningfully across captions. We will test this against a fixed-$v_0$ control.

**证据示例与获取方式（E1）**

| 质疑 | 可展示的证据 | 怎么获得 |
| --- | --- | --- |
| $U_i$ 实际是否更新，$\gamma$ 是否太弱 | 按 epoch 报告有效权重、原始及加权损失、各项对 $U$ 和低秩头参数的梯度范数、最佳 checkpoint 的 epoch。示例字段为 `epoch, best, weighted_dir, grad_U_cover, grad_U_dir` | 先核对表 1/4 的原始权重和训练日志。对固定的真实 COCO batches 分项求梯度并区分「无计算路径」和「有路径、数值为 0」。在同一完整日程下用 E4 的 Full 与 no-direction 比较。需要检验权重量级时，只增加一个 $\gamma=0.05$ 对照，保持启动轮次不变。 |
| Coverage 是否激活 | `p50/p90/p99(d²/D), Pr(d²/D>1), β·Lcover, grad_U_cover` | 用原始坐标和 Woodbury 距离，在覆盖项启用前后抽相同图像计算。以真实日志的非零权重为准。若实际活跃率接近零，如实报告。改阈值属于另一个修订设定。仅在原 hinge 有信号时，才用 E4 的 $\beta=0$ 和可选 $\beta=1$ 检验权重。 |
| 总边际方差是否偏大 | $\operatorname{diag}(\Sigma^v)-\operatorname{diag}(\Sigma^C)$ 的中位数和分位数，以及 $\lVert U\rVert_F^2/\sum_d v_d^v$ | 从同一 checkpoint 导出逐图 $v^v,U,\Sigma^C$，分别比较 base 和 total 的边际偏差，并与 diagonal-only 对照。如果要验证修改公式，把 KL 的图像端替换为 $v^v+\operatorname{diag}(UU^\top)$，以单独的修订变体报告。 |
| $v^t$ 是否退化为常数 | $v^t$ 的分位数、同维度跨 caption 的方差、文本方差头的分项梯度，以及固定 $v^t$ 后的 I2T 排名变化 | 先在早期/后期各取固定 batch，分开记录 $\mathcal L_{\rm match}$ 与方差正则的梯度，确认运行没有额外启用 `sg_set_variance`。固定 checkpoint，将每条 caption 的 $v^t$ 设为 0.04 后重算 Eq. (2)，保留各 caption 均值及其 between-caption 方差。Eq. (4) 中该项可能改变集合 I2T 排名。在集合 T2I 中它是同一个查询对所有图像的公因子，标准中心余弦也不受影响。若需要评估训练贡献，再加固定 $v^t$ 的重训组。 |

**可直接引用的数学证据。** 记 $z=(\Sigma^v+\epsilon I)^{-1}(\mu^t-\mu^v)$，则 $\nabla_U d^2=-2zz^\top U$。活跃 caption 的 coverage 梯度还要乘 $\beta/(KD)$。hinge 内侧为零。方向项的未归一化值介于 $0$ 和 $2\gamma r_0$ 之间，原稿最终权重下上界为 $0.08$。这是损失值的上界，梯度量级仍须分项测量。

**已有的诊断线索。** 仓库一组 9 月 23 日的 COCO/Flickr 日志显示最佳权重在第 2 轮、训练在第 5 轮早停。这批运行没有进入方向监督阶段。该 COCO 运行第 5 轮的未加权 Cover+ 为 0.0002，说明该次运行的 hinge 平均值很小但非零。它们与投稿表格权重的对应关系待核对。如果表格所用权重也在第 9 轮前选出，回信应明确该 checkpoint 未接受方向监督，并把共同跑满日程的实验单独报告。 [旧运行审计](D:/code/DistributionAlignment/tmp/mcdisp_audit_20260923/audit_results.json) · [COCO 旧运行日志](D:/code/DistributionAlignment/logs/train_mcdisp_align.log:44) · [KL、覆盖与方向实现](D:/code/DistributionAlignment/losses/mcdisp_align_losses_kl.py)

### **2. Native baseline objectives and matched caption-set training (W2)**

**Response.** We chose the fine-tuning objectives to follow the respective original methods. **CLIP-FT uses CLIP's symmetric image–text contrastive loss on individual pairs. ProLIP-FT uses ProLIP's pairwise probabilistic objective, with probabilistic contrastive matching, cross-modal inclusion, and variance regularization.** For both baselines, each update samples one reference caption per image from the available annotations. MCDisp-Align jointly uses five reference captions for each training image and aligns the image representation with their caption-set statistics. We will make this distinction explicit in §3.1. [CLIP, §2.3](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf) · [ProLIP, §3.2–3.3](https://proceedings.iclr.cc/paper_files/paper/2025/file/9032e5c9ec394ce768a2fa9bdc56af6c-Paper-Conference.pdf)

The methods draw their training annotations from the same benchmark datasets. Table 1 uses the same five human-provided captions per image for all methods and measures retrieval when caption groups are available. We agree that this protocol makes the grouped-input assumption and that the original comparison leaves a difference in training supervision. Table 2 separately evaluates independently encoded captions, with MSCOCO I2T/T2I R@1 of 69.6%/55.3% for MCDisp-Align and 67.6%/54.6% for CLIP-FT. To estimate how much of the caption-set gain comes from grouped training, we will add five-caption fine-tuning controls and evaluate them under both retrieval protocols.

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

仓库训练脚本能直接查到两组原版基线的 `random.choice`。本地 ProLIP 微调实现使用原论文的跨模态配对项和 VIB，措辞宜用 *follows the original pairwise objective*，避免宣称复现了原论文的完整预训练配方（包括额外的 masked-input 项）。Set 训练目标需新增接线，现有集合评测可复用。 [CLIP-FT 训练](D:/code/DistributionAlignment/scripts/train_clip_baseline.py:129) · [ProLIP-FT 训练](D:/code/DistributionAlignment/scripts/train_prolip.py:167) · [集合评测](D:/code/DistributionAlignment/scripts/eval_setlevel_all.py)

### **3. Standard retrieval, the matching-only ablation, and probabilistic baselines (W3)**

**Response.** Table 2 reports standard retrieval on MSCOCO only. Table 4 includes the Flickr30K ablation, so the missing comparison is the set of Flickr30K baselines under the standard protocol. On MSCOCO, removing $\mathcal L_{\rm align}$ reduces I2T/T2I R@1 from 69.6%/55.3% to 60.8%/48.1%, below the 67.6%/54.6% of CLIP-FT. This establishes a deficit for the matching-only variant in our architecture. Its comparison with CLIP-FT also changes the heads and the training objective. The matched set controls in W2 and the component ablations in W4 will help distinguish these effects.

We agree that the ViT-H/14 ProLIP result mixes differences in backbone and pretraining with differences in the alignment method. We will retain its backbone label and add comparisons with PCME++ and ProbVLM initialized from CLIP ViT-L/14 on MSCOCO. We will use the same test images and captions and state each method's trainable parameters and training recipe. For Flickr30K, the supplementary table will use checkpoints fine-tuned on Flickr30K where available.

**证据示例与获取方式（E3）**

- **MSCOCO 标准检索补表。** 原稿表 2/4 的六个 Recall 作历史参考。在同一候选库上评估 CLIP-FT、Full、w/o alignment、PCME++、ProbVLM，主列统一中心余弦，附列可给各方法原生分数，例如 PCME++ 的 CSD。每行写骨干、初始化、冻结范围、更新预算和 checkpoint 来源。ProbVLM 使用冻结 ViT-L/14 加适配器。PCME++ 使用 ViT-L/14 初始化和自己的训练配方。若加入本地 ProLIP-L/14，应标为「CLIP 初始化的 ProLIP 目标微调」，注明缺少原版概率预训练并先验证初始化及训练是否正常。已有的冒烟或中途中断权重不填入主结果表。 [PCME++ 原论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ad9d6ab10446114cf5482d5e1f971a84-Abstract-Conference.html) · [ProbVLM 原论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Upadhyay_ProbVLM_Probabilistic_Adapter_for_Frozen_Vison-Language_Models_ICCV_2023_paper.pdf) · [本地基线配方与历史状态](D:/code/DistributionAlignment/experiments.md)
- **Flickr30K 缺失对照。** 表 4 已有 Full 76.9%/64.1%、w/o 69.1%/59.7%（标准 I2T/T2I R@1）。核对原表划分、checkpoint 和打分。先找现有 Flickr30K CLIP-FT、MCDisp-Align 权重，再找 ProLIP-FT 等已有模型，在同一 Flickr30K 测试图库补六项 Recall，按实测报告。`scripts/eval_flickr30k.py` 默认用于跨数据集迁移。可改用 `scripts/eval_full_metrics.py --dataset flickr --model … --checkpoint … --tag …`，取普通余弦列。需要核对 ProLIP 的加载配置。只找到 COCO 权重的行只能写「COCO→Flickr30K 迁移」。 [标准评测](D:/code/DistributionAlignment/scripts/eval_full_metrics.py) · [跨数据集评测](D:/code/DistributionAlignment/scripts/eval_flickr30k.py)

### **4. Individual loss terms and caption-set retrieval ablations (W4)**

**Response.** Table 4 removes the entire alignment objective and reports standard retrieval. We will add an MSCOCO caption-set ablation with separate removals of diagonal KL, caption-center coverage, subspace alignment, and variance regularization. We will also compare low-rank-plus-diagonal and diagonal-only image covariances, remove the stop-gradient on the three caption targets, and vary the low rank. The no-direction condition retains the low-rank branch, which allows us to test the effect of direction supervision separately from the branch itself. We will use the gradient and activation measurements in W1 to interpret small or null effects.

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
| Rank sensitivity | 同 Full，单独跑 $r=2$，有余量再补 $r=1$ | 待测 | 待测 | 待测 |

已有八组可通过 `python scripts/run_ablation.py --config <组名> --dataset coco --loss kl --no-freeze-clip` 进入共享训练器。命令入口已存在，需先接上固定且互斥的划分、共同的验证选模指标与输出路径。当前 runner 默认冻结骨干，内置 summary 采用自己的 scorer 与抽样口径。表中应另用同一集合 evaluator 重评各组 checkpoint，确认 `diag_only` 按其实际 rank 加载并保留所有原稿 Recall@1/5/10。mR 为双向六个 Recall 的平均值。用相同 checkpoint 可低成本补标准余弦指标，与表 4 桥接。 [消融配置](D:/code/DistributionAlignment/config.py:561) · [消融入口](D:/code/DistributionAlignment/scripts/run_ablation.py)

Full↔no-direction 检验方向监督，no-direction↔diag-only 检验无方向监督时低秩分支在覆盖中的贡献，Full↔diag-only 检验整条低秩设计。做方向项机制对照时，两组按相同预算跑过第 9–10 轮，报告末轮和验证选出的最佳轮次。若需关闭早停，给所有相关组使用同一规则，并标注这是共同完整日程的补充实验。各项使用同一初始化与图像批次，先一颗种子出表，资源允许再优先复跑关键差异。

## Reviewer 2

We thank the reviewer for asking us to connect the representation analysis to concrete retrieval errors.

### **1. Direct evidence of attribute and relation errors (W1, Q1)**

**Response.** The WSD/BSD results in Tables 3 and 5 summarize average caption geometry. They cannot establish which attributes or relations remain distinguishable. We agree that the manuscript should qualify its claims about semantic distortion and collapse. We will evaluate verified attribute and relation contrasts on the same images for CLIP-FT, matched set-trained CLIP, and MCDisp-Align. We will report when the full model corrects a baseline ranking error and when the reverse occurs. We will also examine whether the probabilistic image representations cover multiple verified caption meanings while keeping hard negatives separated.

**证据示例与获取方式（E5，与 E6 共用样本）**

表 5 已报告 MSCOCO 的 WSD 在 w/o alignment 与 Full 间为 0.402→0.408。这是整体几何变化的对照。用下面的事实级实验检验较小的总体变化是否掩盖了具体属性或关系错误。

1. 从 MSCOCO 未见图中先抽样、后看模型结果。每张图标注清楚可见的对象、属性、动作或关系，构造单事实替换对。以虚构示例说明标注格式。同一图上「a person holding a red umbrella」与「a person holding a blue umbrella」构成属性对。关系对可交换 `in front of` 与 `behind`，保留其他词。两位标注者独立核验正句可见、负句确实与图像冲突，排除颜色不清、视角歧义和真假均成立的样本。
2. 对每个模型记录共同余弦分差 $s(i,c^+)-s(i,c^-)$、正句胜率、硬负例排在正例前的比例，以及配对错→对和对→错次数。若能找到属性或关系相反的真实图，也做逐句 T2I 的成对排序。可加一个文本三元组诊断，以真实描述为 anchor、保留同一事实的释义为正、单事实替换为负，报告语义区分的分差。对照 zero-shot CLIP、原版 CLIP-FT、Set-CLIP 与 Full，避免从单一 checkpoint 的错例反推训练造成了坍缩。
3. 复用 E6 的正负覆盖差。概率模型内先比较 Full、w/o align、diag-only。若 PCME++ 或 $K_{\rm train}=1$ 的 MCDisp 权重合格，则加入单句概率对照，注明训练配方变化。CLIP 无原生协方差，使用共同余弦排名。出一张真实案例图，标出具体事实和最难负例，并列 CLIP-FT/Set-CLIP/Full 的候选排名。若做二维投影，将其作为案例说明，结论依据完整空间的配对统计。

**示例表头。** `图像ID | 属性/关系 | 正句与硬负例 | CLIP-FT分差 | Set-CLIP分差 | Full分差 | 错→对/对→错`。按图像聚类做配对 bootstrap，分别报告属性、关系样本数和置信区间。若只见 WSD 增大或个别错例改善，正文将结论收窄至观察到的排名或几何变化。

### **2. Complementary captions and hard negatives (W2, Q2)**

**Response.** Five captions can express the same visual facts. We will therefore evaluate an annotated MSCOCO subset in which we distinguish repeated descriptions from sets that contribute complementary, visually verified facts. The evaluation will stratify retrieval by caption diversity and compare positive-caption coverage with hard-negative separation. This makes it possible to see whether any gain is concentrated in the cases that motivate the method.

**证据示例与获取方式（E6）**

1. 在 E5 的 MSCOCO 图像池为各 caption 标注事实集合 $F_k$，同义表达归一到同一事实。用 $1-\operatorname{mean}_{a<b}\operatorname{Jaccard}(F_a,F_b)$ 衡量互补性，剔除无可靠事实的描述。固定五句，预先划分低/中/高三档，尽量平衡句长、对象数和场景类型。预算紧可从每档约 30 张起，保留有效样本数和分档阈值。分组过程与被评估模型的得分无关。
2. 每图从相似场景选候选硬负例，保持主对象或背景相同、至少一个可核验事实冲突。候选可用一个固定检索器预筛，再由标注者确认，所有模型共享同一组硬负例。按档统计查询结果，共用原候选图库。主列为集合 I2T/T2I R@1 及 Full−CLIP-FT、Full−Set-CLIP 差值。可附独立 caption 的最差正例排名或 AllHit@5，检验集合正确是否伴随各语义均能被检索。
3. 对概率模型，以式 (6) 的 $d^2/D$ 记录最远真描述和最近硬负描述，报告「最近负例距离 − 最远正例距离」、正例覆盖率、硬负误覆盖率及 $\operatorname{tr}(\Sigma^v)$。用原阈值 1 回答 hinge 行为。跨概率模型比较覆盖率时，可在独立 dev 上校准到同一硬负误覆盖率。共同余弦分差可用于包含 CLIP 的全模型比较。若正例与硬负同时被覆盖，分别报告两个比例，避免把整体方差变大解释为选择性覆盖。

| 互补性分组 | 图像数 | 集合 I2T/T2I R@1 | Full−Set-CLIP | 正/负覆盖率 | 最近负−最远正 |
| --- | --- | --- | --- | --- | --- |
| 低 / 中 / 高（每档一行） | 待标注 | 待测 | 待测及区间 | 待测 | 待测 |

可复用仓库 Mahalanobis 与独立排名逻辑。分档和硬负清单需新增，使用同一份事实标注。 [覆盖计算](D:/code/DistributionAlignment/scripts/ablation_study/h2_coverage.py) · [逐 caption 排名](D:/code/DistributionAlignment/scripts/eval_allhit.py)

### **3. Which alignment components matter? (Q3)**

**Response.** We will report the individual KL, coverage, subspace, and variance-regularization ablations in the same MSCOCO caption-set table described in our response to Reviewer 1. We will also compare the low-rank-plus-diagonal and diagonal-only variants using the same retrieval and hard-negative coverage measures. The comparison between the full model and no-direction isolates direction supervision. The comparison between no-direction and diagonal-only tests the remaining effect of the low-rank branch.

**证据示例与获取方式。** 复用 E4 全部逐项训练。证据表增加 Full/no-direction/diag-only 的「真描述覆盖、硬负误覆盖、最近负−最远正」三列，复用 E6 的同一批图和负例。方向项如未生效或召回差值与零相容，则报告实际轮次、差值及不确定性。`no_direction` 与 `diag_only` 的比较要确认覆盖项有活跃样本。

## Reviewer 3

We thank the reviewer for asking us to clarify the fine-tuning objectives and the role of caption count.

### **1. Does CLIP-FT use the same caption-set objective?**

**Response.** In the submitted experiments, CLIP-FT follows the original CLIP pairwise objective, a symmetric contrastive loss over image–caption pairs. ProLIP-FT likewise follows the pairwise probabilistic objective from its original method. Each baseline samples one caption per image per update. MCDisp-Align jointly uses five captions in each training example. We will state these choices explicitly in §3.1 and include the five-caption CLIP control. Comparing that control with native CLIP-FT will estimate the effect of grouped supervision. Comparing matching-only and full MCDisp-Align under the shared set-matching objective will test the additional effect of distribution alignment.

**证据示例与获取方式。** 复用 E2 的原版 CLIP-FT、Set-CLIP 和同架构的 alignment 开关对照。可给一行简表显示每组「每图每步 caption 数 / 损失 / 集合 R@1 / 标准 R@1」，把主要差值写成「pair→set 提升［待测］pp，匹配目标下增加 $\mathcal L_{\rm align}$ 提升［待测］pp」。符号和叙述随实测调整。训练脚本和两篇原论文的引用见 R1 W2。

### **2. Sensitivity to the number of training captions**

**Response.** The submitted experiments use five references per image. We will train MCDisp-Align on MSCOCO with one, three, or five captions per image while keeping the test gallery fixed. This comparison will measure sensitivity to the number of captions used during training. At one caption, the between-caption covariance is zero and the subspace term has no target. We will report the observed trend across the tested values and its uncertainty.

**证据示例与获取方式（E7）**

| 训练 caption 数 | 测试 caption 数 | 集合 I2T R@1 | 集合 T2I R@1 | 集合 mR |
| --- | --- | --- | --- | --- |
| 1 | 固定 5 | 待测 | 待测 | 待测 |
| 3 | 固定 5 | 待测 | 待测 | 待测 |
| 5 | 固定 5 | 待测 | 待测 | 待测 |

1. 固定每图五条真实描述的随机顺序和 ID，按嵌套前缀取 1/3/5 条，各组用同样的图像、初始化、每批图像数、优化更新数和日程。记录文本曝光数随 $K$ 的变化。先复用满足同一配方的 E4 Full，优先跑 3 对 5，再补 1。所有组用独立固定的五句验证及测试清单。不要使用自动继承训练 $K$ 的内置 final eval。
2. 核对 $r_0=\min(r,K-1,D)$。$K=1$ 跳过方向项，$K=3$ 的有效秩至多 2。主实验报告这一随 $K$ 发生的变化。如果希望区分 caption 数量和可用方向数，可在 $K=3,5$ 用相同 $r=2$ 补做一对控制，复用 E4 的 $K=5,r=2$。
3. 若更高 $K$ 有稳定增益，可写「在 MSCOCO 的 1–5 条范围内，更多真实 caption 有益」，并给配对差值。持平或非单调时直接报告结果。如果还需回答**推理**时多给查询描述是否有效，可固定同一 Full checkpoint，只在 T2I 查询端下采样 1/3/5 条并保持图像库不变，单列该结果。 [现有 caption 清单加载](D:/code/DistributionAlignment/data/manifest_caption_dataset.py) · [训练器评估入口](D:/code/DistributionAlignment/utils/mcdisp_align_trainer.py:1028)

## 开跑与回填

1. **固定评估口径并核对旧结果。** 保存 train/dev/test 图像 ID、五条 caption ID、随机种子、有效样本数及 checkpoint 配置。按图像 ID 检查三个划分互斥。当前 COCO 默认评测路径可能与训练注释路径相同，需显式提供未参与训练或选模的清单或 test-only 注释，并核对现有评测代码读取的实际 ID。能确认原稿图库时优先复用。若采用新图库，所有行同场复评并注明，避免把原表数字和新协议结果放在同一比较列。 [评估数据入口](D:/code/DistributionAlignment/utils/eval_common.py) · [训练数据入口](D:/code/DistributionAlignment/utils/dataset_factory.py)
2. **先拿无需训练的证据。** 核对原权重和早停轮次，做 E1 的梯度与 coverage 诊断、E3 已有权重复评、E5/E6 人工核验与检索分析。其后优先训练 E2 的 Set-CLIP 和 E4 的四项损失消融，复用 Full 与 w/o。补 diag-only/no-sg、同骨干概率基线、ProLIP-set、rank 和 caption 数对照时沿用同一清单。
3. **只回填已核验结果。** 标准检索用 $N$ 张图和 $5N$ 条独立描述，I2T 以任一参考句命中为正，T2I 每句独立查询。集合检索以 $N$ 个预定义集合与 $N$ 张图匹配。每张结果表注明图库大小、打分、实际种子数、选模轮次，并用百分点和配对区间报告差值。待完成的实验用将来时。核验结果后再填写数值和差值。
