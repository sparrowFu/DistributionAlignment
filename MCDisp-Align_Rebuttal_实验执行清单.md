# MCDisp-Align 审稿补充实验执行清单

目前 ProbVLM、CLIP-L/14 和 L/16 三组的标准检索训练已完成，逐项消融正在进行。新增训练集中在 MSCOCO。R1–R3 对应 [Review.md](D:/code/DistributionAlignment/Review.md) 中的三组意见。

| 实验 | 需要重训吗 | 交付结果 |
| --- | --- | --- |
| 1. CLIP-L/14 五描述集合微调 | 新训 Set-FT，复用单句 CLIP 和 MCDisp 权重 | 集合训练对照表 |
| 2. 逐项消融的集合检索 | 复用在跑的消融权重，按配置查缺补漏 | 逐项消融表 |
| 3. 覆盖、低秩梯度和文本方差 | 用已有权重做前向和分项反向 | 机制诊断表 |
| 4. 描述互补性和困难负例 | 人工核验测试子集，复用权重 | 分档结果、错误变化和案例 |
| 5. 每图训练描述数量 | 新训每图 3 条、1 条两组，按条件复用 5 条组 | 描述数量敏感性表 |
| 6. 概率基线与 Flickr30K 标准检索 | 先评现有权重，再查 PCME++ 缺口 | MSCOCO 同骨干表与 Flickr30K 补表 |

开跑前先固定训练、验证和测试图像 ID，确认三者互斥，并保存每张测试图像与五条描述的对应关系。优先沿用能核验的原稿划分。仅设置 `--num-samples 5000` 会从默认注释路径抽样，需要另外确认抽到的是测试图像。所有模型按同一份候选库、正例映射和评分规则复评。 [评估数据加载](D:/code/DistributionAlignment/utils/eval_common.py:54)

**标准检索**是 \(N\) 张图对 \(5N\) 条独立描述。图搜文命中任一正确描述即可，文搜图逐条查询。**集合检索**是 \(N\) 张图对 \(N\) 个五句集合。两项均报告 I2T（图搜文）和 T2I（文搜图）的 R@1/5/10。

已有 MCDisp 和消融检查点可用下面的模板复评。将 `N` 换成实际测试图像数，配置中的图像根目录指向实际位置。测试注释文件只包含事先锁定的测试图像。

```text
python scripts/evaluate_mcdisp_align.py --dataset coco --checkpoint "路径/模型.pt" --captions-path "路径/test-only.parquet" --num-samples N --recall-at-k 1 5 10 --output-path "路径/模型结果.json"
```

输出中，`mc_cos_recall` 是标准检索的共同余弦结果，`cos_recall` 是集合检索的共同余弦结果，`mcdisp_align_recall` 是按论文式（4）打分的集合结果。标准检索的 CLIP、ProbVLM 和 PCME++ 可以用各自评估脚本传入同样的参数。CLIP 集合评估需要将同一个测试注释路径接入集合评估器。 [MCDisp 复评入口](D:/code/DistributionAlignment/scripts/evaluate_mcdisp_align.py:151) · [CLIP 标准检索](D:/code/DistributionAlignment/scripts/evaluate_clip_baseline.py) · [CLIP 集合检索](D:/code/DistributionAlignment/scripts/eval_clip_setlevel.py)

## 1. CLIP-L/14 五描述集合微调

**回答 R1 W2 和 R3。** 给 CLIP 五句集合监督后，MCDisp-Align 的检索优势还剩多少？

| 对照组 | 训练时怎么用描述 | 权重 |
| --- | --- | --- |
| 原版 CLIP-L/14 FT | 每图每步随机采一条，沿用 CLIP 原论文的双向成对对比损失 | 复用已训权重并核对配置 |
| CLIP-L/14 Set-FT | 每图五句分别编码，平均为集合向量，归一化后和图像向量做对比 | 从与原版相同的预训练权重开始新训 |
| MCDisp 去整个对齐项 | 每图五句，保留原稿集合匹配，关闭整个对齐损失 | 复用消融权重 |
| MCDisp Full | 每图五句，原稿集合匹配加分布对齐 | 复用同配方的完整权重 |

1. Set-FT 每批取 \(B\) 张图、每图五句。每图得到一个文本集合向量，和图像向量计算 \(B\times B\) 余弦相似度矩阵。图像和自身描述集合是对角线正例。对矩阵及其转置各算一次交叉熵，取平均。沿用原版 CLIP 的冻结范围、图像 batch、优化器和更新预算，统一验证选模口径。
2. 四组在同一五句图库做集合检索，主对照用共同的中心余弦评分。复用相同权重补跑标准检索。新建 Set-FT 训练入口前可先用一个 batch 验证相似度矩阵尺寸及正例映射。
3. 有第二个训练位时加 **Pair-5**。每步读与 Set-FT 相同的五句，按句子槽位分别算五个 \(B\times B\) 成对对比损失，平均后更新一次。Set-FT 与 Pair-5 的差值用于估计聚合监督的作用。原版单句与 Set-FT 的差值包含单步文本数量和训练目标的共同变化。

**结果表。** `模型 | 每图每步描述数 | 损失 | 集合 I2T/T2I R@1 | 标准 I2T/T2I R@1`。报告 Set-FT 相对原版 CLIP、Full 相对 Set-FT、Full 相对去对齐项的差值。Full 与 Set-FT 还存在预测头和原稿匹配打分差异。若要进一步控制训练评分，可以给两个 MCDisp 对照都用与 Set-FT 一样的集合余弦损失，再分别关闭和保留分布对齐。原版 ProLIP-FT 沿用自身论文的成对概率目标，有余量再加同初始化的 ProLIP Set-FT。 [CLIP 原版训练](D:/code/DistributionAlignment/scripts/train_clip_baseline.py:105) · [集合评估](D:/code/DistributionAlignment/scripts/eval_setlevel_all.py:165)

## 2. 逐项消融的集合检索

**回答 R1 W4 和 R2 Q3。** 哪个损失项对主任务有贡献？低秩分支和方向监督各起什么作用？

| 配置 | 相对完整模型的改动 |
| --- | --- |
| `full_model` | 完整基准，使用与各消融组相同的训练设置 |
| `no_align` | 关闭 KL、覆盖、方向和方差正则，保留集合匹配 |
| `no_kl` | 只关闭对角 KL |
| `no_cover` | 只关闭描述中心覆盖 |
| `no_direction` | 只关闭方向监督，保留 \(r=4\) 及完整协方差覆盖 |
| `no_reg` | 只关闭方差正则 |
| `diag_only` | 设 \(r=0\)，保留对角覆盖，跳过方向项 |
| `no_sg` | 取消三个对齐目标的停止梯度，保留匹配方差梯度日程 |

1. 核对当前各组的损失配置、骨干冻结范围、训练预算和最佳轮次。现有消融入口默认冻结骨干。若在跑各组都冻结，用相同冻结设置的 Full 作基准。
2. 逐组用上面的命令复评已有检查点，取 **Set-level** 的 `mcdisp_align_recall`。当前模型加载器会读检查点记录的秩并重建低秩头，纯对角组需确认加载日志显示 \(r=0\)。跨架构比较时用集合余弦列。 [检查点加载](D:/code/DistributionAlignment/models/mcdisp_align_model.py:555)
3. Full 对去方向组检验方向监督。去方向组对纯对角组检验低秩分支在覆盖项里的作用。若 Full 最佳轮次早于方向项启用，保留最佳权重结果，再用两组共同选取的启用后轮次做机制对照。若还有算力，补 \(r=2\) 与 \(r=4\) 比较，另列 \(r=0\) 纯对角端点。

**结果表。** `变体 | 最佳轮次 | 集合 I2T R@1/5/10 | 集合 T2I R@1/5/10 | 相对 Full 的 R@1 差值`。现有标准检索结果可作补表。 [消融配置](D:/code/DistributionAlignment/config.py:561) · [消融入口](D:/code/DistributionAlignment/scripts/run_ablation.py)

## 3. 覆盖、低秩梯度和文本方差诊断

**回答 R1 W1。** 检查哪些项在真实训练中生效，哪些梯度实际到达低秩或文本方差头。

| 检查项 | 操作 | 记录 |
| --- | --- | --- |
| 覆盖项是否激活 | 在覆盖启用前后，从相同批次导出每条真描述的 \(d^2/D\)，使用训练损失的坐标和协方差 | 中位数、90/99 分位数、超过阈值 1 的比例、加权损失和 \(U\) 的覆盖梯度 |
| 低秩方向监督是否生效 | 查各轮实际权重和最佳轮次。用启用前后的检查点，对覆盖项和方向项分别求梯度 | 实际启用轮次、各项对 \(U\) 和低秩预测头的梯度范数 |
| 图像总边际方差是否偏大 | 导出每图的基础方差、低秩项的边际增量及描述集合边际方差 | \(\operatorname{diag}(\Sigma^v)-\operatorname{diag}(\Sigma^C)\) 的分位数及低秩方差占比 |
| 文本方差是否接近常数 | 导出单条描述的方差和匹配、正则的分项梯度。固定检查点，只把每条描述的方差替换成 0.04，保留均值和描述间离散度，重算式（4） | 方差分位数、同维跨描述变化、集合 I2T 的 R@1 差值和首位候选翻转率 |

从现有日志和检查点开始。按原稿的 10 轮日程，第 1–2 轮预热，第 3–4 轮逐步开启 KL 和匹配损失的方差梯度，第 5 轮启用覆盖，第 9–10 轮启用方向项。实际轮次和权重以日志为准，最佳轮次从保存最佳模型的日志读取。固定少量真实 batch，用 `torch.autograd.grad` 单独计算各项梯度，无需更新参数。分别记录「无梯度路径」「阶段权重为零」和「项已启用但本批梯度为零」。如果覆盖激活率近零，报告实测比例，增大覆盖权重也只会放大已经激活的样本。固定文本方差后，式（4）的集合 T2I 中该方差因子对同一查询的所有候选相同，排名应保持一致。标准中心余弦也应一致，可用两项检查实现。 [KL 损失实现](D:/code/DistributionAlignment/losses/mcdisp_align_losses_kl.py) · [训练日程](D:/code/DistributionAlignment/utils/mcdisp_align_trainer.py:77)

## 4. 描述互补性与困难负例

**回答 R2 W1、W2。** 哪些属性或关系的检索错误被纠正？收益是否集中在描述互补性高的图像？

1. **选图。** 在 MSCOCO 测试集先抽样，之后再看模型结果。标注每条描述的对象、属性和关系，同义事实归并。按五句的事实重合度分低、中、高互补性，首轮可每档约 30 张，记录分档规则和样本数。五句重复同一对象和动作属于低互补。不同句分别补充对象、属性和关系属于较高互补。
2. **选难负例。** 用相似场景的真实图或描述，或只修改一句中的一个事实。例如将「红色雨伞」换成「蓝色雨伞」。核对图像，确认原句成立且修改句确实冲突。所有模型共用同一套核验样本。
3. **跑两组指标。** 分档计算原版 CLIP-L/14、Set-FT 和 MCDisp Full 的集合 R@1。对最小事实对照，比较真描述和难负例的共同余弦分差，记录正例胜率、错→对、对→错次数。用 Full、去覆盖和纯对角权重计算马氏距离，另外报告真描述覆盖率、难负误覆盖率及「最近负例距离减最远真描述距离」。若比较不同尺度的概率模型，可先在验证集校准到相同的难负误覆盖率。

**结果表。** `互补性分档 | 图像数 | 集合 I2T/T2I R@1 | Full−Set-FT | 正例胜率 | 真描述覆盖率 / 难负误覆盖率`。另挑真实案例展示决定排序的属性或关系和各模型排名。用整批统计判断趋势。 [覆盖计算](D:/code/DistributionAlignment/scripts/ablation_study/h2_coverage.py) · [独立描述排名](D:/code/DistributionAlignment/scripts/eval_allhit.py)

## 5. 每图训练描述数量

**回答 R3。** 每图训练可用描述从 1 条增至 3 条、5 条，集合检索怎样变化？

1. 训练清单保留至少五条有效描述的图像。给每张训练图固定一个随机描述顺序，三组分别取前 1、3、5 条。各组使用相同图像、初始化、每批图像数、更新步数及损失日程。先做 3 对 5，再补 1。已有 5 条 Full 的初始化、划分和日程吻合时可复用。
2. 验证和测试一律使用每图同样的五条描述，并单独复评每组检查点。训练器部分内置评估会继承训练描述数，需核对实际读入数量。
3. 每图 1 条时描述间方差为零，方向项跳过。3 条时有效方向数最多为 2。若要控制方向数，再比较同为 \(r=2\) 的 3 条和 5 条组。

**结果表。** `训练描述数 | 测试描述数（均为 5） | 集合 I2T/T2I R@1 | 相对 5 条组的差值`。如趋势持平或非单调，报告实测结果。训练入口可复用 `MCDispAlignTrainConfig` 的清单模式，统一设置 `loss_name="kl"`、`dev_num_captions=5`，仅按组改变 `manifest_num_captions`。现有 `full_k3` 和 `full_kall_r2` 默认是冻结骨干、\(r=2\) 的试验配置，且默认使用另一套损失。`mean_only_k1` 关闭了部分损失，需新增 Full 的单句配置。 [描述数量配置](D:/code/DistributionAlignment/scripts/ablation_study/experiments.py) · [训练器清单与描述数](D:/code/DistributionAlignment/utils/mcdisp_align_trainer.py:538)

## 6. 同骨干概率基线与 Flickr30K 标准检索

**回答 R1 W3。** 补齐同骨干概率基线与 Flickr30K 的标准检索对照。

- **MSCOCO。** 用同一独立描述图库评估已训 CLIP-L/14、ProbVLM、MCDisp Full 和去对齐项。确认 ProbVLM 骨干及初始化与 CLIP-L/14 一致后再列为同骨干比较。L/16 按实际训练配置登记模型名和初始化。PCME++ 如有同骨干的有效权重，直接补评。如果仍缺，保留为待补训练。主列统一中心余弦，可另列各方法原生评分，注明冻结范围和训练预算。 [ProbVLM 评估](D:/code/DistributionAlignment/scripts/evaluate_probvlm.py) · [PCME++ 评估](D:/code/DistributionAlignment/scripts/evaluate_pcmepp.py)
- **Flickr30K。** 找原版 CLIP-FT、ProLIP-FT、MCDisp Full 和去对齐项在 Flickr30K 微调的检查点，用同一 Flickr30K 测试集做一次标准检索。如果只有 COCO 权重，评估结果标为跨数据集迁移。 [标准检索评估](D:/code/DistributionAlignment/scripts/eval_full_metrics.py)

**结果表。** `模型 | 数据集 | 骨干/初始化 | 冻结范围 | 评分 | I2T R@1/5/10 | T2I R@1/5/10`。注明实际有效样本数、评估划分和检查点来源。
