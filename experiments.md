# 实验记录 · Experiments

> 本文件记录图文检索（MSCOCO）训练实验的结果与分析，重点是：
> MCDisp_Align 对角模式（`r=0`，0.5622）与全协方差模式（首次，0.4648）的对比，以及全协方差模式首次实验中
> **第 9 轮 `L_cov` 激活即崩溃** 的根因排查与 P0 修复。
> 方法细节见 `methods.md`。

---

## 1. 实验设置（两版共有）

| 项 | 取值 |
|---|---|
| 数据 | MSCOCO Captions：106,459 训练 / 11,828 验证，每图 K=5 captions |
| 骨干 | CLIP ViT-L/14，**冻结** |
| 优化器 | Adam（CLIP lr 1e-6；MLP lr 5e-5），weight_decay 1e-4 |
| Batch size | 32 |
| Seed | 42 |
| 早停 | patience=3，按 **recall@1** 选最优 checkpoint |
| 评估 | 双向 Recall@1/5/10（验证集对角配对，`compute_recall_bidirectional`） |

---

## 2. 主结果对比

| 方法 | 损失族 | Epochs | 可训练参数 | σ²img 行为 | **Best R@1** | 备注 |
|---|---|---|---|---|---|---|
| MCDisp_Align（对角模式 r=0） | CL + Consist + Var | 30（早停于 25） | 4.72M | 钉死 ~0.102（硬下限 0.1） | **0.5622**（ep22） | 单调上升，无崩溃 |
| MCDisp_Align（全协方差 r=4，首次） | set-NCE+mu+var+cover+cov+reg | 10 | 9.45M | 学习 ~0.12~0.22 | **0.4648**（ep8） | 第 9 轮 cov 激活即崩（→0.265） |

> 注：两者是**不同的损失族**，非小修改对比。全协方差模式是对角模式的架构升级（一般高斯 + 协方差项）。

---

## 3. MCDisp_Align 对角模式轨迹（r=0，0.5622）

`logs/train_mcdisp_align.log`（2026-06-26 起，30 轮）：

| Epoch | Train CL | Val Loss | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|
| 1 | 0.3467 | 0.1561 | 0.372 | 0.649 | 0.760 |
| 5 | 0.0691 | 0.0952 | 0.509 | 0.778 | 0.864 |
| 10 | 0.0516 | 0.0852 | 0.542 | 0.805 | 0.884 |
| 13 | 0.0466 | 0.0812 | 0.554 | 0.817 | 0.890 |
| 16 | 0.0424 | 0.0809 | 0.556 | 0.818 | 0.893 |
| 19 | 0.0413 | 0.0811 | 0.557 | 0.818 | 0.890 |
| 22 | 0.0393 | 0.0806 | **0.562** | 0.822 | 0.894 |
| 25 | 0.0383 | 0.0802 | 0.562 | 0.824 | 0.897（早停） |

- σ²img 全程 ~0.1016（旧代码硬下限 0.1），σ²txt 从 0.15 缓慢增至 0.30。
- **Best recall@1 = 0.5622**（ep22 保存，ep25 触发早停）。

---

## 4. MCDisp_Align 首次实验与第 9 轮崩溃

`logs/train_mcdisp_align.log`（2026-07-01 起，10 轮）。调度：ep1–2 Warmup、ep3–8 Main、ep9–10 Full。

| Epoch | 阶段 | Train Loss | NCE | Var | Cover | Cov | σ²img | **R@1** |
|---|---|---|---|---|---|---|---|---|
| 1 | warmup | 0.5475 | 0.3618 | 0.0304 | 0.2279 | 0.0 | 0.189 | 0.345 |
| 2 | warmup | 0.2409 | 0.1368 | 0.0339 | 0.1796 | 0.0 | 0.224 | 0.400 |
| 3 | main | 0.2182 | 0.1066 | 0.0120 | 0.0029 | 0.0 | 0.140 | 0.421 |
| 4 | main | 0.1965 | 0.0939 | 0.0099 | 0.0005 | 0.0 | 0.124 | 0.435 |
| 5 | main | 0.1821 | 0.0841 | 0.0100 | 0.0002 | 0.0 | 0.121 | 0.450 |
| 6 | main | 0.1738 | 0.0789 | 0.0103 | 0.0001 | 0.0 | 0.119 | 0.457 |
| 7 | main | 0.1659 | 0.0734 | 0.0107 | 0.0000 | 0.0 | 0.117 | 0.463 |
| 8 | main | 0.1607 | 0.0696 | 0.0111 | 0.0000 | 0.0 | 0.116 | **0.465** ← best(0.4648) |
| 9 | full | 0.5184 | 0.1144 | 0.0093 | 0.0000 | **3.0003** | 0.116 | 0.265 ← 崩 |
| 10 | full | 0.4416 | 0.1286 | 0.0094 | 0.0000 | 2.1497 | 0.116 | 0.240 |

### 4.1 崩溃量级核算（ep9）

ep9 total loss = 0.5184 分解：

| 项 | 计算 | 贡献 |
|---|---|---|
| NCE | 1.0 × 0.1144 | 0.114 |
| **cov** | **0.1 × 3.0003** | **0.300（占 58%）** |
| var | 1.0 × 0.0093 | 0.009 |
| cover | 0.5 × 0.0000 | 0 |

→ `L_cov` 一激活就以 0.30 的贡献压过 NCE（0.11），主导整张计算图梯度。

### 4.2 关键证据

- **σ²img 在 ep9–10 崩溃期间稳定在 0.116** → `img_logvar_head` 未被污染。
- **NCE 从 0.0696（ep8）跳到 0.1144（ep9）**，且总 loss 中 mu+reg 部分从 ~0.08 跳到 ~0.365 → 污染定向指向 `img_mu_head`（检索均值）。
- 崩溃精确发生在 **cov stage 0→1 切换点**，且 ep9/ep10 连续恶化 → 可复现，非随机抖动。
- loss 自测中随机 `U` 上 `L_cov = 7.96`（近上限 8），印证 "cov head 冷启动 → 子空间近似正交 → L_cov 近上限"。

---

## 5. 根因（详见 `methods.md` §6）

1. `λ_cov=0.1` 对 `L_cov` 量级（~3，上限 8）过大 → 主导梯度。
2. `img_cov_head` 激活前几乎未训练 → 冷启动近上限 → 梯度极陡。
3. 无梯度裁剪 → 大梯度直接打飞参数。
4. `img_U` 经 `L_cover`（与 `img_mu` 共享）回流污染检索均值；叠加 `qr/eigh/solve` 反向可能产生 Inf/NaN。

---

## 6. P0 修复与验证（已完成）

> 历史记录：本段是当时实验段的临时修复（含 `λ_cov 0.1→0.01`）。**当前正式 `λ_cov` 已按 HTML 方法论回退为 `0.2`**，改由 grad-clip + 线性 ramp + NaN-guard 维持稳定，见 `methods.md` §6.3。下面的崩溃现象与根因分析仍然有效（`L_cov` 量级与 target 形式无关）。

### 6.1 改动清单

| # | 文件 | 改动 |
|---|---|---|
| 1 | `scripts/train_mcdisp_align.py` | 反向后加 `clip_grad_norm_(model.parameters(), MCDISP_ALIGN_GRAD_CLIP_NORM=1.0)` |
| 2 | `config.py` | `MCDISP_ALIGN_LAMBDA_COV`: 0.1 → **0.01**；消融块 7 处同步 |
| 3 | `scripts/train_mcdisp_align.py` `stage_multipliers` | Full 阶段 `cov` **线性 ramp** 0→1 |
| 4 | `losses/mcdisp_align_losses.py` | `L_cov` 非有限值时 `torch.zeros_like` 置零 |

### 6.2 验证证据（均实跑通过）

- **语法**：`python -m py_compile` 三个文件全部通过。
- **ramp 调度**实测：
  - 10 轮：ep9 `cov=0.50`、ep10 `cov=1.00`（旧版是 ep9 直接 0→1 硬切）。
  - 30 轮：ep25–30 `cov=0.17→0.33→0.50→0.67→0.83→1.00`，6 轮平滑爬坡。
- **loss 正常路径**：自带 self-test 通过；随机 `U` 上 `L_cov=7.96`，`grad img_U=0.008`（有限，cov 子图正常回传）。
- **NaN 保护**实测：`finite 3.0→3.0`、`nan→0.0`、`inf→0.0`。
  - 过程中**抓到并修复一个 bug**：初版用 `cov_loss.detach()*0.0`，但 IEEE-754 下 `nan*0=nan`，归零失败；改为 `torch.zeros_like(cov_loss)` 后实测能真正归零。

---

## 7. 待办（P1 / P2）

> P0 仅保证 **不再崩**。要在绝对值上追平/超过 0.5622，还需：

### P1 — 恢复/超越 0.5622
- [ ] `MCDISP_ALIGN_EPOCHS`: 10 → **30**（10 轮时 cov 阶段只有 2 轮，即便不崩也学不到东西）。
- [ ] 重配调度比例（按 30 轮）：warmup 0.1 / main 0.6 / full 0.3。
- [ ] cov head 单独设更低 LR（`create_optimizer` 增加第三个 param group）。
- [ ] 数据驱动初始化 `img_U`（用首批描述偏差 SVD 初始化 cov head）。

### P2 — 关键消融（先定位再决定是否上 cov）
- [ ] `diagonal_only`（cov_rank=0）跑 30 轮：完全绕过崩溃，看 "无 cov 的 MCDisp_Align" 天花板。
- [ ] `no_cov`（cov_rank=4 但 λ_cov=0）跑 30 轮：隔离 cov 的影响。
- [ ] 训练日志加 `grad_norm`（分 head）与 `img_U` 范数监控，便于下次定位。

---

## 8. 复现命令

```bash
# 环境：conda env CudaVersion128Fuxp（用 env python 直接调用）
PY=/home/xpfu/.conda/envs/CudaVersion128Fuxp/bin/python

# 训练（建议先 P0 + --epochs 30 验证止崩与上升趋势）
$PY scripts/train_mcdisp_align.py --epochs 30
# 或：$PY main.py --task train_mcdisp_align

# loss 自测
PYTHONPATH=. $PY losses/mcdisp_align_losses.py

# 查看 ramp 调度
$PY -c "import sys; sys.path.insert(0,'scripts'); import config; from train_mcdisp_align import stage_multipliers as s; [print(e+1, s(e,30,False)['cov']) for e in range(30)]"
```

---

## 9. B4 ProbVLM 基线（新增，待运行）

为 rebuttal 补充的对比基线：ProbVLM（Upadhyay et al., ICCV 2023，[官方代码](https://github.com/ExplainableML/ProbVLM)）。
本项目实现为冻结 CLIP ViT-L/14（D=768，官方论文用 ViT-B/32，D=512，为骨干对齐改接 ViT-L/14）之上
的两个 BayesCap MLP 适配器（图像端/文本端各一，hid=256、3 层、dropout 0.1，逐维广义高斯
(μ, 1/α, β)）。只训练适配器（约 800 万参数），损失为官方 TempComb（T1·L1 + T2·GenGauss NLL，
T1=1.0、T2=5e-2）+ 跨模态项（λ=1e-4），target 为冻结 CLIP 特征；Adam lr=1e-4、逐 epoch 余弦退火
（官方代码的调度器缩进 bug 已按意图修正，见 `scripts/train_probvlm.py` docstring）、100 epochs、
每 epoch ≤2000 batch、早停 patience 10。评测时 (1/α, β) 换算为高斯 logvar
（σ² = α²·Γ(3/β)/Γ(1/β)），接入与 ProLIP 相同的协议 A/B（cosine、CSD、mc_recall、mc_cos_recall）。

仅做 MSCOCO（COCO-only）：Flickr30K / VQA / 多种子聚合脚本暂不接入（`eval_flickr30k.py`、
`eval_vqa_retrieval.py`、`aggregate_eval_multiseed.py` 未改动，后续可按需扩展）。

```bash
PY=/home/xpfu/.conda/envs/CudaVersion128Fuxp/bin/python

# 单测（不需要 CLIP 权重/GPU）
$PY -m pytest tests/test_probvlm.py -v

# 训练 + 评测（冒烟已通过：模型前向、5-batch 训练、64 图评测、main.py 接线）
$PY main.py --task train_probvlm    # 可加 --no-early-stop 复现官方完整 100 epochs
$PY main.py --task eval_probvlm     # 5000 图种子子集，协议 A+B
```

环境备注：`PreTrainedModels/clip-vit-large-patch14` 已恢复为指向
`/mnt/disk1/xpfu/PreTrainedModels/clip-vit-large-patch14` 的符号链接（projection_dim=768 已验证）。
合理性校验：ProbVLM 是保身份适配器，训完 `mc_cos_recall@1` 应接近冻结 CLIP 零基线，差距过大说明适配器训练发散。

---

## 10. B5 PCME++ 基线（新增，待运行）

为 rebuttal 公平性对比新增：PCME++（Chun, ICLR 2024）从 **OpenAI CLIP ViT-L/14 初始化**自训（与
Ours/B1/B2/B4 同骨干同初始化），官方代码以 `pcmepp/` 包形式原样搬运（MIT；仅三处标记性修改：本地权重
加载覆盖、`pkg_resources`/`np.bool`/`torch.load` 的 2026 环境兼容），训练循环按项目脚本习惯重写。

**方法**：魔改 CLIP——骨干最后一层分叉为 mean 支路（预训练权重）与 uncertainty 支路（新初始化
transformer 层，n_unc_layers=1）；输出逐 token 未投影特征经 Linear(1024→1024)+GPO 池化得到
μ（l2 归一化）与 log σ²。损失为闭式概率距离 `d=‖μi−μj‖²+Σ(σi²+σj²)` 上的 BCE（可学习
shift/negative_scale=5）+ VIB(1e-4) + 0.1×伪正样本平滑（平滑项按当前 logits 重新标记不差于真样本的对）。

**配方（全部官方 configs/pcmepp.yaml + README ViT-L/14 路线）**：AdamP lr 5e-4、wd 1e-4、逐层 LR 衰减
0.7（公式硬编码 12——视觉第 12~23 层因子 >1，尾层 lr≈1.8e-4，为官方对 L/14 的实际行为，忠实保留）；
图像骨干 lr×0.01、文本骨干×0.1；criterion 标量 wd=0；batch 128（=官方 8 卡×16）；caption 级采样
（每图 5 条描述各成样本）；前 2 epoch 冻结图像骨干（unc 支路保持可训练）；epoch 2 批内线性 warmup
（乘在损失上）；epoch 15 起 lr×0.1；25 epochs；梯度裁剪 2.0；MSDA（mixup/cutmix α=2、elem 模式、
25% 元素参与）；图像 token 级 SizeAugment 0.2（模型内部）；文本 token 扰动 0.2（mask 10%/随机替换 2%
/删除 8%，随机词表 30522=BertTokenizer 词表大小，官方如此）。

**与官方实现的偏差清单**：
1. 精度默认 bf16 autocast（仅编码器；criterion/GPO 保持 fp32）——为单卡 batch 128 腾显存；`--precision 32-true` 可回到官方 fp32。
2. 无 torch.compile：官方 engine 调用 `torch.compile` 后丢弃返回值，实际并未编译，跳过即等价。
3. 无 SWA（官方 swa_start_epoch=16，非论文主表协议）。
4. 数据：项目 MSCOCO train2017 池（118,287 图 × 5 描述）vs 官方 COCO-2014 Karpathy；图像插值用 PIL BILINEAR 替代 cv2.INTER_LINEAR；验证集用每图 caption[0]（官方 dev split 5k 全描述）。
5. 非有限损失处理与官方一致（官方 `print('Killed due to NaN!!'); exit(0)`；本项目 log 后 raise，结果相同）；DataLoader worker 重播种（官方缺失，属修正）。
6. **精度：fp32（官方 32-true，2026-11-10 起为默认）**。曾默认 bf16 autocast（编码器）以省显存——这正是两次训练崩溃的
   根因（见下方根因调查）；为在单卡 80GB 上跑 fp32 batch 128，对两座 CLIP 塔的 38 个 attention block 启用
   `torch.utils.checkpoint`（逐 block、fp32 重算，数值上与不加点检完全一致；block 内纯 torch 无 RNG，重算精确），
   峰值显存 12.6 GiB（未加点检的 fp32 需 ~76 GiB 激活）。`--no-grad-checkpoint` 可关，`--precision bf16-mixed` 保留但不再推荐。
7. **epochs 5**（官方 25；训练时长预算），epoch 15 的 lr×0.1 降档因此永不触发。
8. **评估协议（2026-11-10 起）**：主指标改为官方论文协议——官方 `pcmepp/evaluation.py`（已原样搬运，仅
   eccv_caption 依赖加保护）的 `compute_csd_sims` + `eval_i2t_val/eval_t2i_val`：每图全部 5 条 caption、
   CSD 相似度、i2t（5 中任一命中）/ t2i（逐 caption）R@1/5/10 + medR。项目协议 A/B 保留用于跨基线对比。

**根因调查（2026-11-10，两次崩溃：25-epoch 跑 epoch 10、5-epoch 跑 epoch 3 batch 1428，均 sigma 支路
logsigma→+23、loss→1e7~1e14、梯度非有限）**：移植本身无 bug（结构/LR 分组/warmup/冻结/matched 矩阵逐行
比对官方 yaml+engine.py 一致，含 sigma_ln_init:null、unc 支路 lr 3.5e-6、ViT-L 尾层 0.7^-11、官方多卡
all-gather 后同为 128×128 匹配矩阵）；350 个训练 batch（含完整 MSDA/SizeAugment）扫描无天然灾难图片；
bf16 与 fp32 forward 差异 0.006。结论：std 支路（无界 exp(排序池化(无 LN 的 unc 输出))）本就运行在临界
状态，这是官方设计的固有属性；官方 fp32 下偶发大 loss 的梯度"巨大但有限"，被 clip 2.0 缩放后照样 step、
把 sigma 拉回（自愈），真 NaN 则 exit；bf16 backward 在该状态产生非有限梯度——无保护时 NaN 步进污染权重
（第一次崩溃）、加跳步保护后自愈梯度永不生效而死锁（第二次）。修复=回到官方 fp32+官方语义，以逐 block
梯度检查点解决显存。诊断脚本：tools/diagnose_pcmepp_std.py、tools/repro_pcmepp_explosion.py、
tools/smoke_fp32_ckpt.py。

**运行记录**：2026-11-09 10:51 首训（25 epochs，bf16）epoch 10 崩溃（epoch 8 权重备份
`pcmepp_coco_best_crashed25ep_run.pt`，val 0.0487）；2026-11-10 上午复跑（5 epochs，bf16）epoch 3 batch 1428
死锁（epoch 2 完整状态备份 `pcmepp_coco_last_epoch2_clean.pt`，val 0.0690）；2026-11-10 下午按官方 fp32
配方（偏差 6）从零重训。该 bf16 epoch-2 权重在官方评估协议下 i2t R@1=65.8 / t2i R@1=49.0（5k 子集，
供参照；论文 ViT-L/14 COCO 5k test i2t R@1≈74.7）。

**验证状态（2026-11-09）**：单测 21 个全过（含 CSD 手工公式、LLRD 分组含 >1 因子、Mixup 行为、扰动界、
转换映射）；权重转换自检双塔余弦=1.000000；冒烟训练（20 步）损失 2512→11.7、峰值显存 10.5 GiB
（冻结期，解冻后预计 30-45 GiB）；冒烟评测 64 图四族指标正常；main.py 接线通过。冒烟产物已清理。

```bash
PY=/home/xpfu/.conda/envs/CudaVersion128Fuxp/bin/python

# 权重转换（一次性，已生成 PreTrainedModels/clip-vit-large-patch14-openai.pt）
$PY tools/convert_hf_clip_to_openai.py --self-check

# 正式训练 + 评测（epochs=5，见偏差 7）
$PY main.py --task train_pcmepp --resume checkpoints/pcmepp_coco_last_epoch2_clean.pt  # 从 epoch 3 续
$PY main.py --task eval_pcmepp     # 5000 图种子子集，协议 A+B
```

环境备注：GPU 上另有常驻进程（~9.8 GiB），训练有算力争抢时实际耗时可能更长；正式运行前建议确认
B4 ProbVLM 训练是否已完成排队（两者串行使用同一张卡）。

---

## 11. B3 ProLIP 骨干变体（l14 主 / l16 辅，2026-11-09 新增）

ProLIP 官方无 ViT-L/14 权重（L/16↔L/14 patch 卷积核与位置网格不同，不可转换），按用户决定同时实现两个变体，
经 `--backbone {h14,l14,l16}` 选择（h14 为原默认，检查点名/结果路径不变）：

- **l14（主）**：`tools/build_prolip_vit_l14.py` 用本地 OpenAI CLIP ViT-L/14 自搭 ProLIP 检查点——骨干权重映射到
  ProLIP 布局（text_projection 用 HF 原方向，防方阵转置陷阱，已有回归测试），位置编码视觉 257→258（第 1 行插
  unc 位）、文本 77→79（末尾追加 2 行），unc 部分按 ProLIP 初始化方案随机（emb N(0,0.01)、proj N(0,width^-0.5)、
  bias −10），logit_scale=ln(10)/logit_bias=−10（ProLIP 自己的 CLIP 转换约定）。复用 CLIP 224 处理器与 BPE 分词器。
  **诚实标注**：无 1.28B 概率预训练，表中写 "ProLIP-objective fine-tune from CLIP init"。
- **l16（辅）**：官方 `SanghyukChun/ProLIP-ViT-L-16-FT-DC-1B-1_28B`（2.57GB，hf-mirror 下载，已校验字节数）。
  **调研发现是 SigLIP 架构**：256 分辨率、SigLIP 归一化（0.5/0.5）、sentencepiece 32100 词表（timm/ViT-B-16-SigLIP
  的 tokenizer.json 快速路径）、context 64、clean=canonicalize、无 quick_gelu。处理器用 SiglipImageProcessor 编程
  构造，分词器文件在 PreTrainedModels/prolip-siglip-tokenizer。
- 环境：恢复了丢失的 h14 三个符号链接（prolip/prolipProcessor/prolipTokenizer → /mnt/disk1）。

**初始化质量诊断（8 对真实 COCO，匹配对 vs 不匹配对余弦差距）**：h14 +0.288、l16 +0.278（健康）；
**l14 仅 +0.003**——视觉侧与 CLIP 余弦 0.98（转换正确），**文本侧严重失效**：ProLIP 文本塔强制双向注意力且从
padding 之后的追加 token 池化，而 CLIP 文本权重是因果掩码训练的（60+ 个 pad token 全参与注意力）。这不是转换
bug（同一映射已被 PCME++ 自检证到 1.000000），是架构-预训练固有失配。**l14 的文本塔需要靠 COCO 微调恢复，
官方 B3 配方（lr 1e-6、5 epochs）可能不够，若微调损失平台期高建议试 lr 1e-5**。l14 zero-shot 的 CSD/不确定性
指标无意义（随机头），仅 cosine 行有效（脚本有警告）。

冒烟（2026-11-09，CPU，GPU 被 B5 训练与外用户占满）：单测 12 个全过；l14 构建+自检通过；l14 训练冒烟
（3 批）损失 364→339、检查点 prolip_l14_coco_best.pt 命名正确；l16 zero-shot 32 图 mc_cos_recall@1=0.969
（链路全通）；main.py --backbone 透传正常。l16 训练冒烟待 GPU 空闲补（优化器状态约 10.3GB，当前放不下）。

```bash
PY=/home/xpfu/.conda/envs/CudaVersion128Fuxp/bin/python
$PY tools/build_prolip_vit_l14.py --self-check          # 已跑过，产物已生成
$PY main.py --task train_prolip --backbone l14           # 正式训练（排队等 B5 结束）
$PY main.py --task eval_prolip --backbone l14
$PY main.py --task train_prolip --backbone l16
$PY main.py --task eval_prolip --backbone l16
```

---

## 12. L_align 组件级消融（rebuttal，2026-11-09 配置完成）

**审稿意见**（3 位审稿人一致）：原 Table 4 只对比了完整模型 vs 移除整条 $\mathcal{L}_{\mathrm{align}}$，要求把
对齐目标拆分为 KL 对齐、caption 中心覆盖、子空间（方向）对齐、方差正则四个组件分别消融，并补充全协方差 vs
仅对角的对比，以说明"中心对齐与正则之外，建模 caption 变化方向是否必要"。按 rebuttal 时间预算仅用 MSCOCO。

**实现**：`config.ABLATION_CONFIGS` 由 4 组扩为 6 组（`no_reg`：λ_reg=0；`diag_only`：cov_rank=0 且 λ_cov=0），
全部走既有 `run_ablation.py` → `MCDispAlignTrainConfig` 单一代码路径（KL 目标、分阶段调度、早停、内置评测），零
训练代码改动——损失项全部权重门控，`_cov_loss` 在 img_U 为 None 时自动跳过，对角模式训练/评测全链路已验证。
对照逻辑：`no_direction`（保留低秩分支、只去方向监督）vs `diag_only`（连容量一起去）分别隔离"监督"与"容量"。

| 组 | λ_kl | λ_cover_pos | λ_cov | λ_reg | cov_rank |
|---|---|---|---|---|---|
| full_model | 1.0 | 0.5 | 0.2 | 0.01 | 4 |
| no_kl | **0** | 0.5 | 0.2 | 0.01 | 4 |
| no_cover | 1.0 | **0** | 0.2 | 0.01 | 4 |
| no_direction | 1.0 | 0.5 | **0** | 0.01 | 4（分支保留） |
| no_reg | 1.0 | 0.5 | 0.2 | **0** | 4 |
| diag_only | 1.0 | 0.5 | **0** | 0.01 | **0** |

```bash
PY=/home/xpfu/.conda/envs/CudaVersion128Fuxp/bin/python
$PY scripts/run_ablation.py --config all --dataset coco     # 6 组串行，汇总表直接填 rebuttal 表
```

排期备注：GPU 现被 B5 PCME++ 训练占用（~2-3 天），本消融排其后；每组 10 epochs 头部训练约 2-3 小时，
6 组共约 12-18 小时。协议与原 Table 4 一致（--eval-samples 500，种子 42）；结果落
`outputs/ablation/ablation_results_coco.json` 与 `outputs/ablation/<组名>/checkpoints/`。

**冒烟验证（2026-11-09 完成）**：`no_reg` 与 `diag_only` 各训 1 epoch（`--eval-samples 64` 接线验证）均正常——
检查点保存/加载（diag_only 为 heads-only 对角模式）、召回选模、结果 JSON、汇总表全通。注意实测单 epoch 约
75 分钟（GPU 与 B5 训练争抢时），正式 6 组 10 epochs 在无争抢时估 21-30 小时、有争抢时可能 3 天以上，建议排
B5 结束后跑。冒烟产物已清理。
