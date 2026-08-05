# 实验记录 · Experiments

> 本文件记录图文检索（MSCOCO）训练实验的结果与分析，重点是：
> UC-CL 旧基线（0.5622）与 MSDA 新方法（0.4648）的对比，以及 MSDA 首次实验中
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
| UC-CL（旧） | CL + Consist + Var | 30（早停于 25） | 4.72M | 钉死 ~0.102（硬下限 0.1） | **0.5622**（ep22） | 单调上升，无崩溃 |
| MSDA（首次） | set-NCE+mu+var+cover+cov+reg | 10 | 9.45M | 学习 ~0.12~0.22 | **0.4648**（ep8） | 第 9 轮 cov 激活即崩（→0.265） |

> 注：两者是**不同的损失族**，非小修改对比。MSDA 是 UC-CL 的架构升级（一般高斯 + 协方差项）。

---

## 3. UC-CL 基线轨迹（0.5622）

`logs/train_dist_align.log`（2026-06-26 起，30 轮）：

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

## 4. MSDA 首次实验与第 9 轮崩溃

`logs/train_dist_align.log`（2026-07-01 起，10 轮）。调度：ep1–2 Warmup、ep3–8 Main、ep9–10 Full。

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
| 1 | `scripts/train_dist_align.py` | 反向后加 `clip_grad_norm_(model.parameters(), MSDA_GRAD_CLIP_NORM=1.0)` |
| 2 | `config.py` | `MSDA_LAMBDA_COV`: 0.1 → **0.01**；消融块 7 处同步 |
| 3 | `scripts/train_dist_align.py` `stage_multipliers` | Full 阶段 `cov` **线性 ramp** 0→1 |
| 4 | `losses/dist_align_losses.py` | `L_cov` 非有限值时 `torch.zeros_like` 置零 |

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
- [ ] `DIST_ALIGN_EPOCHS`: 10 → **30**（10 轮时 cov 阶段只有 2 轮，即便不崩也学不到东西）。
- [ ] 重配调度比例（按 30 轮）：warmup 0.1 / main 0.6 / full 0.3。
- [ ] cov head 单独设更低 LR（`create_optimizer` 增加第三个 param group）。
- [ ] 数据驱动初始化 `img_U`（用首批描述偏差 SVD 初始化 cov head）。

### P2 — 关键消融（先定位再决定是否上 cov）
- [ ] `diagonal_only`（cov_rank=0）跑 30 轮：完全绕过崩溃，看 "无 cov 的 MSDA" 天花板。
- [ ] `no_cov`（cov_rank=4 但 λ_cov=0）跑 30 轮：隔离 cov 的影响。
- [ ] 训练日志加 `grad_norm`（分 head）与 `img_U` 范数监控，便于下次定位。

---

## 8. 复现命令

```bash
# 环境：conda env CudaVersion128Fuxp（用 env python 直接调用）
PY=/home/xpfu/.conda/envs/CudaVersion128Fuxp/bin/python

# 训练（建议先 P0 + --epochs 30 验证止崩与上升趋势）
$PY scripts/train_dist_align.py --epochs 30
# 或：$PY main.py --task train_dist_align

# loss 自测
PYTHONPATH=. $PY losses/dist_align_losses.py

# 查看 ramp 调度
$PY -c "import sys; sys.path.insert(0,'scripts'); import config; from train_dist_align import stage_multipliers as s; [print(e+1, s(e,30,False)['cov']) for e in range(30)]"
```
