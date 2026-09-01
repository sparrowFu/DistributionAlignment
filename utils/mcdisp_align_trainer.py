"""MCDisp_Align training orchestration: warmup ramp for the dispersion terms, grad-norm clipping, recall/loss best-checkpoint selection, opt-in early stopping, best+last checkpointing, and resume. Full runs and ablation variants share one code path, differing only in the loss weights / ``cov_rank`` / ``num_captions`` / ``dataset`` / checkpoint paths passed via :class:`MCDispAlignTrainConfig`. The pure helper functions are exported for direct unit testing."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import config
from data.caption_dataset import filter_none_collate
from losses.mcdisp_align_losses import MCDispAlignLoss
from models.mcdisp_align_model import MCDispAlignModel
from utils.dataset_factory import build_train_dataset
from utils.eval_common import build_eval_dataloader
from utils.lr_scheduler import apply_lr_for_epoch
from utils.logger import get_logger
from utils.retrieval import compute_multicaption_recall

logger = get_logger("mcdisp_align_trainer")


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable)
# ---------------------------------------------------------------------------

def create_optimizer(
    model: MCDispAlignModel,
    freeze_clip: bool,
    clip_lr: float,
    mlp_lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    """Create optimizer with different learning rates for CLIP and MLP/cov heads.

    The image covariance head gets its own param group with ``weight_decay=0``:
    its final layer starts at a small scale (near-diagonal Sigma at init), and
    under Adam an L2-only gradient normalizes to a constant-magnitude
    ``sign(w)`` step that would drive those 1e-2-scale init weights to EXACTLY
    zero within ~|w|/lr steps (observed: a dead U head at evaluation despite
    non-zero init). Exempting it from weight decay keeps U's learning purely
    loss-driven (the gaussian L_match reaches U from step 0; L_dir joins after
    the warmup ramp)."""
    head_params = (
        list(model.img_mu_head.parameters())
        + list(model.img_logvar_head.parameters())
        + list(model.text_mu_head.parameters())
        + list(model.text_logvar_head.parameters())
    )
    groups = [{"params": head_params, "lr": mlp_lr, "weight_decay": weight_decay}]
    if getattr(model, "cov_rank", 0) > 0:
        groups.append({"params": list(model.img_cov_head.parameters()),
                       "lr": mlp_lr, "weight_decay": 0.0})

    if freeze_clip:
        # Only train distribution + image covariance heads
        return optim.Adam(groups)

    # CLIP and heads with different learning rates (CLIP keeps weight decay)
    return optim.Adam(
        [{"params": model.clip_model.parameters(),
          "lr": clip_lr, "weight_decay": weight_decay}] + groups,
    )


def warmup_ramp(
    epoch: int, step: int, steps_per_epoch: int, total_epochs: int, warmup_frac: float
) -> float:
    """Linear 0 -> 1 ramp for the dispersion/containment terms (L_var /
    L_dir / L_cov) over the first ``warmup_frac`` of the total optimizer
    steps; 1.0 afterwards.

    The caption heads train from scratch on frozen CLIP features, so the
    dispersion statistics (s_t^2, S_t) and the caption-mean containment
    targets need a few steps to differentiate before they supervise the
    image variance, directions, and ellipsoid. L_match / L_mu / R_prior are
    always on. ``warmup_frac=0`` disables the ramp.
    """
    if warmup_frac <= 0:
        return 1.0
    total_steps = max(1, total_epochs * max(1, steps_per_epoch))
    warmup_steps = max(1, int(total_steps * warmup_frac))
    progress = epoch * steps_per_epoch + step
    return min(1.0, progress / warmup_steps)


_HEAD_NAMES = ("img_mu_head", "img_logvar_head", "img_cov_head",
               "text_mu_head", "text_logvar_head")


def head_grad_norms(model) -> Dict[str, float]:
    """Per-head total grad norm (sqrt of sum of squared param-grad norms).

    Call BEFORE ``clip_grad_norm_`` to see which head dominates the gradient at
    stage transitions. Heads that don't exist (e.g. ``img_cov_head`` when
    cov_rank=0) are skipped. Returns ``{<head>_grad_norm: float}``.
    """
    out: Dict[str, float] = {}
    for name in _HEAD_NAMES:
        head = getattr(model, name, None)
        if head is None:
            continue
        sq = None
        for p in head.parameters():
            if p.grad is not None:
                ps = p.grad.detach().pow(2).sum()
                sq = ps if sq is None else sq + ps
        out[f"{name}_grad_norm"] = float(torch.sqrt(sq)) if sq is not None else 0.0
    return out


# ---------------------------------------------------------------------------
# Epoch functions
# ---------------------------------------------------------------------------

def train_epoch(
    model: MCDispAlignModel,
    dataloader: DataLoader,
    criterion: MCDispAlignLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    desc_prefix: str = "",
    total_epochs: int = 1,
    base_lambda_var: Optional[float] = None,
    base_lambda_dir: Optional[float] = None,
    base_lambda_cov: Optional[float] = None,
    warmup_frac: float = 0.0,
) -> Dict[str, float]:
    """Train for one epoch.

    L_var, L_dir and L_cov are ramped per optimizer step by ``warmup_ramp``
    (the dispersion statistics and containment targets need a few steps to
    mature); L_match / L_mu / R_prior are always on. Also monitors a variance
    floor-collapse ratio.
    """
    # nn.Module.train() is recursive: it would also put the CLIP backbone
    # into train mode (dropout on). A frozen backbone must stay deterministic
    # -> reset it to eval. No-op for dropout=0.0 CLIP checkpoints; matters
    # for any backbone with non-zero dropout.
    model.train()
    if model.freeze_clip:
        model.clip_model.eval()

    if base_lambda_var is None:
        base_lambda_var = criterion.lambda_var
    if base_lambda_dir is None:
        base_lambda_dir = criterion.lambda_dir
    if base_lambda_cov is None:
        base_lambda_cov = getattr(criterion, "lambda_cov", 0.0)
    steps_per_epoch = len(dataloader)
    var_near = config.MCDISP_ALIGN_VAR_FLOOR * config.MCDISP_ALIGN_VAR_FLOOR_NEAR_MULT
    floor_thresh = config.MCDISP_ALIGN_VAR_FLOOR * 2.0

    totals = {k: 0.0 for k in (
        # four-group atomics (paper §3.3); "loss" maps to loss_dict["total"]
        "loss", "match", "match_i2t", "match_t2i", "cov", "cov_viol",
        "cov_viol_img", "mu", "var", "reg", "dir", "disp",
        # weighted contributions
        "weighted_match", "weighted_cov", "weighted_mu", "weighted_var",
        "weighted_reg", "weighted_dir",
        # diagonal-variance statistics
        "img_diag_var_mean", "img_diag_var_median", "img_diag_var_min", "img_diag_var_max",
        # full-marginal statistics (d_v + sum_r U_r^2, A02)
        "img_marginal_var_mean", "img_marginal_var_median", "img_marginal_var_min",
        "img_marginal_var_max",
        # low-rank energy statistics
        "img_lowrank_var_mean",
        # text-side statistics
        "text_var_mean", "cap_var_mean", "caption_spread_mean",
        "caption_spread_median", "caption_spread_max", "var_over_spread",
        # raw readouts
        "mu_mse_raw", "marginal_log_mse",
    )}
    grad_accum = {k: 0.0 for k in (
        "img_mu_head_grad_norm", "img_logvar_head_grad_norm", "img_cov_head_grad_norm",
        "text_mu_head_grad_norm", "text_logvar_head_grad_norm",
        "global_grad_norm_before_clip", "global_grad_norm_after_clip",
        # A05/A06 stability accounting
        "dir_skipped_frac", "nonfinite_grad_steps",
    )}
    floor_ratio_sum = 0.0
    floor_severe = 0
    clip_steps = 0
    nonfinite_steps = 0
    nonfinite_grad_steps = 0
    processed_batches = 0

    desc = f"{desc_prefix}Epoch {epoch + 1}" if desc_prefix else f"Epoch {epoch + 1}"
    pbar = tqdm(dataloader, desc=desc)

    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue
        processed_batches += 1

        pil_images = batch["image"]            # List[PIL.Image]
        caption_lists = batch["captions"]      # List[List[str]]

        # Process images with CLIP processor
        pixel_values = model.process_images(pil_images).to(device)  # [B, 3, 224, 224]

        # Process text captions: flatten B*K captions, then reshape to [B, K, max_len]
        batch_size = len(pil_images)
        num_captions = len(caption_lists[0])
        all_captions = []
        for caption_list in caption_lists:
            all_captions.extend(caption_list)
        text_inputs = model.process_text(all_captions)
        input_ids = text_inputs["input_ids"].view(batch_size, num_captions, -1).to(device)
        attention_mask = text_inputs["attention_mask"].view(batch_size, num_captions, -1).to(device)

        # Per-step warmup ramp for the dispersion/containment terms
        # (L_var / L_dir / L_cov).
        ramp = warmup_ramp(epoch, batch_idx, steps_per_epoch, total_epochs, warmup_frac)
        criterion.lambda_var = base_lambda_var * ramp
        criterion.lambda_dir = base_lambda_dir * ramp
        criterion.lambda_cov = base_lambda_cov * ramp

        outputs = model(pixel_values, input_ids, attention_mask)

        # Compute MCDisp_Align loss
        loss, loss_dict = criterion(
            outputs['img_mu'], outputs['img_logvar'], outputs['img_U'],
            outputs['text_mu'], outputs['text_logvar'],
            outputs['text_mus'], outputs['text_logvars'], outputs.get('text_Us'),
        )

        if not torch.isfinite(loss):
            # Plan §8.1: count non-finite loss/grad steps; skip the update.
            nonfinite_steps += 1
            optimizer.zero_grad()
            logger.warning(f"Non-finite loss at batch {batch_idx}; step skipped.")
            continue

        optimizer.zero_grad()
        loss.backward()
        # A06: 前向有限 ≠ 反向有限（QR 等算子的反向在退化输入下可产生
        # 非有限梯度）。裁剪返回的总范数若非有限，丢弃本步更新，防止
        # NaN 梯度污染 Adam 动量。`continue` 同时跳过本批的指标累计
        # （totals / grad norms / floor 监控）——被丢弃的步不进统计。
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.MCDISP_ALIGN_GRAD_CLIP_NORM)
        if not math.isfinite(float(total_norm)):
            optimizer.zero_grad(set_to_none=True)
            nonfinite_grad_steps += 1
            grad_accum["nonfinite_grad_steps"] += 1.0
            logger.warning(f"Non-finite grad norm at batch {batch_idx}; "
                           "update skipped.")
            continue
        # Per-head grad norms AFTER the guard (who dominates at stage transitions?)
        head_norms = head_grad_norms(model)
        optimizer.step()
        grad_before = float(total_norm)
        grad_after = min(grad_before, config.MCDISP_ALIGN_GRAD_CLIP_NORM)
        if grad_before > config.MCDISP_ALIGN_GRAD_CLIP_NORM:
            clip_steps += 1

        # Accumulate losses
        for k in totals:
            src = "total" if k == "loss" else k
            totals[k] += loss_dict[src]

        # A05: fraction of samples dropped by the L_dir spectral rank guard
        # (dir_valid / dir_total are ints in the loss dict).
        grad_accum["dir_skipped_frac"] += (
            1.0 - loss_dict["dir_valid"] / max(loss_dict["dir_total"], 1))

        # Variance floor-collapse monitor (do NOT raise the floor -- it masks collapse)
        with torch.no_grad():
            img_var = torch.exp(outputs['img_logvar'])
            floor_ratio = (img_var < var_near).float().mean().item()
            floor_ratio_sum += floor_ratio
            if floor_ratio > config.MCDISP_ALIGN_VAR_FLOOR_RATIO_WARN and loss_dict['img_diag_var_mean'] < floor_thresh:
                floor_severe += 1

        # Accumulate per-head grad norms + global before/after clip (P1 #12)
        for k in ("img_mu_head_grad_norm", "img_logvar_head_grad_norm",
                  "img_cov_head_grad_norm", "text_mu_head_grad_norm", "text_logvar_head_grad_norm"):
            grad_accum[k] += head_norms.get(k, 0.0)
        grad_accum["global_grad_norm_before_clip"] += grad_before
        grad_accum["global_grad_norm_after_clip"] += grad_after

        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'match': f"{loss_dict['match']:.4f}",
            't2i': f"{loss_dict['match_t2i']:.4f}",
            'mu': f"{loss_dict['mu']:.3f}",
            'var': f"{loss_dict['var']:.3f}",
            'dir': f"{loss_dict['dir']:.3f}",
        })

    num_batches = max(processed_batches, 1)
    metrics = {k: v / num_batches for k, v in totals.items()}
    metrics.update({k: v / num_batches for k, v in grad_accum.items()})
    metrics["floor_ratio"] = floor_ratio_sum / num_batches
    metrics["floor_severe"] = floor_severe
    # Plan §8.1/§12.5 stability accounting.
    metrics["clip_step_ratio"] = clip_steps / num_batches
    metrics["nonfinite_steps"] = nonfinite_steps
    return metrics


@torch.no_grad()
def evaluate(
    model: MCDispAlignModel,
    dataloader: DataLoader,
    criterion: MCDispAlignLoss,
    device: torch.device,
    compute_recall: bool = False,
    recall_k_values=None,
) -> Dict[str, float]:
    """Evaluate the model (loss + optional retrieval Recall@K).

    Retrieval is ALWAYS the standard multi-caption protocol (N images vs N*K
    captions, any-hit I2T / per-caption T2I) under the plain-cosine MCDisp_Align
    score -- the same protocol serves checkpoint selection (``select_by="recall"``
    uses ``mc_recall@1``; ``"mr"`` uses the mean over K of ``mc_recall@k``) and
    final evaluation, so train-time selection and test-time reporting are the
    same yardstick.
    """
    model.eval()

    # mu/var/dir/cov* are the unweighted ATOMIC values -- computed regardless
    # of the run's lambda switches (lambda=0 zeroes weight and gradient, not
    # the readout), so this doubles as the ablation diagnostics aggregate.
    totals = {k: 0.0 for k in
              ("loss", "match", "mu", "var", "reg", "dir", "cov", "cov_viol",
               "cov_viol_img", "img_marginal_var_mean")}
    processed_batches = 0
    # dir diagnostics: weight each batch's projection error by its valid-sample
    # count (batches with dir_valid=0 contribute nothing, not a zero).
    dir_valid_sum = 0
    dir_total_sum = 0
    dir_weighted_sum = 0.0
    feats = {k: [] for k in ("img_mu", "img_logvar", "text_mus", "text_logvars")} \
        if compute_recall else None

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        if batch is None:
            continue
        processed_batches += 1

        pil_images = batch["image"]
        caption_lists = batch["captions"]

        pixel_values = model.process_images(pil_images).to(device)
        batch_size = len(pil_images)
        num_captions = len(caption_lists[0])
        all_captions = []
        for caption_list in caption_lists:
            all_captions.extend(caption_list)
        text_inputs = model.process_text(all_captions)
        input_ids = text_inputs["input_ids"].view(batch_size, num_captions, -1).to(device)
        attention_mask = text_inputs["attention_mask"].view(batch_size, num_captions, -1).to(device)

        outputs = model(pixel_values, input_ids, attention_mask)

        loss, loss_dict = criterion(
            outputs['img_mu'], outputs['img_logvar'], outputs['img_U'],
            outputs['text_mu'], outputs['text_logvar'],
            outputs['text_mus'], outputs['text_logvars'], outputs.get('text_Us'),
        )

        for k in totals:
            totals[k] += loss_dict["total" if k == "loss" else k]
        dir_valid_sum += loss_dict["dir_valid"]
        dir_total_sum += loss_dict["dir_total"]
        dir_weighted_sum += loss_dict["dir"] * loss_dict["dir_valid"]

        if feats is not None:
            feats["img_mu"].append(outputs['img_mu'].cpu())
            feats["img_logvar"].append(outputs['img_logvar'].cpu())
            feats["text_mus"].append(outputs['text_mus'].cpu())
            feats["text_logvars"].append(outputs['text_logvars'].cpu())

        pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

    num_batches = max(processed_batches, 1)
    metrics = {k: v / num_batches for k, v in totals.items()}
    # exact valid-weighted projection error + guard pass rate
    metrics["dir"] = dir_weighted_sum / max(dir_valid_sum, 1)
    metrics["dir_valid_frac"] = dir_valid_sum / max(dir_total_sum, 1)

    # Standard multi-caption retrieval (N vs N*K) under the MCDisp_Align score.
    # The val loader uses shuffle=False, so concatenated img_mu[i] stays aligned
    # with its own per-caption block text_mus[i] (flattened [i*K, i*K+K)).
    if compute_recall and recall_k_values and feats and feats["img_mu"]:
        img_mu = torch.cat(feats["img_mu"], dim=0).to(device)
        img_lv = torch.cat(feats["img_logvar"], dim=0).to(device)
        text_mus = torch.cat(feats["text_mus"], dim=0).to(device)
        text_lvs = torch.cat(feats["text_logvars"], dim=0).to(device)
        mc = compute_multicaption_recall(
            img_mu, img_lv, text_mus, text_lvs, recall_k_values, tau=criterion.tau)
        metrics.update(mc)
        metrics["mr"] = sum(mc[f"mc_recall@{k}"] for k in recall_k_values) / len(recall_k_values)

    return metrics


# ---------------------------------------------------------------------------
# Configuration + orchestrator
# ---------------------------------------------------------------------------

@dataclass
class MCDispAlignTrainConfig:
    """All knobs for one MCDisp_Align training run (full or ablation)."""

    # --- Data ---
    dataset: str = "coco"
    num_workers: int = field(default_factory=lambda: config.NUM_WORKERS)
    val_split: float = 0.1
    num_captions_override: Optional[int] = None   # ablation k1/k3/k5; None -> dataset default
    captions_path: Optional[str] = None           # coco override
    images_dir: Optional[str] = None              # coco override

    # --- Training ---
    epochs: int = field(default_factory=lambda: config.MCDISP_ALIGN_EPOCHS)
    batch_size: int = field(default_factory=lambda: config.MCDISP_ALIGN_BATCH_SIZE)
    clip_lr: float = field(default_factory=lambda: config.MCDISP_ALIGN_CLIP_LR)
    mlp_lr: float = field(default_factory=lambda: config.MCDISP_ALIGN_MLP_LR)
    weight_decay: float = field(default_factory=lambda: config.MCDISP_ALIGN_WEIGHT_DECAY)

    # --- Model ---
    freeze_clip: bool = field(default_factory=lambda: config.MCDISP_ALIGN_FREEZE_CLIP)
    cov_rank: int = field(default_factory=lambda: config.MCDISP_ALIGN_COV_RANK)
    dropout_rate: float = field(default_factory=lambda: config.MCDISP_ALIGN_DROPOUT_RATE)

    # --- MCDisp_Align objective (paper §3.3, four-group: match+cov/mu/var+reg/dir) ---
    lambda_match: float = field(default_factory=lambda: config.MCDISP_ALIGN_LAMBDA_MATCH)
    lambda_cov: float = field(default_factory=lambda: config.MCDISP_ALIGN_LAMBDA_COV)
    cov_alpha: float = field(default_factory=lambda: config.MCDISP_ALIGN_COV_ALPHA)
    lambda_mu: float = field(default_factory=lambda: config.MCDISP_ALIGN_LAMBDA_MU)
    lambda_var: float = field(default_factory=lambda: config.MCDISP_ALIGN_LAMBDA_VAR)
    lambda_reg: float = field(default_factory=lambda: config.MCDISP_ALIGN_LAMBDA_REG)
    lambda_dir: float = field(default_factory=lambda: config.MCDISP_ALIGN_LAMBDA_DIR)
    tau: float = field(default_factory=lambda: config.MCDISP_ALIGN_TAU)
    tau_match: float = field(default_factory=lambda: config.MCDISP_ALIGN_TAU_MATCH)
    sigma0_sq: float = field(default_factory=lambda: config.MCDISP_ALIGN_SIGMA0_SQ)
    match_score: str = field(default_factory=lambda: config.MCDISP_ALIGN_MATCH_SCORE)
    dir_eig_rel_tol: float = field(default_factory=lambda: config.MCDISP_ALIGN_DIR_EIG_REL_TOL)
    warmup_frac: float = field(default_factory=lambda: config.MCDISP_ALIGN_WARMUP_FRAC)

    # --- Schedule / selection ---
    lr_scheduler: str = field(default_factory=lambda: config.LR_SCHEDULER)
    warmup_epochs: int = field(default_factory=lambda: config.LR_WARMUP_EPOCHS)
    min_lr_ratio: float = field(default_factory=lambda: config.LR_MIN_LR_RATIO)
    select_by: str = "recall"                     # "recall" (multi-caption mc_recall@1) or "loss"
    early_stop_patience: int = 3
    no_early_stop: bool = False
    seed: int = field(default_factory=lambda: config.SEED)

    # --- System / output ---
    device: str = "cuda"
    tag: str = ""                                  # log/tqdm prefix (e.g. ablation name)
    checkpoint_dir: Optional[Path] = None          # standard naming root
    model_name: str = "mcdisp_align"                 # -> {model_name}_{dataset}_best|last.pt
    best_ckpt_path: Optional[Path] = None          # explicit override (ablation)
    last_ckpt_path: Optional[Path] = None
    resume_path: Optional[Path] = None
    skip_training: bool = False                    # ablation: eval existing checkpoint only

    # --- Final retrieval eval (ablation reporting); None = skip ---
    eval_num_samples: Optional[int] = None
    recall_k_values: tuple = field(default_factory=lambda: tuple(config.RECALL_AT_K))

    @property
    def best_path(self) -> Path:
        if self.best_ckpt_path is not None:
            return self.best_ckpt_path
        root = self.checkpoint_dir or config.CHECKPOINT_DIR
        return Path(root) / f"{self.model_name}_{self.dataset}_best.pt"

    @property
    def last_path(self) -> Path:
        if self.last_ckpt_path is not None:
            return self.last_ckpt_path
        root = self.checkpoint_dir or config.CHECKPOINT_DIR
        return Path(root) / f"{self.model_name}_{self.dataset}_last.pt"


def _build_loaders(cfg: MCDispAlignTrainConfig):
    """Build train/val DataLoaders (registry dataset + seeded random_split)."""
    full_dataset = build_train_dataset(
        dataset=cfg.dataset,
        num_captions=cfg.num_captions_override,
        captions_path=cfg.captions_path,
        images_dir=cfg.images_dir,
    )
    val_size = int(len(full_dataset) * cfg.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=filter_none_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=filter_none_collate,
    )
    return train_loader, val_loader, train_size, val_size


def run_mcdisp_align_training(cfg: MCDispAlignTrainConfig, log) -> Dict:
    """Run a full MCDisp_Align training (+ optional final retrieval eval).

    Returns a dict with ``best_val_loss``, ``best_recall``, ``select_by``, the
    last epoch index, and (when ``cfg.eval_num_samples`` is set) ``retrieval``
    metrics on the dataset's eval split — the same scorers evaluate_mcdisp_align uses.
    """
    from utils.seed import set_seed
    set_seed(cfg.seed)

    prefix = f"[{cfg.tag}] " if cfg.tag else ""

    log.info("=" * 60)
    log.info(f"{prefix}MCDisp_Align Training")
    log.info("=" * 60)
    log.info(f"{prefix}Dataset: {cfg.dataset} | Epochs: {cfg.epochs} | Batch: {cfg.batch_size}")
    log.info(f"{prefix}CLIP LR: {cfg.clip_lr} | MLP LR: {cfg.mlp_lr} | Freeze CLIP: {cfg.freeze_clip}")
    log.info(f"{prefix}Cov rank r: {cfg.cov_rank} | Tau (fixed): {cfg.tau} | Warmup frac: {cfg.warmup_frac}")
    log.info(f"{prefix}Loss weights: match={cfg.lambda_match} cov={cfg.lambda_cov}(alpha={cfg.cov_alpha}) "
             f"mu={cfg.lambda_mu} var={cfg.lambda_var} reg={cfg.lambda_reg} dir={cfg.lambda_dir} "
             f"| sigma_0^2={cfg.sigma0_sq} | match_score={cfg.match_score} "
             f"tau_match={cfg.tau_match}")
    log.info(f"{prefix}Select by: {cfg.select_by} | Device: {cfg.device}")
    log.info("=" * 60)

    # Model
    log.info(f"{prefix}Creating model...")
    model = MCDispAlignModel(
        freeze_clip=cfg.freeze_clip,
        dropout_rate=cfg.dropout_rate,
        cov_rank=cfg.cov_rank,
    )
    model = model.to(cfg.device)
    log.info(f"{prefix}Model created with {model.num_trainable_parameters():,} trainable parameters")

    # Loss
    criterion = MCDispAlignLoss(
        lambda_match=cfg.lambda_match,
        lambda_cov=cfg.lambda_cov,
        lambda_mu=cfg.lambda_mu,
        lambda_var=cfg.lambda_var,
        lambda_reg=cfg.lambda_reg,
        lambda_dir=cfg.lambda_dir,
        tau=cfg.tau,
        tau_match=cfg.tau_match,
        sigma0_sq=cfg.sigma0_sq,
        match_score=cfg.match_score,
        cov_alpha=cfg.cov_alpha,
        dir_eig_rel_tol=cfg.dir_eig_rel_tol,
    )
    log.info(f"{prefix}Using MCDisp_Align loss (paper §3.3 four-group objective: "
             "match+cov/mu/var+reg/dir)")
    criterion = criterion.to(cfg.device)

    # Optimizer
    optimizer = create_optimizer(model, cfg.freeze_clip, cfg.clip_lr, cfg.mlp_lr, cfg.weight_decay)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    best_recall = -float('inf')
    patience_counter = 0

    if cfg.resume_path:
        resume_path = Path(cfg.resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        log.info(f"{prefix}Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(str(resume_path), map_location=cfg.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(cfg.device)
        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        best_recall = checkpoint.get("best_recall", -float('inf'))
        patience_counter = checkpoint.get("patience_counter", 0)
        base_lrs = checkpoint.get("base_lrs", base_lrs)
        # A13: the checkpoint must come from the same objective; a missing tag
        # means a legacy (pre four-group-v2) checkpoint.
        if checkpoint.get("objective_version") != config.MCDISP_ALIGN_OBJECTIVE_VERSION:
            log.warning(
                f"{prefix}Checkpoint objective_version "
                f"{checkpoint.get('objective_version')!r} != "
                f"{config.MCDISP_ALIGN_OBJECTIVE_VERSION!r} (legacy objective; "
                "resuming initializes weights only, not an equivalent-objective "
                "continuation).")
        log.info(f"{prefix}Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}, "
                 f"best_recall: {best_recall:.4f}, patience_counter: {patience_counter}")

    # Data
    train_loader, val_loader, train_size, val_size = _build_loaders(cfg)
    log.info(f"{prefix}Train samples: {train_size}, Val samples: {val_size} (dataset={cfg.dataset})")
    log.info(f"{prefix}Batch size: {cfg.batch_size}, Train batches per epoch: {len(train_loader)}")

    best_path = cfg.best_path
    last_epoch = start_epoch

    if not cfg.skip_training:
        log.info(f"{prefix}Starting training from epoch {start_epoch + 1}...")
        for epoch in range(start_epoch, cfg.epochs):
            last_epoch = epoch

            apply_lr_for_epoch(optimizer, base_lrs, epoch, cfg.epochs,
                               cfg.warmup_epochs, cfg.min_lr_ratio,
                               cfg.lr_scheduler, log)

            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, cfg.device, epoch,
                desc_prefix=cfg.tag, total_epochs=cfg.epochs,
                base_lambda_var=cfg.lambda_var, base_lambda_dir=cfg.lambda_dir,
                base_lambda_cov=cfg.lambda_cov, warmup_frac=cfg.warmup_frac,
            )
            val_metrics = evaluate(
                model, val_loader, criterion, cfg.device,
                compute_recall=(cfg.select_by in ("recall", "mr")),
                recall_k_values=list(cfg.recall_k_values),
            )

            log.info(
                f"{prefix}Epoch {epoch + 1}/{cfg.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Match: {train_metrics['match']:.4f}, "
                f"Cov: {train_metrics['cov']:.2f} (viol {train_metrics['cov_viol']:.2f}), "
                f"Mu: {train_metrics['mu']:.4f}, Var: {train_metrics['var']:.4f}, "
                f"Reg: {train_metrics['reg']:.4f}, Dir: {train_metrics['dir']:.4f}, "
                f"σ²diag: {train_metrics['img_diag_var_mean']:.4f}, "
                f"FloorR: {train_metrics.get('floor_ratio', 0):.3f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"σ²marg: {val_metrics['img_marginal_var_mean']:.4f}, "
                f"mc R@1/5/10: {val_metrics.get('mc_recall@1', 0):.3f}/"
                f"{val_metrics.get('mc_recall@5', 0):.3f}/{val_metrics.get('mc_recall@10', 0):.3f}, "
                f"mR: {val_metrics.get('mr', 0):.3f}"
            )

            if train_metrics.get("floor_severe", 0) > 0:
                log.warning(
                    f"{prefix}SEVERE: variance floor collapse at epoch {epoch + 1}: "
                    f"{train_metrics['floor_severe']} batches had "
                    f">{config.MCDISP_ALIGN_VAR_FLOOR_RATIO_WARN:.0%} of dims near the floor "
                    f"(σ²diag={train_metrics['img_diag_var_mean']:.4f}, mean FloorR={train_metrics['floor_ratio']:.3f}). "
                    f"Do NOT raise MCDISP_ALIGN_VAR_FLOOR -- it masks the collapse; check L_var / the warmup ramp."
                )

            # Per-epoch diagnostics: weighted contributions, variance/cov stats, per-head grad norms
            u_over_diag = (train_metrics['img_lowrank_var_mean']
                           / (train_metrics['img_diag_var_mean'] + 1e-12))
            log.info(
                f"{prefix}  diag: weighted[match={train_metrics['weighted_match']:.3f} "
                f"cov={train_metrics['weighted_cov']:.3f} "
                f"mu={train_metrics['weighted_mu']:.3f} var={train_metrics['weighted_var']:.3f} "
                f"reg={train_metrics['weighted_reg']:.3f} dir={train_metrics['weighted_dir']:.3f}] | "
                f"σ²diag[min/med/mean/max]={train_metrics['img_diag_var_min']:.4f}/"
                f"{train_metrics['img_diag_var_median']:.4f}/"
                f"{train_metrics['img_diag_var_mean']:.4f}/{train_metrics['img_diag_var_max']:.4f} | "
                f"s²[mean/med/max]={train_metrics['caption_spread_mean']:.4f}/"
                f"{train_metrics['caption_spread_median']:.4f}/{train_metrics['caption_spread_max']:.4f} "
                f"txtσ²={train_metrics['text_var_mean']:.4f} capσ²={train_metrics['cap_var_mean']:.4f} "
                f"σ²/s²={train_metrics['var_over_spread']:.2f} | U/diag={u_over_diag:.2f} | "
                f"grad[μv={train_metrics['img_mu_head_grad_norm']:.3f} "
                f"σ²v={train_metrics['img_logvar_head_grad_norm']:.3f} "
                f"Uv={train_metrics.get('img_cov_head_grad_norm', 0):.3f} "
                f"μt={train_metrics['text_mu_head_grad_norm']:.3f} "
                f"σ²t={train_metrics['text_logvar_head_grad_norm']:.3f}] "
                f"gnorm[{train_metrics['global_grad_norm_before_clip']:.2f}->"
                f"{train_metrics['global_grad_norm_after_clip']:.2f}] "
                f"dirSkip={train_metrics['dir_skipped_frac']:.2f} "
                f"nonfinite[loss={train_metrics['nonfinite_steps']} "
                f"grad={train_metrics['nonfinite_grad_steps']:.2f}]"
            )

            # Best-checkpoint selection. "recall" = multi-caption mc_recall@1
            # (standard N vs N*K protocol, same as final evaluation).
            if cfg.select_by == "recall" and "mc_recall@1" in val_metrics:
                current_score = val_metrics["mc_recall@1"]
                improved = current_score > best_recall
                if improved:
                    best_recall = current_score
            elif cfg.select_by == "mr" and "mr" in val_metrics:
                # Ablation study: unified multi-caption development mR.
                current_score = val_metrics["mr"]
                improved = current_score > best_recall
                if improved:
                    best_recall = current_score
            else:
                current_score = val_metrics["loss"]
                improved = current_score < best_val_loss
                if improved:
                    best_val_loss = current_score

            if improved:
                model.save(str(best_path), epoch=epoch + 1,
                           objective_version=config.MCDISP_ALIGN_OBJECTIVE_VERSION)
                if cfg.select_by == "loss":
                    score_str = f"val_loss: {best_val_loss:.4f}"
                else:
                    score_str = f"{cfg.select_by}: {best_recall:.4f}"
                log.info(f"{prefix}Best model saved ({score_str}) -> {best_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                log.info(f"{prefix}No improvement. Patience: {patience_counter}/{cfg.early_stop_patience}")

            if not cfg.no_early_stop and patience_counter >= cfg.early_stop_patience:
                log.info(f"{prefix}Early stopping triggered at epoch {epoch + 1}")
                break

        # Save final (last) checkpoint with full training state for resumption
        last_path = cfg.last_path
        final_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": last_epoch + 1,
            "best_val_loss": best_val_loss,
            "best_recall": best_recall,
            "patience_counter": patience_counter,
            "base_lrs": base_lrs,
            "select_by": cfg.select_by,
            # A13: objective identity of the saved run
            "objective_version": config.MCDISP_ALIGN_OBJECTIVE_VERSION,
        }
        torch.save(final_state, str(last_path))
        log.info(f"{prefix}Final model saved to {last_path}")
        log.info(f"{prefix}Best val loss: {best_val_loss:.4f} | Best mc_recall@1: {best_recall:.4f} "
                 f"(selected by: {cfg.select_by})")

    results = {
        "best_val_loss": best_val_loss,
        "best_recall": best_recall,
        "select_by": cfg.select_by,
        "last_epoch": last_epoch,
        "best_checkpoint": str(best_path),
    }

    # Optional final retrieval eval on the dataset's eval split (ablation reporting),
    # using the SAME scorers as evaluate_mcdisp_align.
    if cfg.eval_num_samples is not None and best_path.exists():
        log.info(f"{prefix}Final retrieval eval on {cfg.dataset} "
                 f"(num_samples={cfg.eval_num_samples})...")
        model.load(str(best_path))
        eval_loader, n_eval = build_eval_dataloader(
            cfg.dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            num_samples=cfg.eval_num_samples,
            num_captions=cfg.num_captions_override,
        )
        eval_metrics = evaluate(
            model, eval_loader, criterion, cfg.device,
            compute_recall=True, recall_k_values=list(cfg.recall_k_values),
        )
        results["retrieval"] = {
            "num_samples": n_eval,
            "mc_recall": {f"R@{k}": eval_metrics.get(f"mc_recall@{k}", 0.0)
                          for k in cfg.recall_k_values},
        }
        # Ablation diagnostics (unweighted atomics on the eval split; the
        # expected-verification table of the w/o Coverage/Mean/Variance/
        # Direction variants): coverage rates and the three alignment errors.
        results["alignment"] = {
            "coverage_caption": 1.0 - eval_metrics.get("cov_viol", 0.0),
            "coverage_set": 1.0 - eval_metrics.get("cov_viol_img", 0.0),
            "center_mse": eval_metrics.get("mu", 0.0),
            "var_log_mse": eval_metrics.get("var", 0.0),
            "dir_proj_err": eval_metrics.get("dir", 0.0),
            "dir_valid_frac": (
                eval_metrics.get("dir_valid_frac", 0.0)),
        }
        log.info(f"{prefix}Final mc R@1/5/10: "
                 f"{eval_metrics.get('mc_recall@1', 0):.3f}/"
                 f"{eval_metrics.get('mc_recall@5', 0):.3f}/"
                 f"{eval_metrics.get('mc_recall@10', 0):.3f}")

    log.info(f"{prefix}Training completed!")
    return results
