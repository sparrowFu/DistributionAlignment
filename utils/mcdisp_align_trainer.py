"""MCDisp_Align training orchestration: staged loss schedule, grad-norm clipping, recall/loss best-checkpoint selection, early stopping, best+last checkpointing, and resume. Full runs and ablation variants share one code path, differing only in the loss weights / ``cov_rank`` / ``num_captions`` / ``dataset`` / checkpoint paths passed via :class:`MCDispAlignTrainConfig`. The per-epoch functions are exported for direct unit testing."""

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
from utils.retrieval import (
    compute_multicaption_recall,
    compute_recall_bidirectional,
    compute_recall_mcdisp_align_chunked,
)


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
    """Create optimizer with different learning rates for CLIP and MLP/cov heads."""
    head_params = (
        list(model.img_mu_head.parameters())
        + list(model.img_logvar_head.parameters())
        + list(model.text_mu_head.parameters())
        + list(model.text_logvar_head.parameters())
    )
    if getattr(model, "cov_rank", 0) > 0:
        head_params += list(model.img_cov_head.parameters())

    if freeze_clip:
        # Only train distribution + image covariance heads
        return optim.Adam(head_params, lr=mlp_lr, weight_decay=weight_decay)

    # CLIP and heads with different learning rates
    return optim.Adam(
        [
            {"params": model.clip_model.parameters(), "lr": clip_lr},
            {"params": head_params, "lr": mlp_lr},
        ],
        weight_decay=weight_decay,
    )


def _stage_bounds(total: int):
    """5-stage epoch boundaries: (warmup_end, var_bootstrap_end, pos_coverage_end, full_start).

    Warmup = WARMUP_FRAC; Full = FULL_FRAC; the middle is split into thirds for
    Var-Bootstrap / Pos-Coverage / Neg-Repulsion.
    """
    warmup_end = max(1, int(round(total * config.MCDISP_ALIGN_STAGE_WARMUP_FRAC)))
    full_start = max(warmup_end + 3,
                     total - max(1, int(round(total * config.MCDISP_ALIGN_STAGE_FULL_FRAC))))
    middle = max(3, full_start - warmup_end)
    third = max(1, middle // 3)
    return warmup_end, warmup_end + third, warmup_end + 2 * third, full_start


def stage_multipliers(epoch: int, total: int, no_staged: bool) -> Dict[str, float]:
    """Per-loss multipliers for the 5-stage MCDisp_Align schedule.

    Stages (by epoch fraction):
      Mean-Warmup   : L_set + L_mu + L_reg                 (var/cover/cov off)
      Var-Bootstrap : + L_var (ramped per-step via var_ramp)
      Pos-Coverage  : + L_cover_pos
      Neg-Repulsion : + L_cover_neg
      Full          : + L_cov (per-epoch ramp 0 -> 1)

    var & uncertainty_grad_alpha are RAMPED PER OPTIMIZER STEP in train_epoch
    (var_ramp / alpha_schedule); the per-epoch "var" here just marks the stage
    (0 in warmup, 1 once Var-Bootstrap begins). cover_pos/cover_neg/cov are gated
    per-epoch. L_reg is always on.
    """
    base = {"ctr": 1.0, "mu": 1.0, "reg": 1.0,
            "var": 1.0, "cover_pos": 1.0, "cover_neg": 1.0, "cov": 1.0}
    if no_staged or total <= 0:
        base["stage"] = "full"
        return base
    we, vb, pe, fs = _stage_bounds(total)
    if epoch < we:
        base.update(var=0.0, cover_pos=0.0, cover_neg=0.0, cov=0.0); base["stage"] = "warmup"
    elif epoch < vb:
        base.update(cover_pos=0.0, cover_neg=0.0, cov=0.0); base["stage"] = "var_bootstrap"
    elif epoch < pe:
        base.update(cover_neg=0.0, cov=0.0); base["stage"] = "pos_coverage"
    elif epoch < fs:
        base.update(cov=0.0); base["stage"] = "neg_repulsion"
    else:
        full_len = max(1, total - fs)
        base["cov"] = min(1.0, (epoch - fs + 1) / full_len); base["stage"] = "full"
    return base


def var_ramp(epoch: int, step: int, steps_per_epoch: int, total: int) -> float:
    """L_var multiplier ramped per optimizer step.

    0 in Warmup; 0.05 -> 1.0 linearly across Var-Bootstrap; 1.0 afterwards. The
    gradual ramp avoids the hard 0->1 step that collapsed sigma^2 at the Main
    boundary (sigma^2 sat above the immature caption spread, so a full-strength
    L_var yanked it to the floor).
    """
    we, vb, _pe, _fs = _stage_bounds(total)
    if epoch < we:
        return 0.0
    if epoch < vb:
        span = max(1, (vb - we) * steps_per_epoch)
        progress = (epoch - we) * steps_per_epoch + step
        return 0.05 + (1.0 - 0.05) * min(1.0, progress / span)
    return 1.0


def alpha_schedule(epoch: int, step: int, steps_per_epoch: int, total: int) -> float:
    """uncertainty_grad_alpha schedule (straight-through L_set -> sigma^2 grad scale).

    0 in Warmup (block L_set from pulling sigma^2 down); 0 -> 1 across
    Var-Bootstrap (tied to var_ramp); 1.0 afterwards.
    """
    we, vb, _pe, _fs = _stage_bounds(total)
    if epoch < we:
        return 0.0
    if epoch < vb:
        span = max(1, (vb - we) * steps_per_epoch)
        progress = (epoch - we) * steps_per_epoch + step
        return min(1.0, progress / span)
    return 1.0


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
    criterion,  # MCDispAlignLoss | MCDispAlignKLLoss (cfg.loss_name)
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    desc_prefix: str = "",
    total_epochs: int = 1,
    base_lambda_var: Optional[float] = None,
    base_lambda_kl: Optional[float] = None,
) -> Dict[str, float]:
    """Train for one epoch.

    L_var and uncertainty_grad_alpha are ramped PER OPTIMIZER STEP (var_ramp /
    alpha_schedule) to avoid the hard Main-boundary step that collapsed sigma^2.
    The KL variant (loss_name="kl") has lambda_kl in place of the
    (lambda_mu, lambda_var) pair; its KL term rides the SAME var_ramp so the
    anti-collapse schedule (off while the caption heads are immature) is
    preserved. Also monitors a variance floor-collapse ratio.
    """
    model.train()
    if not model.freeze_clip:
        model.clip_model.train()

    if base_lambda_var is None:
        base_lambda_var = getattr(criterion, "lambda_var", 0.0)
    if base_lambda_kl is None:
        base_lambda_kl = getattr(criterion, "lambda_kl", 0.0)
    steps_per_epoch = len(dataloader)
    var_near = config.MCDISP_ALIGN_VAR_FLOOR * config.MCDISP_ALIGN_VAR_FLOOR_NEAR_MULT
    floor_thresh = config.MCDISP_ALIGN_VAR_FLOOR * 2.0

    totals = {k: 0.0 for k in (
        "loss", "set_nce", "mu", "var", "cover_pos", "cover_neg", "cov", "reg", "img_var_avg",
        # weighted contributions (P1 #9)
        "weighted_set_nce", "weighted_mu", "weighted_var", "weighted_cover_pos",
        "weighted_cover_neg", "weighted_cov", "weighted_reg",
        # variance statistics (P1 #10)
        "img_var_min", "img_var_median", "img_var_mean", "img_var_max",
        "text_var_mean", "caption_spread_mean", "caption_spread_median", "caption_spread_max",
        # low-rank covariance statistics (P1 #11)
        "u_energy", "diag_var_energy", "u_over_diag",
    )}
    grad_accum = {k: 0.0 for k in (
        "img_mu_head_grad_norm", "img_logvar_head_grad_norm", "img_cov_head_grad_norm",
        "text_mu_head_grad_norm", "text_logvar_head_grad_norm",
        "global_grad_norm_before_clip", "global_grad_norm_after_clip",
    )}
    floor_ratio_sum = 0.0
    floor_severe = 0
    clip_steps = 0
    nonfinite_steps = 0
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

        # Per-step L_var ramp + uncertainty_grad_alpha (anti-collapse scheduling).
        # The KL variant has no lambda_var/lambda_mu attributes; its lambda_kl
        # rides the same ramp (0 in Warmup, 0.05->1 in Var-Bootstrap, 1 after).
        ramp = var_ramp(epoch, batch_idx, steps_per_epoch, total_epochs)
        if hasattr(criterion, "lambda_var"):
            criterion.lambda_var = base_lambda_var * ramp
        if hasattr(criterion, "lambda_kl"):
            criterion.lambda_kl = base_lambda_kl * ramp
        criterion.uncertainty_grad_alpha = alpha_schedule(
            epoch, batch_idx, steps_per_epoch, total_epochs)

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
        # Per-head grad norms BEFORE clip (who dominates at stage transitions?)
        head_norms = head_grad_norms(model)
        # Clip global grad norm (returns the pre-clip total norm over ALL params);
        # protects against L_cov / cover spikes destabilizing the retrieval means.
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.MCDISP_ALIGN_GRAD_CLIP_NORM)
        optimizer.step()
        grad_before = float(total_norm)
        grad_after = min(grad_before, config.MCDISP_ALIGN_GRAD_CLIP_NORM)
        if math.isfinite(grad_before) and grad_before > config.MCDISP_ALIGN_GRAD_CLIP_NORM:
            clip_steps += 1

        # Accumulate losses
        for k in totals:
            src = "total" if k == "loss" else k
            totals[k] += loss_dict[src]

        # Variance floor-collapse monitor (do NOT raise the floor -- it masks collapse)
        with torch.no_grad():
            img_var = torch.exp(outputs['img_logvar'])
            floor_ratio = (img_var < var_near).float().mean().item()
            floor_ratio_sum += floor_ratio
            if floor_ratio > config.MCDISP_ALIGN_VAR_FLOOR_RATIO_WARN and loss_dict['img_var_avg'] < floor_thresh:
                floor_severe += 1

        # Accumulate per-head grad norms + global before/after clip (P1 #12)
        for k in ("img_mu_head_grad_norm", "img_logvar_head_grad_norm",
                  "img_cov_head_grad_norm", "text_mu_head_grad_norm", "text_logvar_head_grad_norm"):
            grad_accum[k] += head_norms.get(k, 0.0)
        grad_accum["global_grad_norm_before_clip"] += grad_before
        grad_accum["global_grad_norm_after_clip"] += grad_after

        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'NCE': f"{loss_dict['set_nce']:.4f}",
            'mu': f"{loss_dict['mu']:.3f}",
            'var': f"{loss_dict['var']:.3f}",
            'cov': f"{loss_dict['cov']:.3f}",
            'σ²i': f"{loss_dict['img_var_avg']:.3f}",
            'flr': f"{floor_ratio:.2f}",
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
    criterion,  # MCDispAlignLoss | MCDispAlignKLLoss (cfg.loss_name)
    device: torch.device,
    compute_recall: bool = False,
    recall_k_values=None,
    multicaption: bool = False,
) -> Dict[str, float]:
    """Evaluate the model (loss + optional retrieval Recall@K).

    ``multicaption=True`` additionally computes the standard multi-caption
    protocol (N images vs N*K captions, any-hit I2T / per-caption T2I, plan
    §3.3) and its mR (mean of the 6 recall values, plan §6.1). This is the
    ablation study's checkpoint-selection metric (``select_by="mr"``); the
    merged-center recall stays available as a diagnostic.
    """
    model.eval()

    totals = {k: 0.0 for k in
              ("loss", "set_nce", "mu", "var", "cover_pos", "cover_neg", "cov", "reg", "img_var_avg")}
    processed_batches = 0
    feats = {k: [] for k in ("img_mu", "text_mu", "img_logvar", "text_logvar")} \
        if compute_recall else None
    cap_feats = {k: [] for k in ("text_mus", "text_logvars")} \
        if (compute_recall and multicaption) else None

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

        if feats is not None:
            feats["img_mu"].append(outputs['img_mu'].cpu())
            feats["text_mu"].append(outputs['text_mu'].cpu())
            feats["img_logvar"].append(outputs['img_logvar'].cpu())
            feats["text_logvar"].append(outputs['text_logvar'].cpu())
        if cap_feats is not None:
            cap_feats["text_mus"].append(outputs['text_mus'].cpu())
            cap_feats["text_logvars"].append(outputs['text_logvars'].cpu())

        pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

    num_batches = max(processed_batches, 1)
    metrics = {k: v / num_batches for k, v in totals.items()}

    # Retrieval Recall@K (image<->text, diagonal pairing). The val loader uses
    # shuffle=False, so concatenated img_mu[i] stays aligned with its own
    # caption-set text_mu[i] -> the diagonal is the positive pair.
    # Primary score: MCDisp_Align uncertainty-discounted cosine (= what L_set optimizes).
    # Secondary: plain cosine-on-means (mean-only retrieval mode).
    if compute_recall and recall_k_values and feats and feats["img_mu"]:
        img_mu = torch.cat(feats["img_mu"], dim=0).to(device)
        text_mu = torch.cat(feats["text_mu"], dim=0).to(device)
        img_lv = torch.cat(feats["img_logvar"], dim=0).to(device)
        text_lv = torch.cat(feats["text_logvar"], dim=0).to(device)
        mcdisp_align = compute_recall_mcdisp_align_chunked(
            img_mu, img_lv, text_mu, text_lv, recall_k_values, tau=criterion.tau)
        metrics.update(mcdisp_align)
        cos = compute_recall_bidirectional(img_mu, text_mu, recall_k_values, normalize=True)
        for k in recall_k_values:
            metrics[f"cos_recall@{k}"] = (cos[f"recall_i2t@{k}"] + cos[f"recall_t2i@{k}"]) / 2

        # Standard multi-caption protocol (plan §3.3) + mR (plan §6.1).
        if cap_feats is not None and cap_feats["text_mus"]:
            text_mus = torch.cat(cap_feats["text_mus"], dim=0).to(device)
            text_lvs = torch.cat(cap_feats["text_logvars"], dim=0).to(device)
            mc = compute_multicaption_recall(
                img_mu, img_lv, text_mus, text_lvs, recall_k_values, tau=criterion.tau)
            for k in recall_k_values:
                metrics[f"mc_recall@{k}"] = mc[f"mc_recall@{k}"]
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

    # --- Manifest-backed data (ablation study, plan §3.1/§3.2) ---
    # When both are set, train/dev loaders come from the image-exclusive audit
    # manifests instead of the registry + random_split path (legacy default).
    train_manifest: Optional[Path] = None
    dev_manifest: Optional[Path] = None
    manifest_num_captions: int = 5          # training K (random-sampled when mode="random")
    manifest_sample_mode: str = "first"     # "first" | "random" (K=1/K=3 regimes)
    dev_num_captions: int = 5               # uniform dev protocol for checkpoint selection

    # --- Training ---
    epochs: int = field(default_factory=lambda: config.MCDISP_ALIGN_EPOCHS)
    batch_size: int = field(default_factory=lambda: config.MCDISP_ALIGN_BATCH_SIZE)
    clip_lr: float = field(default_factory=lambda: config.MCDISP_ALIGN_CLIP_LR)
    mlp_lr: float = field(default_factory=lambda: config.MCDISP_ALIGN_MLP_LR)
    weight_decay: float = field(default_factory=lambda: config.MCDISP_ALIGN_WEIGHT_DECAY)

    # --- Model ---
    freeze_clip: bool = field(default_factory=lambda: config.MCDISP_ALIGN_FREEZE_CLIP)
    cov_rank: int = field(default_factory=lambda: config.MCDISP_ALIGN_COV_RANK)
    distribution_merging: str = field(default_factory=lambda: config.MCDISP_ALIGN_DISTRIBUTION_MERGING)
    dropout_rate: float = field(default_factory=lambda: config.MCDISP_ALIGN_DROPOUT_RATE)

    # --- MCDisp_Align loss weights ---
    lambda_ctr: float = field(default_factory=lambda: config.MCDISP_ALIGN_LAMBDA_CTR)
    lambda_mu: float = field(default_factory=lambda: config.MCDISP_ALIGN_LAMBDA_MU)
    lambda_var: float = field(default_factory=lambda: config.MCDISP_ALIGN_LAMBDA_VAR)
    lambda_cover_pos: float = field(default_factory=lambda: config.MCDISP_ALIGN_LAMBDA_COVER_POS)
    lambda_cover_neg: float = field(default_factory=lambda: config.MCDISP_ALIGN_LAMBDA_COVER_NEG)
    lambda_cov: float = field(default_factory=lambda: config.MCDISP_ALIGN_LAMBDA_COV)
    lambda_reg: float = field(default_factory=lambda: config.MCDISP_ALIGN_LAMBDA_REG)
    tau: float = field(default_factory=lambda: config.MCDISP_ALIGN_TAU)
    m_pos: float = field(default_factory=lambda: config.MCDISP_ALIGN_M_POS)
    m_neg: float = field(default_factory=lambda: config.MCDISP_ALIGN_M_NEG)
    target_var: float = field(default_factory=lambda: config.MCDISP_ALIGN_TARGET_VAR)
    use_uncertainty_sim: bool = field(default_factory=lambda: config.MCDISP_ALIGN_USE_UNCERTAINTY_SIM)

    # --- Loss selection ---
    # "standard" -> MCDispAlignLoss (exact original construction, default);
    # "kl"       -> losses.mcdisp_align_losses_kl.MCDispAlignKLLoss, where the
    #               (lambda_mu, lambda_var) pair is replaced by a single
    #               lambda_kl (the KL folds mean+variance alignment into one
    #               term). Both classes share the forward signature and the
    #               loss_dict key contract, so training/eval/selection code
    #               downstream is unchanged.
    loss_name: str = "standard"                   # "standard" | "kl"
    lambda_kl: float = 1.0                        # KL term weight ("kl" only)

    # --- Schedule / selection ---
    no_staged: bool = False
    lr_scheduler: str = field(default_factory=lambda: config.LR_SCHEDULER)
    warmup_epochs: int = field(default_factory=lambda: config.LR_WARMUP_EPOCHS)
    min_lr_ratio: float = field(default_factory=lambda: config.LR_MIN_LR_RATIO)
    select_by: str = "recall"                     # "recall" (MCDisp_Align R@1) or "loss"
    early_stop_patience: int = 3
    no_early_stop: bool = False
    seed: int = field(default_factory=lambda: config.SEED)

    # --- System / output ---
    device: str = "cuda"
    tag: str = ""                                  # log/tqdm prefix (e.g. ablation name)
    checkpoint_dir: Optional[Path] = None          # standard naming root
    model_name: str = "mcdisp_align"               # -> {model_name}[_kl]_{dataset}_best|last.pt
    best_ckpt_path: Optional[Path] = None          # explicit override (ablation)
    last_ckpt_path: Optional[Path] = None
    resume_path: Optional[Path] = None
    skip_training: bool = False                    # ablation: eval existing checkpoint only

    # --- Final retrieval eval (ablation reporting); None = skip ---
    eval_num_samples: Optional[int] = None
    recall_k_values: tuple = field(default_factory=lambda: tuple(config.RECALL_AT_K))

    @property
    def ckpt_model_name(self) -> str:
        """Model name used in the DEFAULT checkpoint filenames. The KL loss
        gets its own tag so the two objectives never overwrite each other's
        weights: "standard" -> mcdisp_align_{dataset}_best|last.pt (unchanged
        legacy name), "kl" -> mcdisp_align_kl_{dataset}_best|last.pt. Explicit
        best_ckpt_path / last_ckpt_path overrides bypass this entirely."""
        suffix = "_kl" if self.loss_name == "kl" else ""
        return f"{self.model_name}{suffix}"

    @property
    def best_path(self) -> Path:
        if self.best_ckpt_path is not None:
            return self.best_ckpt_path
        root = self.checkpoint_dir or config.CHECKPOINT_DIR
        return Path(root) / f"{self.ckpt_model_name}_{self.dataset}_best.pt"

    @property
    def last_path(self) -> Path:
        if self.last_ckpt_path is not None:
            return self.last_ckpt_path
        root = self.checkpoint_dir or config.CHECKPOINT_DIR
        return Path(root) / f"{self.ckpt_model_name}_{self.dataset}_last.pt"


def _build_loaders(cfg: MCDispAlignTrainConfig):
    """Build train/val DataLoaders.

    Manifest path (ablation study): image-exclusive audit manifests, no caption
    repeat-padding; the dev loader is deterministic (first-K) with a UNIFORM
    K for every config so checkpoint selection is comparable across variants
    (plan §8: 不使用不同组成的 validation 指标).
    """
    if cfg.train_manifest is not None and cfg.dev_manifest is not None:
        from data.manifest_caption_dataset import ManifestCaptionDataset

        train_ds = ManifestCaptionDataset(
            cfg.train_manifest, config.IMAGES_DIR,
            num_captions=cfg.manifest_num_captions,
            sample_mode=cfg.manifest_sample_mode,
        )
        dev_ds = ManifestCaptionDataset(
            cfg.dev_manifest, config.IMAGES_DIR,
            num_captions=cfg.dev_num_captions,
            sample_mode="first",
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=filter_none_collate,
        )
        val_loader = DataLoader(
            dev_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=filter_none_collate,
        )
        return train_loader, val_loader, len(train_ds), len(dev_ds)

    # Legacy registry path (random_split of the full training pool).
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
    log.info(f"{prefix}Cov rank r: {cfg.cov_rank} | Tau (fixed): {cfg.tau} | Staged: {not cfg.no_staged}")
    log.info(f"{prefix}Loss function: {cfg.loss_name}"
             + (f" (lambda_kl={cfg.lambda_kl}; replaces mu/var weights)"
                if cfg.loss_name == "kl" else ""))
    log.info(f"{prefix}Checkpoints: best={cfg.best_path} | last={cfg.last_path}")
    log.info(f"{prefix}Loss weights: ctr={cfg.lambda_ctr} mu={cfg.lambda_mu} var={cfg.lambda_var} "
             f"cover_pos={cfg.lambda_cover_pos} cover_neg={cfg.lambda_cover_neg} cov={cfg.lambda_cov} reg={cfg.lambda_reg}")
    log.info(f"{prefix}Select by: {cfg.select_by} | Device: {cfg.device}")
    log.info("=" * 60)

    # Model
    log.info(f"{prefix}Creating model...")
    model = MCDispAlignModel(
        freeze_clip=cfg.freeze_clip,
        distribution_merging=cfg.distribution_merging,
        dropout_rate=cfg.dropout_rate,
        cov_rank=cfg.cov_rank,
    )
    model = model.to(cfg.device)
    log.info(f"{prefix}Model created with {model.num_trainable_parameters():,} trainable parameters")

    # Loss, selected by cfg.loss_name. "standard" keeps the exact original
    # MCDispAlignLoss construction; "kl" swaps in the KL variant (same forward
    # signature and loss_dict key contract, so nothing downstream changes).
    if cfg.loss_name == "kl":
        from losses.mcdisp_align_losses_kl import MCDispAlignKLLoss
        criterion = MCDispAlignKLLoss(
            lambda_ctr=cfg.lambda_ctr,
            lambda_kl=cfg.lambda_kl,
            lambda_cover_pos=cfg.lambda_cover_pos,
            lambda_cover_neg=cfg.lambda_cover_neg,
            lambda_cov=cfg.lambda_cov,
            lambda_reg=cfg.lambda_reg,
            tau=cfg.tau,
            m_pos=cfg.m_pos,
            target_var=cfg.target_var,
            m_neg=cfg.m_neg,
            use_uncertainty_sim=cfg.use_uncertainty_sim,
        )
        log.info(f"{prefix}Using MCDisp_Align KL-variant loss "
                 f"(lambda_kl={cfg.lambda_kl}; mu/var terms folded into the KL)")
    elif cfg.loss_name == "standard":
        criterion = MCDispAlignLoss(
            lambda_ctr=cfg.lambda_ctr,
            lambda_mu=cfg.lambda_mu,
            lambda_var=cfg.lambda_var,
            lambda_cover_pos=cfg.lambda_cover_pos,
            lambda_cover_neg=cfg.lambda_cover_neg,
            lambda_cov=cfg.lambda_cov,
            lambda_reg=cfg.lambda_reg,
            tau=cfg.tau,
            m_pos=cfg.m_pos,
            target_var=cfg.target_var,
            m_neg=cfg.m_neg,
            use_uncertainty_sim=cfg.use_uncertainty_sim,
        )
        log.info(f"{prefix}Using MCDisp_Align loss (uncertainty-discounted cosine L_set)")
    else:
        raise ValueError(
            f"Unknown loss_name {cfg.loss_name!r} (expected 'standard' or 'kl')")
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
        # The checkpoint must come from the same objective; a missing key means
        # a legacy (pre-loss-selection) checkpoint, assumed "standard".
        ckpt_loss = checkpoint.get("loss_name", "standard")
        if ckpt_loss != cfg.loss_name:
            log.warning(
                f"{prefix}Checkpoint loss_name {ckpt_loss!r} != cfg.loss_name "
                f"{cfg.loss_name!r} (resuming initializes weights only, not an "
                "equivalent-objective continuation).")
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

            # Staged loss schedule
            mult = stage_multipliers(epoch, cfg.epochs, cfg.no_staged)
            criterion.lambda_ctr = cfg.lambda_ctr * mult["ctr"]
            if hasattr(criterion, "lambda_mu"):
                criterion.lambda_mu = cfg.lambda_mu * mult["mu"]
            # KL variant: no lambda_mu/lambda_var here; its lambda_kl is fully
            # controlled by the PER-STEP var_ramp inside train_epoch (off in
            # Warmup, ramped in Var-Bootstrap, 1 afterwards) -- the same
            # anti-collapse schedule L_var went through.
            # var & uncertainty_grad_alpha are ramped PER STEP in train_epoch (var_ramp / alpha_schedule)
            criterion.lambda_cover_pos = cfg.lambda_cover_pos * mult["cover_pos"]
            criterion.lambda_cover_neg = cfg.lambda_cover_neg * mult["cover_neg"]
            criterion.lambda_cov = cfg.lambda_cov * mult["cov"]
            criterion.lambda_reg = cfg.lambda_reg * mult["reg"]
            log.info(f"{prefix}Epoch {epoch + 1} stage={mult.get('stage')} multipliers: {mult}")

            apply_lr_for_epoch(optimizer, base_lrs, epoch, cfg.epochs,
                               cfg.warmup_epochs, cfg.min_lr_ratio,
                               cfg.lr_scheduler, log)

            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, cfg.device, epoch,
                desc_prefix=cfg.tag, total_epochs=cfg.epochs, base_lambda_var=cfg.lambda_var,
                base_lambda_kl=cfg.lambda_kl,
            )
            val_metrics = evaluate(
                model, val_loader, criterion, cfg.device,
                compute_recall=(cfg.select_by in ("recall", "mr")),
                recall_k_values=list(cfg.recall_k_values),
                multicaption=(cfg.select_by == "mr"),
            )

            log.info(
                f"{prefix}Epoch {epoch + 1}/{cfg.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, NCE: {train_metrics['set_nce']:.4f}, "
                f"mu: {train_metrics['mu']:.4f}, Var: {train_metrics['var']:.4f}, "
                f"Cover+: {train_metrics['cover_pos']:.4f}, Cover-: {train_metrics['cover_neg']:.4f}, "
                f"Cov: {train_metrics['cov']:.4f}, "
                f"Reg: {train_metrics['reg']:.4f}, σ²img: {train_metrics['img_var_avg']:.4f}, "
                f"FloorR: {train_metrics.get('floor_ratio', 0):.3f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, NCE: {val_metrics['set_nce']:.4f}, "
                f"σ²img: {val_metrics['img_var_avg']:.4f}, "
                f"MCDisp_Align R@1/5/10: {val_metrics.get('mcdisp_align_recall@1', 0):.3f}/"
                f"{val_metrics.get('mcdisp_align_recall@5', 0):.3f}/{val_metrics.get('mcdisp_align_recall@10', 0):.3f}, "
                f"mR(mc): {val_metrics.get('mr', 0):.3f}, "
                f"Cos R@1/5/10: {val_metrics.get('cos_recall@1', 0):.3f}/"
                f"{val_metrics.get('cos_recall@5', 0):.3f}/{val_metrics.get('cos_recall@10', 0):.3f}"
            )

            if train_metrics.get("floor_severe", 0) > 0:
                log.warning(
                    f"{prefix}SEVERE: variance floor collapse at epoch {epoch + 1} "
                    f"(stage={mult.get('stage')}): {train_metrics['floor_severe']} batches had "
                    f">{config.MCDISP_ALIGN_VAR_FLOOR_RATIO_WARN:.0%} of dims near the floor "
                    f"(σ²img={train_metrics['img_var_avg']:.4f}, mean FloorR={train_metrics['floor_ratio']:.3f}). "
                    f"Do NOT raise MCDISP_ALIGN_VAR_FLOOR -- it masks the collapse; check the loss schedule."
                )

            # Per-epoch diagnostics (P1): weighted contributions, variance/cov stats, per-head grad norms
            log.info(
                f"{prefix}  diag: weighted[NCE={train_metrics['weighted_set_nce']:.3f} "
                f"mu={train_metrics['weighted_mu']:.3f} var={train_metrics['weighted_var']:.3f} "
                f"c+={train_metrics['weighted_cover_pos']:.3f} c-={train_metrics['weighted_cover_neg']:.3f} "
                f"cov={train_metrics['weighted_cov']:.3f} reg={train_metrics['weighted_reg']:.3f}] | "
                f"σ²[min/med/mean/max]={train_metrics['img_var_min']:.4f}/{train_metrics['img_var_median']:.4f}/"
                f"{train_metrics['img_var_mean']:.4f}/{train_metrics['img_var_max']:.4f} | "
                f"s²[mean/med/max]={train_metrics['caption_spread_mean']:.4f}/"
                f"{train_metrics['caption_spread_median']:.4f}/{train_metrics['caption_spread_max']:.4f} "
                f"txtσ²={train_metrics['text_var_mean']:.4f} | U/diag={train_metrics['u_over_diag']:.2f} | "
                f"grad[μv={train_metrics['img_mu_head_grad_norm']:.3f} "
                f"σ²v={train_metrics['img_logvar_head_grad_norm']:.3f} "
                f"Uv={train_metrics.get('img_cov_head_grad_norm', 0):.3f} "
                f"μt={train_metrics['text_mu_head_grad_norm']:.3f} "
                f"σ²t={train_metrics['text_logvar_head_grad_norm']:.3f}] "
                f"gnorm[{train_metrics['global_grad_norm_before_clip']:.2f}->{train_metrics['global_grad_norm_after_clip']:.2f}]"
            )

            # Best-checkpoint selection
            if cfg.select_by == "recall" and "mcdisp_align_recall@1" in val_metrics:
                current_score = val_metrics["mcdisp_align_recall@1"]
                improved = current_score > best_recall
                if improved:
                    best_recall = current_score
            elif cfg.select_by == "mr" and "mr" in val_metrics:
                # Ablation study: unified multi-caption development mR (plan §8).
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
                model.save(str(best_path))
                score_str = (f"mcdisp_align_recall@1: {best_recall:.4f}" if cfg.select_by == "recall"
                             else f"val_loss: {best_val_loss:.4f}")
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
            "loss_name": cfg.loss_name,  # objective tag for the resume guard
        }
        torch.save(final_state, str(last_path))
        log.info(f"{prefix}Final model saved to {last_path}")
        log.info(f"{prefix}Best val loss: {best_val_loss:.4f} | Best MCDisp_Align recall@1: {best_recall:.4f} "
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
            "mcdisp_align_recall": {f"R@{k}": eval_metrics.get(f"mcdisp_align_recall@{k}", 0.0)
                            for k in cfg.recall_k_values},
            "cos_recall": {f"R@{k}": eval_metrics.get(f"cos_recall@{k}", 0.0)
                           for k in cfg.recall_k_values},
        }
        log.info(f"{prefix}Final MCDisp_Align R@1/5/10: "
                 f"{eval_metrics.get('mcdisp_align_recall@1', 0):.3f}/"
                 f"{eval_metrics.get('mcdisp_align_recall@5', 0):.3f}/"
                 f"{eval_metrics.get('mcdisp_align_recall@10', 0):.3f}")

    log.info(f"{prefix}Training completed!")
    return results
