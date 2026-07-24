"""
GaussianImageDistribution - Shared MSDA (dist_align) training orchestration.

Both ``scripts/train_dist_align.py`` (full training) and ``scripts/run_ablation.py``
(ablation variants) call :func:`run_dist_align_training`, so an ablation variant
is trained with EXACTLY the same logic as the real model — staged loss schedule,
grad-norm clipping, recall/loss best-checkpoint selection, early stopping,
best+last checkpointing, and resume — differing only in the loss weights /
``cov_rank`` / ``num_captions`` / ``dataset`` / checkpoint paths passed via
:class:`DistAlignTrainConfig`.

The per-epoch functions (:func:`train_epoch`, :func:`evaluate`,
:func:`create_optimizer`, :func:`stage_multipliers`) are also exported for direct
unit testing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import config
from data.caption_dataset import filter_none_collate
from losses.dist_align_losses import MSDALoss
from models.dist_align_model import DistributionAlignmentModel
from utils.dataset_factory import build_train_dataset
from utils.eval_common import build_eval_dataloader
from utils.lr_scheduler import apply_lr_for_epoch
from utils.retrieval import (
    compute_recall_bidirectional,
    compute_recall_msda_chunked,
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable)
# ---------------------------------------------------------------------------

def create_optimizer(
    model: DistributionAlignmentModel,
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


def stage_multipliers(epoch: int, total: int, no_staged: bool) -> Dict[str, float]:
    """Per-loss multipliers for the staged MSDA schedule.

    Warm-up: L_set + L_mu (+ L_reg always on). Main: + L_var + L_cover.
    Full: linearly ramp L_cov 0 -> 1 (the cov head is untrained before this;
    a hard step would dominate and destabilize the retrieval means).
    L_reg is always on (pure stabilizer).
    """
    base = {"ctr": 1.0, "mu": 1.0, "var": 1.0, "cover": 1.0, "cov": 1.0, "reg": 1.0}
    if no_staged or total <= 0:
        return base
    warmup_end = max(1, int(round(total * config.MSDA_STAGE_WARMUP_FRAC)))
    main_end = max(warmup_end + 1,
                   int(round(total * (config.MSDA_STAGE_WARMUP_FRAC + config.MSDA_STAGE_MAIN_FRAC))))
    if epoch < warmup_end:
        base.update(var=0.0, cover=0.0, cov=0.0)
    elif epoch < main_end:
        base.update(cov=0.0)
    else:
        full_len = max(1, total - main_end)
        j = epoch - main_end
        base["cov"] = min(1.0, (j + 1) / full_len)
    return base


# ---------------------------------------------------------------------------
# Epoch functions (ported verbatim from the original train_dist_align.py)
# ---------------------------------------------------------------------------

def train_epoch(
    model: DistributionAlignmentModel,
    dataloader: DataLoader,
    criterion: MSDALoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    desc_prefix: str = "",
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    if not model.freeze_clip:
        model.clip_model.train()

    totals = {k: 0.0 for k in
              ("loss", "set_nce", "mu", "var", "cover", "cov", "reg", "img_var_avg")}
    processed_batches = 0

    desc = f"{desc_prefix}Epoch {epoch + 1}" if desc_prefix else f"Epoch {epoch + 1}"
    pbar = tqdm(dataloader, desc=desc)

    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue
        processed_batches += 1

        # Get data - PIL images and text lists
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

        # Forward pass
        outputs = model(pixel_values, input_ids, attention_mask)

        # Compute MSDA loss
        loss, loss_dict = criterion(
            outputs['img_mu'], outputs['img_logvar'], outputs['img_U'],
            outputs['text_mu'], outputs['text_logvar'],
            outputs['text_mus'], outputs['text_logvars'], outputs.get('text_Us'),
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Clip global grad norm to protect against L_cov / cover spikes that can
        # destabilize the retrieval means.
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MSDA_GRAD_CLIP_NORM)
        optimizer.step()

        # Accumulate losses
        for k in totals:
            kk = "loss" if k == "loss" else k
            src = "total" if k == "loss" else k
            totals[k] += loss_dict[src]

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'NCE': f"{loss_dict['set_nce']:.4f}",
            'mu': f"{loss_dict['mu']:.3f}",
            'var': f"{loss_dict['var']:.3f}",
            'cov': f"{loss_dict['cov']:.3f}",
            'σ²i': f"{loss_dict['img_var_avg']:.3f}",
        })

    num_batches = max(processed_batches, 1)
    return {k: v / num_batches for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: DistributionAlignmentModel,
    dataloader: DataLoader,
    criterion: MSDALoss,
    device: torch.device,
    compute_recall: bool = False,
    recall_k_values=None,
) -> Dict[str, float]:
    """Evaluate the model (loss + optional retrieval Recall@K)."""
    model.eval()

    totals = {k: 0.0 for k in
              ("loss", "set_nce", "mu", "var", "cover", "cov", "reg", "img_var_avg")}
    processed_batches = 0
    feats = {k: [] for k in ("img_mu", "text_mu", "img_logvar", "text_logvar")} \
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

        if feats is not None:
            feats["img_mu"].append(outputs['img_mu'].cpu())
            feats["text_mu"].append(outputs['text_mu'].cpu())
            feats["img_logvar"].append(outputs['img_logvar'].cpu())
            feats["text_logvar"].append(outputs['text_logvar'].cpu())

        pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

    num_batches = max(processed_batches, 1)
    metrics = {k: v / num_batches for k, v in totals.items()}

    # Retrieval Recall@K (image<->text, diagonal pairing). The val loader uses
    # shuffle=False, so concatenated img_mu[i] stays aligned with its own
    # caption-set text_mu[i] -> the diagonal is the positive pair.
    # Primary score: MSDA uncertainty-discounted cosine (= what L_set optimizes).
    # Secondary: plain cosine-on-means (methodology's mean-only retrieval mode).
    if compute_recall and recall_k_values and feats and feats["img_mu"]:
        img_mu = torch.cat(feats["img_mu"], dim=0).to(device)
        text_mu = torch.cat(feats["text_mu"], dim=0).to(device)
        img_lv = torch.cat(feats["img_logvar"], dim=0).to(device)
        text_lv = torch.cat(feats["text_logvar"], dim=0).to(device)
        msda = compute_recall_msda_chunked(
            img_mu, img_lv, text_mu, text_lv, recall_k_values, tau=criterion.tau)
        metrics.update(msda)
        cos = compute_recall_bidirectional(img_mu, text_mu, recall_k_values, normalize=True)
        for k in recall_k_values:
            metrics[f"cos_recall@{k}"] = (cos[f"recall_i2t@{k}"] + cos[f"recall_t2i@{k}"]) / 2

    return metrics


# ---------------------------------------------------------------------------
# Configuration + orchestrator
# ---------------------------------------------------------------------------

@dataclass
class DistAlignTrainConfig:
    """All knobs for one MSDA training run (full or ablation)."""

    # --- Data ---
    dataset: str = "coco"
    num_workers: int = field(default_factory=lambda: config.NUM_WORKERS)
    val_split: float = 0.1
    num_captions_override: Optional[int] = None   # ablation k1/k3/k5; None -> dataset default
    captions_path: Optional[str] = None           # coco override
    images_dir: Optional[str] = None              # coco override

    # --- Training ---
    epochs: int = field(default_factory=lambda: config.DIST_ALIGN_EPOCHS)
    batch_size: int = field(default_factory=lambda: config.DIST_ALIGN_BATCH_SIZE)
    clip_lr: float = field(default_factory=lambda: config.DIST_ALIGN_CLIP_LR)
    mlp_lr: float = field(default_factory=lambda: config.DIST_ALIGN_MLP_LR)
    weight_decay: float = field(default_factory=lambda: config.DIST_ALIGN_WEIGHT_DECAY)

    # --- Model ---
    freeze_clip: bool = field(default_factory=lambda: config.DIST_ALIGN_FREEZE_CLIP)
    cov_rank: int = field(default_factory=lambda: config.MSDA_COV_RANK)
    distribution_merging: str = field(default_factory=lambda: config.DIST_ALIGN_DISTRIBUTION_MERGING)
    dropout_rate: float = field(default_factory=lambda: config.DIST_ALIGN_DROPOUT_RATE)

    # --- MSDA loss weights ---
    lambda_ctr: float = field(default_factory=lambda: config.MSDA_LAMBDA_CTR)
    lambda_mu: float = field(default_factory=lambda: config.MSDA_LAMBDA_MU)
    lambda_var: float = field(default_factory=lambda: config.MSDA_LAMBDA_VAR)
    lambda_cover: float = field(default_factory=lambda: config.MSDA_LAMBDA_COVER)
    lambda_cov: float = field(default_factory=lambda: config.MSDA_LAMBDA_COV)
    lambda_reg: float = field(default_factory=lambda: config.MSDA_LAMBDA_REG)
    tau: float = field(default_factory=lambda: config.MSDA_TAU)
    m_pos: float = field(default_factory=lambda: config.MSDA_M_POS)
    m_neg: float = field(default_factory=lambda: config.MSDA_M_NEG)
    target_var: float = field(default_factory=lambda: config.MSDA_TARGET_VAR)
    use_uncertainty_sim: bool = field(default_factory=lambda: config.MSDA_USE_UNCERTAINTY_SIM)

    # --- Schedule / selection ---
    no_staged: bool = False
    lr_scheduler: str = field(default_factory=lambda: config.LR_SCHEDULER)
    warmup_epochs: int = field(default_factory=lambda: config.LR_WARMUP_EPOCHS)
    min_lr_ratio: float = field(default_factory=lambda: config.LR_MIN_LR_RATIO)
    select_by: str = "recall"                     # "recall" (MSDA R@1) or "loss"
    early_stop_patience: int = 3
    no_early_stop: bool = False
    seed: int = field(default_factory=lambda: config.SEED)

    # --- System / output ---
    device: str = "cuda"
    tag: str = ""                                  # log/tqdm prefix (e.g. ablation name)
    checkpoint_dir: Optional[Path] = None          # standard naming root
    model_name: str = "dist_align"                 # -> {model_name}_{dataset}_best|last.pt
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


def _build_loaders(cfg: DistAlignTrainConfig):
    """Build train/val DataLoaders from the registry-selected training dataset."""
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


def run_dist_align_training(cfg: DistAlignTrainConfig, log) -> Dict:
    """Run a full MSDA training (+ optional final retrieval eval).

    Returns a dict with ``best_val_loss``, ``best_recall``, ``select_by``, the
    last epoch index, and (when ``cfg.eval_num_samples`` is set) ``retrieval``
    metrics on the dataset's eval split — the same scorers evaluate_dist_align uses.
    """
    from utils.seed import set_seed
    set_seed(cfg.seed)

    prefix = f"[{cfg.tag}] " if cfg.tag else ""

    log.info("=" * 60)
    log.info(f"{prefix}MSDA Distribution Alignment Training")
    log.info("=" * 60)
    log.info(f"{prefix}Dataset: {cfg.dataset} | Epochs: {cfg.epochs} | Batch: {cfg.batch_size}")
    log.info(f"{prefix}CLIP LR: {cfg.clip_lr} | MLP LR: {cfg.mlp_lr} | Freeze CLIP: {cfg.freeze_clip}")
    log.info(f"{prefix}Cov rank r: {cfg.cov_rank} | Tau (fixed): {cfg.tau} | Staged: {not cfg.no_staged}")
    log.info(f"{prefix}Loss weights: ctr={cfg.lambda_ctr} mu={cfg.lambda_mu} var={cfg.lambda_var} "
             f"cover={cfg.lambda_cover} cov={cfg.lambda_cov} reg={cfg.lambda_reg}")
    log.info(f"{prefix}Select by: {cfg.select_by} | Device: {cfg.device}")
    log.info("=" * 60)

    # Model
    log.info(f"{prefix}Creating model...")
    model = DistributionAlignmentModel(
        freeze_clip=cfg.freeze_clip,
        distribution_merging=cfg.distribution_merging,
        dropout_rate=cfg.dropout_rate,
        cov_rank=cfg.cov_rank,
    )
    model = model.to(cfg.device)
    log.info(f"{prefix}Model created with {model.num_trainable_parameters():,} trainable parameters")

    # Loss
    criterion = MSDALoss(
        lambda_ctr=cfg.lambda_ctr,
        lambda_mu=cfg.lambda_mu,
        lambda_var=cfg.lambda_var,
        lambda_cover=cfg.lambda_cover,
        lambda_cov=cfg.lambda_cov,
        lambda_reg=cfg.lambda_reg,
        tau=cfg.tau,
        m_pos=cfg.m_pos,
        target_var=cfg.target_var,
        m_neg=cfg.m_neg,
        use_uncertainty_sim=cfg.use_uncertainty_sim,
    )
    log.info(f"{prefix}Using MSDA loss (uncertainty-discounted cosine L_set)")
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
            criterion.lambda_mu = cfg.lambda_mu * mult["mu"]
            criterion.lambda_var = cfg.lambda_var * mult["var"]
            criterion.lambda_cover = cfg.lambda_cover * mult["cover"]
            criterion.lambda_cov = cfg.lambda_cov * mult["cov"]
            criterion.lambda_reg = cfg.lambda_reg * mult["reg"]
            log.info(f"{prefix}Epoch {epoch + 1} stage multipliers: {mult}")

            apply_lr_for_epoch(optimizer, base_lrs, epoch, cfg.epochs,
                               cfg.warmup_epochs, cfg.min_lr_ratio,
                               cfg.lr_scheduler, log)

            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, cfg.device, epoch,
                desc_prefix=cfg.tag,
            )
            val_metrics = evaluate(
                model, val_loader, criterion, cfg.device,
                compute_recall=(cfg.select_by == "recall"),
                recall_k_values=list(cfg.recall_k_values),
            )

            log.info(
                f"{prefix}Epoch {epoch + 1}/{cfg.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, NCE: {train_metrics['set_nce']:.4f}, "
                f"mu: {train_metrics['mu']:.4f}, Var: {train_metrics['var']:.4f}, "
                f"Cover: {train_metrics['cover']:.4f}, Cov: {train_metrics['cov']:.4f}, "
                f"Reg: {train_metrics['reg']:.4f}, σ²img: {train_metrics['img_var_avg']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, NCE: {val_metrics['set_nce']:.4f}, "
                f"σ²img: {val_metrics['img_var_avg']:.4f}, "
                f"MSDA R@1/5/10: {val_metrics.get('msda_recall@1', 0):.3f}/"
                f"{val_metrics.get('msda_recall@5', 0):.3f}/{val_metrics.get('msda_recall@10', 0):.3f}, "
                f"Cos R@1/5/10: {val_metrics.get('cos_recall@1', 0):.3f}/"
                f"{val_metrics.get('cos_recall@5', 0):.3f}/{val_metrics.get('cos_recall@10', 0):.3f}"
            )

            # Best-checkpoint selection
            if cfg.select_by == "recall" and "msda_recall@1" in val_metrics:
                current_score = val_metrics["msda_recall@1"]
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
                score_str = (f"msda_recall@1: {best_recall:.4f}" if cfg.select_by == "recall"
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
        }
        torch.save(final_state, str(last_path))
        log.info(f"{prefix}Final model saved to {last_path}")
        log.info(f"{prefix}Best val loss: {best_val_loss:.4f} | Best MSDA recall@1: {best_recall:.4f} "
                 f"(selected by: {cfg.select_by})")

    results = {
        "best_val_loss": best_val_loss,
        "best_recall": best_recall,
        "select_by": cfg.select_by,
        "last_epoch": last_epoch,
        "best_checkpoint": str(best_path),
    }

    # Optional final retrieval eval on the dataset's eval split (ablation reporting),
    # using the SAME scorers as evaluate_dist_align.
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
            "msda_recall": {f"R@{k}": eval_metrics.get(f"msda_recall@{k}", 0.0)
                            for k in cfg.recall_k_values},
            "cos_recall": {f"R@{k}": eval_metrics.get(f"cos_recall@{k}", 0.0)
                           for k in cfg.recall_k_values},
        }
        log.info(f"{prefix}Final MSDA R@1/5/10: "
                 f"{eval_metrics.get('msda_recall@1', 0):.3f}/"
                 f"{eval_metrics.get('msda_recall@5', 0):.3f}/"
                 f"{eval_metrics.get('msda_recall@10', 0):.3f}")

    log.info(f"{prefix}Training completed!")
    return results
