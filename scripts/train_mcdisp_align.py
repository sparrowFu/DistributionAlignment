"""
MCDisp_Align Training Script (CLI)

Thin command-line wrapper that parses CLI args into a training config and invokes the shared trainer (warmup ramp, grad clipping, recall/loss best-checkpoint selection, opt-in early stopping, best+last checkpointing, resume).

Trains the MCDisp_Align (Multi-Caption Semantic Dispersion Guided Distribution
Alignment) model per the paper's §3 (docs/MCDisp_Align/iclr2027_conference.tex):
each image and each of its K captions is encoded as a Gaussian; the K
per-caption distributions form ONE text distribution by moment matching
(Sigma_bar_t = S_t + mean Sigma_k^t); the image distribution
(Sigma_v = diag(sigma_v^2) + U_v U_v^T) is aligned with it through the
four-group objective  lambda_match*L_match + lambda_mu*L_mu +
(lambda_var*L_var + lambda_reg*R_prior) + lambda_dir*L_dir
(gaussian-overlap distribution-to-set contrastive, raw-coordinate center
alignment, log-space full-marginal variance regression, weak caption-variance
prior, subspace alignment with the caption variation directions).

Checkpoint selection is by the multi-caption plain-cosine Recall@1, so the
selection metric and the reported metric agree.
"""

import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.dataset_registry import VALID_DATASETS
from utils.mcdisp_align_trainer import MCDispAlignTrainConfig, run_mcdisp_align_training
from utils.logger import get_logger, log_exception


logger = get_logger("train_mcdisp_align", config.TRAIN_MCDISP_ALIGN_LOG_PATH)

# Exclude faulty CPU cores (e.g. unstable CPU 2) before DataLoader workers and
# torch threads are created. Inherited by forked worker processes.
from utils.cpu_affinity import apply_cpu_affinity
apply_cpu_affinity()


def parse_args():
    parser = argparse.ArgumentParser(description="Train MCDisp_Align Model")

    parser.add_argument("--captions-path", type=str, default=None,
                        help="Path to captions parquet file (coco only; uses config default if None)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Path to images directory (coco only; uses config default if None)")
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=list(VALID_DATASETS),
                        help="Training dataset: selects both the training data and the "
                             "checkpoint-name tag (coco=MSCOCO, flickr=flickr30k)")

    parser.add_argument("--epochs", type=int, default=config.MCDISP_ALIGN_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.MCDISP_ALIGN_BATCH_SIZE,
                        help="Training batch size")
    parser.add_argument("--clip-lr", type=float, default=config.MCDISP_ALIGN_CLIP_LR,
                        help="Learning rate for CLIP (if fine-tuning)")
    parser.add_argument("--mlp-lr", type=float, default=config.MCDISP_ALIGN_MLP_LR,
                        help="Learning rate for MLP / covariance heads")
    parser.add_argument("--weight-decay", type=float, default=config.MCDISP_ALIGN_WEIGHT_DECAY,
                        help="Weight decay")
    parser.add_argument("--lr-scheduler", type=str, default=config.LR_SCHEDULER,
                        choices=["none", "cosine"],
                        help="LR schedule: 'cosine' (cosine + linear warmup) or 'none' "
                             "(constant LR). Scales both clip-lr and mlp-lr param groups.")
    parser.add_argument("--warmup-epochs", type=int, default=config.LR_WARMUP_EPOCHS,
                        help="Linear warmup epochs for the cosine schedule (0 disables warmup)")
    parser.add_argument("--min-lr-ratio", type=float, default=config.LR_MIN_LR_RATIO,
                        help="Cosine floor as a fraction of the base LR")

    parser.add_argument("--lambda-match", type=float, default=config.MCDISP_ALIGN_LAMBDA_MATCH,
                        help="weight of L_match (distribution-to-set bidirectional contrastive)")
    parser.add_argument("--lambda-cov", type=float, default=config.MCDISP_ALIGN_LAMBDA_COV,
                        help="weight of L_cov (caption-mean containment in the image "
                             "confidence ellipsoid; match group)")
    parser.add_argument("--cov-alpha", type=float, default=config.MCDISP_ALIGN_COV_ALPHA,
                        help="confidence level alpha of the L_cov ellipsoid "
                             "(q_alpha = chi-square alpha-quantile with D dof)")
    parser.add_argument("--lambda-mu", type=float, default=config.MCDISP_ALIGN_LAMBDA_MU,
                        help="weight of L_mu (raw-coordinate center alignment)")
    parser.add_argument("--lambda-var", type=float, default=config.MCDISP_ALIGN_LAMBDA_VAR,
                        help="weight of the variance alignment L_var (core)")
    parser.add_argument("--lambda-reg", type=float, default=config.MCDISP_ALIGN_LAMBDA_REG,
                        help="weight of R_prior (weak caption-variance prior; ex --lambda-cal)")
    parser.add_argument("--lambda-dir", type=float, default=config.MCDISP_ALIGN_LAMBDA_DIR,
                        help="weight of the direction alignment L_dir")
    parser.add_argument("--tau", type=float, default=config.MCDISP_ALIGN_TAU,
                        help="Fixed temperature in the cosine match score / retrieval scoring (not learnable)")
    parser.add_argument("--tau-match", type=float, default=config.MCDISP_ALIGN_TAU_MATCH,
                        help="Fixed temperature of the gaussian overlap match logits (per-dim normalized scores)")
    parser.add_argument("--match-score", type=str, default=config.MCDISP_ALIGN_MATCH_SCORE,
                        choices=["gaussian", "cosine"],
                        help="L_match score: 'gaussian' pairwise overlap (default; also "
                             "supervises the variances) or 'cosine' of the means (ablation baseline)")
    parser.add_argument("--lambda-ctr", type=float, default=None, dest="lambda_ctr",
                        help="[DEPRECATED] alias of --lambda-match; overrides it when given")
    parser.add_argument("--lambda-cal", type=float, default=None, dest="lambda_cal",
                        help="[DEPRECATED] alias of --lambda-reg; overrides it when given")
    parser.add_argument("--sigma0-sq", type=float, default=config.MCDISP_ALIGN_SIGMA0_SQ,
                        help="caption-variance prior sigma_0^2 for R_prior")
    parser.add_argument("--warmup-frac", type=float, default=config.MCDISP_ALIGN_WARMUP_FRAC,
                        help="L_var/L_dir ramp linearly 0->1 over this fraction of total steps")

    parser.add_argument("--cov-rank", type=int, default=config.MCDISP_ALIGN_COV_RANK,
                        help="Low-rank covariance rank r for the image side (0 = diagonal only)")
    parser.add_argument("--freeze-clip", action="store_true", default=config.MCDISP_ALIGN_FREEZE_CLIP,
                        help="Freeze CLIP parameters")
    parser.add_argument("--no-freeze-clip", action="store_false", dest="freeze_clip",
                        help="Don't freeze CLIP parameters")
    parser.add_argument("--dropout-rate", type=float, default=config.MCDISP_ALIGN_DROPOUT_RATE,
                        help="Dropout rate for MLP heads")

    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    # Validation and early stopping arguments
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation set ratio (default: 0.1)")
    parser.add_argument("--early-stop-patience", type=int, default=3,
                        help="Early stopping patience in epochs (default: 3)")
    parser.add_argument("--early-stop", action="store_true",
                        help="Opt IN to early stopping. Default is a fixed "
                             "budget: all epochs run, no early stopping.")
    parser.add_argument("--select-by", type=str, default="recall",
                        choices=["recall", "mr", "overlap", "ellip", "loss"],
                        help="Best-checkpoint selection metric: 'recall' "
                             "(multi-caption cosine mc_recall@1), 'mr' (mean over "
                             "K of cosine), 'overlap' (Gaussian-overlap scored "
                             "mc_overlap_recall@1 -- the distribution-aware "
                             "criterion where the learned variances enter the "
                             "score), 'ellip' (ellipsoid-membership scored "
                             "mc_ellip_recall@1), or 'loss' (val loss, lower "
                             "better). All recall criteria: higher better. "
                             "Default: recall")

    # Output arguments
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint directory (uses config default if None)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from "
                             "(e.g. checkpoints/mcdisp_align_coco_last.pt). "
                             "Restores model weights, optimizer state, epoch, and best_recall.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Deprecated CLI aliases: --lambda-ctr -> lambda_match, --lambda-cal ->
    # lambda_reg. An explicitly given alias overrides the new flag (same
    # precedence the loss kwargs used before their removal).
    lambda_match, lambda_reg = args.lambda_match, args.lambda_reg
    if args.lambda_ctr is not None:
        logger.warning("deprecated: --lambda-ctr -> --lambda-match (will be removed)")
        lambda_match = args.lambda_ctr
    if args.lambda_cal is not None:
        logger.warning("deprecated: --lambda-cal -> --lambda-reg (will be removed)")
        lambda_reg = args.lambda_cal

    cfg = MCDispAlignTrainConfig(
        # data
        dataset=args.dataset,
        captions_path=args.captions_path,
        images_dir=args.images_dir,
        # training
        epochs=args.epochs,
        batch_size=args.batch_size,
        clip_lr=args.clip_lr,
        mlp_lr=args.mlp_lr,
        weight_decay=args.weight_decay,
        # model
        freeze_clip=args.freeze_clip,
        cov_rank=args.cov_rank,
        dropout_rate=args.dropout_rate,
        # objective (paper §3.3, four-group: match+cov/mu/var+reg/dir)
        lambda_match=lambda_match,
        lambda_cov=args.lambda_cov,
        cov_alpha=args.cov_alpha,
        lambda_mu=args.lambda_mu,
        lambda_var=args.lambda_var,
        lambda_reg=lambda_reg,
        lambda_dir=args.lambda_dir,
        tau=args.tau,
        tau_match=args.tau_match,
        sigma0_sq=args.sigma0_sq,
        match_score=args.match_score,
        warmup_frac=args.warmup_frac,
        # schedule / selection
        lr_scheduler=args.lr_scheduler,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        select_by=args.select_by,
        early_stop_patience=args.early_stop_patience,
        no_early_stop=not args.early_stop,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        # output / resume
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        resume_path=Path(args.resume) if args.resume else None,
    )

    run_mcdisp_align_training(cfg, logger)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "Training failed")
        raise
