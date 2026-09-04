"""
MCDisp_Align Training Script (CLI)

Thin command-line wrapper that parses CLI args into a training config and invokes the shared trainer (staged loss schedule, grad clipping, recall/loss best-checkpoint selection, early stopping, best+last checkpointing, resume).

Trains the MCDisp_Align (Multi-Caption Semantic Dispersion Guided Distribution Alignment) model, which
models image and text embeddings as Gaussians. The image uses a general
covariance Sigma_v = diag(sigma_v^2) + U_v U_v^T; text is diagonal-only (v1).
The image variance is supervised toward the multi-caption semantic spread, and
the image low-rank directions toward the caption deviation directions.

Checkpoint selection is by the MCDisp_Align uncertainty-discounted cosine Recall@1
(the same score L_set optimizes), so the trained objective, the selection
metric and the reported metric all agree.
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

    parser.add_argument("--lambda-ctr", type=float, default=config.MCDISP_ALIGN_LAMBDA_CTR)
    parser.add_argument("--lambda-mu", type=float, default=config.MCDISP_ALIGN_LAMBDA_MU)
    parser.add_argument("--lambda-var", type=float, default=config.MCDISP_ALIGN_LAMBDA_VAR)
    parser.add_argument("--lambda-cover-pos", type=float, default=config.MCDISP_ALIGN_LAMBDA_COVER_POS)
    parser.add_argument("--lambda-cover-neg", type=float, default=config.MCDISP_ALIGN_LAMBDA_COVER_NEG,
                        help="weight of L_cover's optional negative-repulsion (0 = pos-only)")
    parser.add_argument("--lambda-cov", type=float, default=config.MCDISP_ALIGN_LAMBDA_COV)
    parser.add_argument("--lambda-reg", type=float, default=config.MCDISP_ALIGN_LAMBDA_REG)
    parser.add_argument("--tau", type=float, default=config.MCDISP_ALIGN_TAU,
                        help="Fixed temperature in the L_set similarity (not learnable)")
    parser.add_argument("--m-pos", type=float, default=config.MCDISP_ALIGN_M_POS,
                        help="L_cover positive coverage margin (per-D normalized Mahalanobis)")
    parser.add_argument("--target-var", type=float, default=config.MCDISP_ALIGN_TARGET_VAR,
                        help="L_reg variance prior sigma_0^2")
    parser.add_argument("--m-neg", type=float, default=config.MCDISP_ALIGN_M_NEG,
                        help="L_cover negative repulsion margin")
    parser.add_argument("--use-uncertainty-sim", action="store_true",
                        default=config.MCDISP_ALIGN_USE_UNCERTAINTY_SIM,
                        help="L_set uses the uncertainty-discounted score (default)")
    parser.add_argument("--no-uncertainty-sim", dest="use_uncertainty_sim",
                        action="store_false",
                        help="L_set uses plain cosine (ablation)")

    parser.add_argument("--loss", type=str, default="standard",
                        choices=["standard", "kl"],
                        help="Training loss: 'standard' = MCDispAlignLoss "
                             "(separate L_mu/L_var terms, the original "
                             "objective, default); 'kl' = MCDispAlignKLLoss "
                             "(L_mu+L_var folded into one KL(p_v||p_t) term; "
                             "--lambda-kl then replaces --lambda-mu/--lambda-var)")
    parser.add_argument("--lambda-kl", type=float, default=1.0,
                        help="Weight of the KL alignment term (only used with --loss kl)")

    parser.add_argument("--cov-rank", type=int, default=config.MCDISP_ALIGN_COV_RANK,
                        help="Low-rank covariance rank r for the image side (0 = diagonal only)")
    parser.add_argument("--freeze-clip", action="store_true", default=config.MCDISP_ALIGN_FREEZE_CLIP,
                        help="Freeze CLIP parameters")
    parser.add_argument("--no-freeze-clip", action="store_false", dest="freeze_clip",
                        help="Don't freeze CLIP parameters")
    parser.add_argument("--distribution-merging", type=str, default=config.MCDISP_ALIGN_DISTRIBUTION_MERGING,
                        choices=["moment_matching", "poe", "simple"],
                        help="Method for merging multiple text distributions")
    parser.add_argument("--dropout-rate", type=float, default=config.MCDISP_ALIGN_DROPOUT_RATE,
                        help="Dropout rate for MLP heads")
    parser.add_argument("--no-staged", action="store_true",
                        help="Disable 3-stage schedule; use all losses from epoch 1")

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
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping")
    parser.add_argument("--select-by", type=str, default="recall",
                        choices=["recall", "loss"],
                        help="Best-checkpoint selection metric: 'recall' "
                             "(MCDisp_Align R@1, higher better) or 'loss' "
                             "(val loss, lower better). Default: recall")

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
        distribution_merging=args.distribution_merging,
        dropout_rate=args.dropout_rate,
        # loss weights
        lambda_ctr=args.lambda_ctr,
        lambda_mu=args.lambda_mu,
        lambda_var=args.lambda_var,
        loss_name=args.loss,
        lambda_kl=args.lambda_kl,
        lambda_cover_pos=args.lambda_cover_pos,
        lambda_cover_neg=args.lambda_cover_neg,
        lambda_cov=args.lambda_cov,
        lambda_reg=args.lambda_reg,
        tau=args.tau,
        m_pos=args.m_pos,
        m_neg=args.m_neg,
        target_var=args.target_var,
        use_uncertainty_sim=args.use_uncertainty_sim,
        # schedule / selection
        no_staged=args.no_staged,
        lr_scheduler=args.lr_scheduler,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        select_by=args.select_by,
        early_stop_patience=args.early_stop_patience,
        no_early_stop=args.no_early_stop,
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
