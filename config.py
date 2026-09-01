"""Centralized configuration: project paths and hyperparameters."""

import platform
from pathlib import Path

# =============================================================================
# Platform Detection
# =============================================================================
IS_WINDOWS = platform.system() == "Windows"

# =============================================================================
# Project Root - MODIFY THIS when migrating to a new system
# =============================================================================
# Auto-detect project root based on platform
if IS_WINDOWS:
    PROJECT_ROOT = Path("D:/code/causality/GaussianImageDistribution")
else:
    # Ubuntu/Linux default - modify "your_name" to your actual username
    PROJECT_ROOT = Path("/home/xpfu/WorkSpace/DistributionAlignment")

# =============================================================================
# Dataset Paths
# =============================================================================
# Parquet file containing image-caption pairs
# Format: DatasetDict with features ["url", "caption", "image_file_name"]
CAPTIONS_PATH = PROJECT_ROOT / "TrainDatasets" / "mscoco_captions" / "captions" / "train-00000-of-00001.parquet"

# Directory containing all images
IMAGES_DIR = PROJECT_ROOT / "TrainDatasets" / "mscoco_captions" / "images"

# =============================================================================
# Pre-trained Model Paths (Local Only)
# =============================================================================
# CLIP ViT-Large-Patch14 model directory
CLIP_VIT_L_14_PATH = PROJECT_ROOT / "PreTrainedModels" / "clip-vit-large-patch14"

# =============================================================================
# Output Directories
# =============================================================================
# Checkpoint directory for saving model weights
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# General output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Log directory
LOG_DIR = PROJECT_ROOT / "logs"

# =============================================================================
# Log File Paths
# =============================================================================
# Main application log
MAIN_LOG_PATH = LOG_DIR / "main.log"

# Training log for CLIP baseline
TRAIN_CLIP_BASELINE_LOG_PATH = LOG_DIR / "train_clip_baseline.log"

# Evaluation log for CLIP baseline
EVAL_CLIP_BASELINE_LOG_PATH = LOG_DIR / "evaluate_clip_baseline.log"

# Evaluation log for CLIP zero-shot
EVAL_CLIP_ZERO_SHOT_LOG_PATH = LOG_DIR / "evaluate_clip_zero_shot.log"

# =============================================================================
# Checkpoint Paths
# =============================================================================
# Best checkpoint (lowest validation loss)
CLIP_BASELINE_BEST_CKPT = CHECKPOINT_DIR / "clip_baseline_coco_best.pt"

# =============================================================================
# Evaluation Results Paths
# =============================================================================
# JSON file for evaluation results
CLIP_BASELINE_EVAL_RESULTS_PATH = OUTPUT_DIR / "clip_baseline_eval_results.json"

# JSON file for CLIP zero-shot evaluation results
CLIP_ZERO_SHOT_EVAL_RESULTS_PATH = OUTPUT_DIR / "clip_zero_shot_eval_results.json"

# =============================================================================
# Random Seed
# =============================================================================
SEED = 42

# =============================================================================
# Learning-Rate Scheduling (shared across all training scripts)
# =============================================================================
# Cosine annealing with linear warmup, applied per-epoch.
# LR_SCHEDULER: "cosine" (default) or "none" for a constant LR (legacy behavior).
# LR_WARMUP_EPOCHS: linear warmup length (0 disables warmup, pure cosine).
# LR_MIN_LR_RATIO: cosine floor as a fraction of the base LR (e.g. 5e-5 -> 1e-6).
LR_SCHEDULER = "cosine"
LR_WARMUP_EPOCHS = 1
LR_MIN_LR_RATIO = 0.02

# =============================================================================
# Dataset Settings
# =============================================================================
# Number of captions to use per image (minimum 5, will pad if necessary)
NUM_CAPTIONS = 5

# Number of worker processes for data loading
# Set to 0 for single-process data loading (no multiprocessing/threads)
# This ensures maximum compatibility across all platforms
# Linux: 8 workers for parallel data loading; Windows: 0 to avoid fork issues
NUM_WORKERS = 0 if IS_WINDOWS else 8

# =============================================================================
# CPU Affinity (server-specific)
# =============================================================================
# Faulty CPU cores to exclude from this process. CPU 2 on this server is
# unstable: when the training process or a DataLoader worker is scheduled onto
# it, the run crashes with SIGSEGV (exit code -11) at a random point mid-run.
# Listed cores are removed from the process CPU affinity at startup via
# utils.cpu_affinity.apply_cpu_affinity (no-op on Windows). Adjust per machine.
EXCLUDED_CPUS = [2]

# =============================================================================
# CLIP Baseline Training Hyperparameters
# =============================================================================
# Number of training epochs
CLIP_BASELINE_EPOCHS = 5

# Training batch size
CLIP_BASELINE_BATCH_SIZE = 32

# Learning rate
CLIP_BASELINE_LR = 1e-6

# Weight decay for regularization
CLIP_BASELINE_WEIGHT_DECAY = 1e-4

# Temperature parameter for contrastive loss
CLIP_BASELINE_TEMPERATURE = 0.07

# Whether to freeze the image encoder
CLIP_BASELINE_FREEZE_IMAGE = False

# Whether to freeze the text encoder
CLIP_BASELINE_FREEZE_TEXT = False

# =============================================================================
# Evaluation Settings
# =============================================================================
# Recall@K values to compute
RECALL_AT_K = [1, 5, 10]

# Batch size for evaluation (can be larger than training)
EVAL_BATCH_SIZE = 64

# =============================================================================
# MCDisp_Align Model Configuration
# =============================================================================
# Training log for distribution alignment model
TRAIN_MCDISP_ALIGN_LOG_PATH = LOG_DIR / "train_mcdisp_align.log"

# Evaluation log for distribution alignment model
EVAL_MCDISP_ALIGN_LOG_PATH = LOG_DIR / "evaluate_mcdisp_align.log"

# Checkpoint paths
MCDISP_ALIGN_BEST_CKPT = CHECKPOINT_DIR / "mcdisp_align_coco_best.pt"

# Evaluation results path
MCDISP_ALIGN_EVAL_RESULTS_PATH = OUTPUT_DIR / "mcdisp_align_eval_results.json"

# Training hyperparameters
# 15 epochs fixed budget (no early stop by default): all four loss terms are on
# after a short warmup ramp, and the budget must let the from-scratch heads
# converge on frozen CLIP features.
MCDISP_ALIGN_EPOCHS = 15
# 64 (retrieval-max configuration): double the InfoNCE negatives of the 32
# regime (B*K = 320 captions in the softmax), which strengthens the
# discriminative training of the retrieval means. Chosen to maximize test-pool
# retrieval (the goal), not batch-parity with the baselines.
MCDISP_ALIGN_BATCH_SIZE = 64
MCDISP_ALIGN_CLIP_LR = 1e-6  # Learning rate for CLIP (fine-tuning mode)
MCDISP_ALIGN_MLP_LR = 5e-5   # Learning rate for MLP distribution heads (trained from scratch; balanced for convergence vs overfitting)
MCDISP_ALIGN_WEIGHT_DECAY = 1e-4
# Fine-tune CLIP alongside the heads (retrieval-max configuration): the CLIP
# baseline is itself fully fine-tuned at 1e-6 on the target dataset, so a
# frozen-feature head (a near-linear probe capped at zero-shot quality) would
# concede in-domain adaptation to it. Same clip_lr as the baseline for a
# like-for-like adaptation rate; best-checkpoint selection is by val recall,
# which guards against late-epoch drift.
MCDISP_ALIGN_FREEZE_CLIP = False

# Distribution configuration
MCDISP_ALIGN_DROPOUT_RATE = 0.1         # Dropout rate for MLP heads

# =============================================================================
# MCDisp_Align: Multi-Caption Semantic Dispersion Guided Distribution Alignment
# =============================================================================
MCDISP_ALIGN_OBJECTIVE_VERSION = "four-group-v3"   # A13: checkpoint 兼容标识 (v3 = 匹配组新增 L_cov 椭球包含约束; v2 = 四组目标重构)
# Method overview (paper §3, docs/MCDisp_Align/iclr2027_conference.tex):
# Each image and each of its K captions is encoded as a Gaussian through
# lightweight heads. The K per-caption distributions form ONE text
# distribution by moment matching (eq:caption_set_moments):
#     Sigma_bar_t = S_t + (1/K) sum_k Sigma_k^t,
#     diag(Sigma_bar_t) = s_t^2 + (1/K) sum_k sigma_k^2,
# i.e. the text variance adds the empirical caption dispersion to the averaged
# caption variance. The image distribution (Sigma_v = diag(sigma_v^2) +
# U_v U_v^T) is aligned with the text distribution through its parameters:
#
#     L = lambda_match*L_match + lambda_cov*L_cov + lambda_mu*L_mu
#       + (lambda_var*L_var + lambda_reg*R_prior) + lambda_dir*L_dir
#
# L_match : distribution-to-set bidirectional contrastive between the image
#           Gaussian and the B*K caption Gaussians. Score switchable
#           (MCDISP_ALIGN_MATCH_SCORE): "gaussian" pairwise overlap (default;
#           the match itself supervises the variances) or plain cosine
#           (cosine_match ablation baseline, means only).
# L_cov   : confidence-ellipsoid containment (match group): hinge on the
#           Mahalanobis distance of each sg caption mean to the image
#           Gaussian, charged only above the chi2_D(alpha) quantile. Zero
#           loss certifies every caption mean inside the image's
#           alpha-ellipsoid (cov_viol = violating fraction).
# L_mu    : explicit center alignment MSE(mu_v, sg[mu_t]) in RAW coordinates.
# L_var   : log-space regression of the FULL image marginal variance
#           d_v + sum_r U_r^2 to the stop-gradient text variance
#           diag(Sigma_bar_t) (encodes the empirical dispersion).
# R_prior : caption variances calibrated to the prior sigma_0^2 (weak prior;
#           renamed from L_cal).
# L_dir   : subspace alignment between U_v and the top-r eigenvectors of the
#           between-caption covariance S_t (r capped at min(r, K-1, D)),
#           guarded by the spectral rank check (MCDISP_ALIGN_DIR_EIG_REL_TOL).
MCDISP_ALIGN_COV_RANK = 4                 # low-rank covariance rank r for the IMAGE side (0 = diagonal only)
MCDISP_ALIGN_MATCH_SCORE = "gaussian"     # "gaussian"(L_match overlap score) | "cosine"(消融对照, means only)
MCDISP_ALIGN_TAU = 0.07                   # FIXED temperature in the COSINE match score / retrieval scoring (not learnable)
MCDISP_ALIGN_TAU_MATCH = 1.0              # 重叠分数的固定温度（分数逐维归一, O(1) 温度即可）
MCDISP_ALIGN_LAMBDA_MATCH = 1.0           # weight for L_match (0 in the no_match-style ablations)
MCDISP_ALIGN_LAMBDA_COV = 0.01            # weight for the containment hinge L_cov (match group; 0 in the no_cov ablation)
MCDISP_ALIGN_COV_ALPHA = 0.95             # confidence level alpha of the L_cov ellipsoid (q_alpha = chi2.ppf(alpha, D))
MCDISP_ALIGN_LAMBDA_MU = 0.5              # weight for the raw-coordinate center alignment L_mu (0 in the no_mu ablation)
MCDISP_ALIGN_LAMBDA_VAR = 1.0             # weight for the variance alignment L_var (core)
MCDISP_ALIGN_LAMBDA_REG = 0.01            # weight for the weak caption prior R_prior (原 LAMBDA_CAL 迁移)
MCDISP_ALIGN_LAMBDA_DIR = 0.5             # weight for the direction alignment L_dir
MCDISP_ALIGN_SIGMA0_SQ = 0.04             # prior sigma_0^2 for caption calibration (= measured MSCOCO caption spread)
MCDISP_ALIGN_DIR_EIG_REL_TOL = 1e-3       # A05 实际谱秩相对阈值 (eigenvalue counts iff > max_eig*tol)
MCDISP_ALIGN_WARMUP_FRAC = 0.1            # L_var/L_dir/L_cov ramp linearly 0->1 over the first 10% of total steps (caption heads train from scratch; the dispersion statistics and containment targets need a few steps to mature). L_match/L_mu/R_prior are always on.
MCDISP_ALIGN_VAR_FLOOR = 1e-4             # numerical floor on sigma^2 (softplus positivity guard; NOT a semantic floor -- the range is learned via L_var)
MCDISP_ALIGN_VAR_FLOOR_NEAR_MULT = 10     # variance floor-collapse monitor: a dim is "near floor" if sigma^2 < MCDISP_ALIGN_VAR_FLOOR_NEAR_MULT * MCDISP_ALIGN_VAR_FLOOR
MCDISP_ALIGN_VAR_FLOOR_RATIO_WARN = 0.5   # warn when >this fraction of dims are near-floor AND mean sigma^2 < 2*MCDISP_ALIGN_VAR_FLOOR
MCDISP_ALIGN_COV_EPS = 1e-6               # numerical epsilon for log / eigh
MCDISP_ALIGN_GRAD_CLIP_NORM = 1.0         # global grad-norm clip (clip_grad_norm_) -- guards against eigh/QR spikes destabilizing the retrieval means
# Deprecated likelihood-rewrite knobs (no longer used by the training/loss path).
MCDISP_ALIGN_USE_LOGDET = True            # deprecated: loglik-eval only
MCDISP_ALIGN_PER_DIM_NORMALIZE = True     # deprecated: loglik-eval only

# =============================================================================
# Baseline B3: ProLIP Configuration (real ProLIP ViT-H/14 via the `prolip` lib)
# =============================================================================
# Three local artifacts (no network needed).
#   model     : ProLIPHF weights (config.json + model.safetensors), embed_dim 1024
#   processor : CLIP image processor (openai/clip-vit-base-patch16) -- correct
#               CLIP normalization for the ViT-H/14 backbone
#   tokenizer : HFTokenizer (apple/DFN5B-CLIP-ViT-H-14), CLIP BPE, context 77
PROLIP_MODEL_PATH = PROJECT_ROOT / "PreTrainedModels" / "prolip"
PROLIP_PROCESSOR_PATH = PROJECT_ROOT / "PreTrainedModels" / "prolipProcessor"
PROLIP_TOKENIZER_PATH = PROJECT_ROOT / "PreTrainedModels" / "prolipTokenizer"
PROLIP_CONTEXT_LENGTH = 77

# ProLIP checkpoint and output paths
PROLIP_BEST_CKPT = CHECKPOINT_DIR / "prolip_coco_best.pt"
PROLIP_EVAL_RESULTS_PATH = OUTPUT_DIR / "prolip_eval_results.json"
PROLIP_ZERO_SHOT_EVAL_RESULTS_PATH = OUTPUT_DIR / "prolip_zero_shot_eval_results.json"
TRAIN_PROLIP_LOG_PATH = LOG_DIR / "train_prolip.log"
EVAL_PROLIP_LOG_PATH = LOG_DIR / "evaluate_prolip.log"
EVAL_PROLIP_ZERO_SHOT_LOG_PATH = LOG_DIR / "evaluate_prolip_zero_shot.log"

# ProLIP ViT-H/14 embedding dimension (mean / log-variance head output dim)
PROLIP_EMBED_DIM = 1024

# Fine-tuning hyperparameters (full fine-tuning of the whole ProLIP model with the ProLIP inclusion loss).
PROLIP_EPOCHS = 5
PROLIP_BATCH_SIZE = 32          # unified training regime (was 16); ViT-H/14 full
                                # fine-tune at 32 fits the 85GB GPU when idle --
                                # fall back to 16 if the card is shared
PROLIP_LR = 1e-6                # full-backbone fine-tune -> small LR
PROLIP_WEIGHT_DECAY = 1e-4
PROLIP_TEMPERATURE = 0.07       # legacy alias (inclusion loss uses learned logit_scale)

# ProLIP inclusion-loss weights
PROLIP_PPCL_LAMBDA = 1.0        # probabilistic pairwise contrastive loss (always on)
PROLIP_INCLUSION_ALPHA = 1.0    # inclusion loss weight (image subset text); 0 disables
PROLIP_VIB_BETA = 1.0e-5        # variational information bottleneck KL weight

# =============================================================================
# Flickr30K Dataset Configuration
# =============================================================================
FLICKR30K_ROOT = PROJECT_ROOT / "TrainDatasets" / "flickr30k"
FLICKR30K_IMAGES_DIR = FLICKR30K_ROOT / "flickr30k_images"
FLICKR30K_CAPTIONS_PATH = FLICKR30K_ROOT / "captions.txt"
FLICKR30K_NUM_CAPTIONS = 5

# =============================================================================
# Experiment 4: OOD Detection Configuration
# =============================================================================
OOD_RESULTS_DIR = OUTPUT_DIR / "ood_detection"
OOD_LOG_PATH = LOG_DIR / "ood_detection.log"

# OOD datasets: will be downloaded via torchvision if not present
OOD_DATA_DIR = PROJECT_ROOT / "TrainDatasets" / "ood"

# =============================================================================
# Experiment 7: σ Semantic Analysis Configuration
# =============================================================================
SIGMA_ANALYSIS_RESULTS_DIR = OUTPUT_DIR / "sigma_analysis"
SIGMA_ANALYSIS_LOG_PATH = LOG_DIR / "sigma_analysis.log"

# =============================================================================
# Experiment 8: Modality Gap Visualization Configuration
# =============================================================================

# =============================================================================
# Utility Functions
# =============================================================================
def ensure_project_dirs() -> None:
    """
    Ensure all required project directories exist.
    Creates directories if they don't exist.
    """
    directories = [
        CHECKPOINT_DIR,
        OUTPUT_DIR,
        LOG_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    # Also ensure log directory exists for log files
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def print_config() -> None:
    """Print current configuration for verification."""
    print("=" * 60)
    print("DistributionAlignment Configuration")
    print("=" * 60)
    print(f"Platform: {'Windows' if IS_WINDOWS else 'Linux/Unix'}")
    print(f"Project Root: {PROJECT_ROOT}")
    print("")
    print("Dataset Paths:")
    print(f"  Captions: {CAPTIONS_PATH}")
    print(f"  Images: {IMAGES_DIR}")
    print("")
    print("Model Paths:")
    print(f"  CLIP ViT-L/14: {CLIP_VIT_L_14_PATH}")
    print("")
    print("Output Paths:")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Outputs: {OUTPUT_DIR}")
    print(f"  Logs: {LOG_DIR}")
    print("")
    print("Training Hyperparameters:")
    print(f"  Epochs: {CLIP_BASELINE_EPOCHS}")
    print(f"  Batch Size: {CLIP_BASELINE_BATCH_SIZE}")
    print(f"  Learning Rate: {CLIP_BASELINE_LR}")
    print(f"  Weight Decay: {CLIP_BASELINE_WEIGHT_DECAY}")
    print(f"  Temperature: {CLIP_BASELINE_TEMPERATURE}")
    print(f"  Freeze Image: {CLIP_BASELINE_FREEZE_IMAGE}")
    print(f"  Freeze Text: {CLIP_BASELINE_FREEZE_TEXT}")
    print("=" * 60)

if __name__ == "__main__":
    ensure_project_dirs()
    print_config()
