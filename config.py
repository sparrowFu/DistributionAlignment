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
# 15 epochs: with the 5-stage schedule the Full stage (last 20%, incl. L_cov)
# needs ~3 epochs -- at 10 epochs + early stopping the run died at E7 and
# L_cov never trained (traincoco.log 2026-08-25). Fixed budget, no early stop.
MCDISP_ALIGN_EPOCHS = 15
# 64: doubles the InfoNCE negatives; frozen backbone builds no CLIP graph, so
# the memory cost is modest (fall back to 48/32 if the GPU is shared).
MCDISP_ALIGN_BATCH_SIZE = 64
MCDISP_ALIGN_CLIP_LR = 1e-6  # Learning rate for CLIP (if fine-tuning)
MCDISP_ALIGN_MLP_LR = 5e-5   # Learning rate for MLP distribution heads (trained from scratch; balanced for convergence vs overfitting)
MCDISP_ALIGN_WEIGHT_DECAY = 1e-4
MCDISP_ALIGN_FREEZE_CLIP = True  # Whether to freeze CLIP parameters

# Distribution configuration
MCDISP_ALIGN_DROPOUT_RATE = 0.1         # Dropout rate for MLP heads

# =============================================================================
# MCDisp_Align: Multi-Caption Semantic Dispersion Guided Distribution Alignment
# =============================================================================
# Method overview:
# image and text are modeled as Gaussians. The image uses a general covariance
# Sigma_v = diag(sigma_v^2) + U_v U_v^T (U_v in R^{D x r}); text is diagonal-only
# (v1). The image variance is supervised toward the multi-caption semantic spread,
# and the image low-rank directions toward the caption deviation directions.
#
# Total loss = lambda_ctr*L_set + lambda_mu*L_mu + lambda_var*L_var
#            + lambda_cover*L_cover + lambda_cov*L_cov + lambda_reg*L_reg
# where L_set is a bidirectional InfoNCE on the uncertainty-discounted cosine
# similarity  sim = (mu_v . mu_t) / (tau * sqrt(1+mean sigma_v^2) * sqrt(1+mean sigma_t^2)).
MCDISP_ALIGN_COV_RANK = 4                 # low-rank covariance rank r for the IMAGE side (0 = diagonal only)
MCDISP_ALIGN_TAU = 0.07                   # FIXED temperature in the L_set similarity (not learnable)
MCDISP_ALIGN_LAMBDA_CTR = 1.0             # weight for the set contrastive loss L_set
MCDISP_ALIGN_LAMBDA_MU = 0.5              # weight for the mean-center alignment loss L_mu
MCDISP_ALIGN_LAMBDA_VAR = 1.0             # weight for the variance semantic consistency loss L_var (core)
MCDISP_ALIGN_LAMBDA_COVER_POS = 0.5       # weight for L_cover positive coverage
MCDISP_ALIGN_LAMBDA_COVER_NEG = 0.0       # weight for L_cover negative repulsion (0 = off)
MCDISP_ALIGN_LAMBDA_COV = 0.01            # weight for L_cov -- STABILITY-RUN value (Full-cov crash risk); official target 0.2 once Full stage verified stable
MCDISP_ALIGN_LAMBDA_REG = 0.01            # weight for the variance regularization loss L_reg
MCDISP_ALIGN_M_POS = 1.0                  # L_cover positive coverage margin (per-D normalized Mahalanobis)
MCDISP_ALIGN_M_NEG = 2.0                  # L_cover negative repulsion margin
MCDISP_ALIGN_TARGET_VAR = 0.04            # L_reg variance prior sigma_0^2 (= measured MSCOCO caption spread)
MCDISP_ALIGN_USE_UNCERTAINTY_SIM = True   # L_set/retrieval use the uncertainty-discounted score (False = plain cosine)
MCDISP_ALIGN_VAR_FLOOR = 1e-4             # numerical floor on sigma^2 (softplus positivity + div-by-zero guard; NOT a semantic floor -- the range is learned via L_var / L_reg)
MCDISP_ALIGN_VAR_FLOOR_NEAR_MULT = 10     # variance floor-collapse monitor: a dim is "near floor" if sigma^2 < MCDISP_ALIGN_VAR_FLOOR_NEAR_MULT * MCDISP_ALIGN_VAR_FLOOR
MCDISP_ALIGN_VAR_FLOOR_RATIO_WARN = 0.5   # warn when >this fraction of dims are near-floor AND mean sigma^2 < 2*MCDISP_ALIGN_VAR_FLOOR
MCDISP_ALIGN_COV_EPS = 1e-6               # numerical epsilon for Mahalanobis / log
MCDISP_ALIGN_GRAD_CLIP_NORM = 1.0         # global grad-norm clip (clip_grad_norm_) -- guards against L_cov / cover spikes destabilizing the retrieval means
# Deprecated likelihood-rewrite knobs (no longer used by the training/loss path).
MCDISP_ALIGN_USE_LOGDET = True            # deprecated: loglik-eval only
MCDISP_ALIGN_PER_DIM_NORMALIZE = True     # deprecated: loglik-eval only

# 3-stage training schedule (fraction of total epochs each stage spans)
MCDISP_ALIGN_STAGE_WARMUP_FRAC = 0.2      # L_set + L_mu (+ L_reg always on)
MCDISP_ALIGN_STAGE_FULL_FRAC = 0.2        # + L_cov (ramped 0 -> 1)

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
PROLIP_BATCH_SIZE = 16          # ViT-H/14 is large; keep modest to fit GPU
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
