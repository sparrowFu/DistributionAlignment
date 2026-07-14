"""
GaussianImageDistribution - Configuration Module

This module centralizes all paths and hyperparameters for the project.
It supports cross-platform usage (Windows/Ubuntu) by using pathlib.Path.

To migrate between systems:
1. Modify PROJECT_ROOT to point to the project root on the new system
2. All other paths will adjust automatically

Windows default:
    PROJECT_ROOT = D:/code/causality/GaussianImageDistribution

Ubuntu default:
    PROJECT_ROOT = /home/your_name/code/causality/GaussianImageDistribution
"""

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


# =============================================================================
# Checkpoint Paths
# =============================================================================
# Best checkpoint (lowest validation loss)
CLIP_BASELINE_BEST_CKPT = CHECKPOINT_DIR / "clip_baseline_best.pt"

# Last checkpoint (end of training)
CLIP_BASELINE_LAST_CKPT = CHECKPOINT_DIR / "clip_baseline_last.pt"


# =============================================================================
# Evaluation Results Paths
# =============================================================================
# JSON file for evaluation results
CLIP_BASELINE_EVAL_RESULTS_PATH = OUTPUT_DIR / "clip_baseline_eval_results.json"


# =============================================================================
# Random Seed
# =============================================================================
SEED = 42


# =============================================================================
# Learning-Rate Scheduling (shared across all training scripts)
# =============================================================================
# Cosine annealing with linear warmup, applied per-epoch. See utils/lr_scheduler.py.
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
# Distribution Alignment Model Configuration
# =============================================================================
# Training log for distribution alignment model
TRAIN_DIST_ALIGN_LOG_PATH = LOG_DIR / "train_dist_align.log"

# Evaluation log for distribution alignment model
EVAL_DIST_ALIGN_LOG_PATH = LOG_DIR / "evaluate_dist_align.log"

# Checkpoint paths
DIST_ALIGN_BEST_CKPT = CHECKPOINT_DIR / "dist_align_best.pt"
DIST_ALIGN_LAST_CKPT = CHECKPOINT_DIR / "dist_align_last.pt"

# Evaluation results path
DIST_ALIGN_EVAL_RESULTS_PATH = OUTPUT_DIR / "dist_align_eval_results.json"

# Training hyperparameters
DIST_ALIGN_EPOCHS = 10
DIST_ALIGN_BATCH_SIZE = 32
DIST_ALIGN_CLIP_LR = 1e-6  # Learning rate for CLIP (if fine-tuning)
DIST_ALIGN_MLP_LR = 5e-5   # Learning rate for MLP distribution heads (trained from scratch; balanced for convergence vs overfitting)
DIST_ALIGN_WEIGHT_DECAY = 1e-4
DIST_ALIGN_TEMPERATURE = 0.07
DIST_ALIGN_FREEZE_CLIP = True  # Whether to freeze CLIP parameters

# Loss function weights
DIST_ALIGN_LAMBDA_CONTRASTIVE = 1.0  # Weight for contrastive loss
DIST_ALIGN_LAMBDA_KL = 10.0          # Weight for KL divergence loss (legacy; unused by MSDA; ProLIP sets lambda_kl=0.0)
DIST_ALIGN_LAMBDA_VAR = 0.1           # Weight for variance regularization (used by ProLIP baseline)

# Distribution configuration
DIST_ALIGN_DROPOUT_RATE = 0.1         # Dropout rate for MLP heads
DIST_ALIGN_DISTRIBUTION_MERGING = "moment_matching"  # Method: "moment_matching", "poe", "simple"
DIST_ALIGN_KL_TYPE = "symmetric"      # KL divergence type: "symmetric", "forward", "reverse", "wasserstein"
DIST_ALIGN_TARGET_VARIANCE = 0.5      # Target variance for regularization (used by ProLIP baseline)


# =============================================================================
# MSDA: Multi-caption Semantic Distribution Alignment (replaces UC-CL/OT/KL)
# =============================================================================
# Image and text are both modeled as general Gaussians N(mu, Sigma) with a
# learned covariance Sigma = diag(sigma^2) + U U^T (U in R^{D x r}). r controls
# off-diagonal capacity; r=0 falls back to diagonal. The same modeling is applied
# symmetrically to image and text.
MSDA_COV_RANK = 4                 # low-rank covariance rank r (0 = diagonal only)
MSDA_TAU = 0.07                   # temperature for L_set-NCE similarity
MSDA_LAMBDA_CTR = 1.0             # weight for set-level contrastive loss
MSDA_LAMBDA_MU = 0.5              # weight for mean-center alignment loss
MSDA_LAMBDA_VAR = 1.0             # weight for variance semantic consistency (core)
MSDA_LAMBDA_COVER = 0.5           # weight for multi-caption coverage loss
MSDA_LAMBDA_COV = 0.01            # weight for covariance direction alignment (was 0.1: cov_loss magnitude ~2r=8, so 0.1*3~0.3 dominated NCE and crashed Recall@1 the moment L_cov activated)
MSDA_LAMBDA_REG = 0.01            # weight for variance regularization
MSDA_VAR_LOSS_MODE = "rescaled"   # L_var target mode: "rescaled" (mean-match; fix) | "raw" (original)
MSDA_M_POS = 1.0                  # coverage positive radius (per-dim normalized)
MSDA_TARGET_VAR = 0.5             # target variance sigma_0^2 for L_reg
MSDA_VAR_FLOOR = 1e-4             # numerical floor on sigma^2 (softplus positivity + div-by-zero guard; NOT a semantic floor -- the range is learned via L_var / L_reg)
MSDA_COV_EPS = 1e-6               # numerical epsilon for Mahalanobis / log
MSDA_USE_NEG_COVER = False        # optional negative coverage repulsion
MSDA_M_NEG = 2.0                  # negative coverage margin
MSDA_GRAD_CLIP_NORM = 1.0         # global grad-norm clip (clip_grad_norm_) -- guards against L_cov / cover spikes destabilizing the retrieval means

# 3-stage training schedule (fraction of total epochs each stage spans)
MSDA_STAGE_WARMUP_FRAC = 0.2      # L_set-NCE + L_mu
MSDA_STAGE_MAIN_FRAC = 0.6        # + L_var + L_cover
MSDA_STAGE_FULL_FRAC = 0.2        # + L_cov


# =============================================================================
# VQA Dataset Paths
# =============================================================================
VQA_TRAIN_QUESTIONS = PROJECT_ROOT / "TrainDatasets" / "mscoco_captions" / "train" / "questions.txt"
VQA_TRAIN_IMG_FILENAMES = PROJECT_ROOT / "TrainDatasets" / "mscoco_captions" / "train" / "img_filenames.txt"
VQA_TRAIN_TYPES = PROJECT_ROOT / "TrainDatasets" / "mscoco_captions" / "train" / "types.txt"
VQA_TRAIN_ANSWERS = PROJECT_ROOT / "TrainDatasets" / "mscoco_captions" / "train" / "answers.txt"

VQA_TEST_QUESTIONS = PROJECT_ROOT / "TrainDatasets" / "mscoco_captions" / "test" / "questions_filtered.txt"
VQA_TEST_IMG_FILENAMES = PROJECT_ROOT / "TrainDatasets" / "mscoco_captions" / "test" / "img_filenames_filtered.txt"
VQA_TEST_TYPES = PROJECT_ROOT / "TrainDatasets" / "mscoco_captions" / "test" / "types_filtered.txt"
VQA_TEST_ANSWERS = PROJECT_ROOT / "TrainDatasets" / "mscoco_captions" / "test" / "answers_filtered.txt"

# VQA images share the same directory as MSCOCO captions
VQA_IMAGES_DIR = IMAGES_DIR

# =============================================================================
# VQA Training Hyperparameters
# =============================================================================
VQA_EPOCHS = 10
VQA_BATCH_SIZE = 64
VQA_LR = 5e-5
VQA_WEIGHT_DECAY = 1e-4
VQA_HIDDEN_DIM = 512
VQA_DROPOUT = 0.1
VQA_NUM_WORKERS = 0 if IS_WINDOWS else 8
VQA_VAL_SPLIT = 0.1
VQA_EARLY_STOP_PATIENCE = 3

# Distribution-Aware VQA settings (dist_align only)
# Number of Monte Carlo samples during evaluation (0 = disabled, use deterministic mu)
# When enabled, samples z = mu + eps*sigma multiple times and averages predictions
VQA_DIST_NUM_MC_SAMPLES = 0

# =============================================================================
# VQA Checkpoint Paths
# =============================================================================
VQA_DIST_ALIGN_CKPT = CHECKPOINT_DIR / "vqa_dist_align_best.pt"
VQA_CLIP_BASELINE_CKPT = CHECKPOINT_DIR / "vqa_clip_baseline_best.pt"
VQA_LOG_PATH = LOG_DIR / "train_vqa.log"


# =============================================================================
# LLM VQA Evaluation Configuration
# =============================================================================
# API configuration file path (stores API keys, never commit this file)
API_CONFIG_PATH = PROJECT_ROOT / "api_config.json"

# LLM VQA evaluation log
EVAL_LLM_VQA_LOG_PATH = LOG_DIR / "eval_llm_vqa.log"
EVALUATE_LLM_VQA_LOG_PATH = LOG_DIR / "evaluate_llm_vqa.log"

# LLM VQA results output (per model)
LLM_VQA_RESULT_PATHS = {
    "qwen3.5-4b": OUTPUT_DIR / "llm_vqa_qwen3.5-4b_results.json",
}

# Models to evaluate
LLM_MODELS = {
    "Qwen/Qwen3.5-4B": "qwen3.5-4b",
}

# API call settings
LLM_API_DELAY = 0.5          # Delay between API calls (seconds)
LLM_API_MAX_RETRIES = 5      # Maximum retries for failed API calls
LLM_API_RETRY_WAIT = 5       # Base wait time for retries (seconds)
LLM_API_TIMEOUT = 60         # Request timeout (seconds)

# -----------------------------------------------------------------------------
# Caption-generation backend selection (build_vqa_expansions.py)
#   "local" -> transformers, load an open-source LLM from a local directory
#   "api"   -> original OpenAI-compatible HTTP path (SiliconFlow, etc.)
# All LLM_LOCAL_* settings are defaults; each can be overridden on the CLI:
#   --backend --model-path --batch-size --dtype --device --no-batch
# -----------------------------------------------------------------------------
LLM_BACKEND = "local"                # "local" | "api"

# Which local model family to load.
#   "gemma" -> AutoProcessor + AutoModelForMultimodalLM (text-only generation)
#   "llama" -> AutoTokenizer  + AutoModelForCausalLM
LLM_LOCAL_MODEL_KIND = "gemma"       # "gemma" | "llama"

# Default local paths for each family (override with --model-path or LLM_LOCAL_MODEL_PATH).
LLM_LOCAL_GEMMA_PATH = "/home/xpfu/WorkSpace/OpenSourceLLM/gemma"
LLM_LOCAL_LLAMA_PATH = "/home/xpfu/WorkSpace/OpenSourceLLM/llama"

# Explicit override for any family. When non-empty, used regardless of model kind.
LLM_LOCAL_MODEL_PATH = ""
LLM_LOCAL_MODEL_NAME = ""            # display name for logs only (may be "")

LLM_LOCAL_DTYPE = "bf16"             # "bf16" | "fp16" | "fp32"
LLM_LOCAL_DEVICE = "cuda"            # "cuda" | "cuda:0" | "cpu" | "auto"
LLM_LOCAL_BATCH_SIZE = 32            # captions per forward pass (batched mode)
LLM_LOCAL_MAX_NEW_TOKENS = 96        # matches API max_tokens
LLM_LOCAL_DO_SAMPLE = False          # greedy decode == API temperature 0 (reproducible)
LLM_LOCAL_TRUST_REMOTE_CODE = False  # set True for models needing custom code


# =============================================================================
# Baseline B3: ProLIP Configuration (real ProLIP ViT-H/14 via the `prolip` lib)
# =============================================================================
# Three local artifacts (no network needed). Loaded by models/prolip_model.py.
#   model     : ProLIPHF weights (config.json + model.safetensors), embed_dim 1024
#   processor : CLIP image processor (openai/clip-vit-base-patch16) -- correct
#               CLIP normalization for the ViT-H/14 backbone
#   tokenizer : HFTokenizer (apple/DFN5B-CLIP-ViT-H-14), CLIP BPE, context 77
PROLIP_MODEL_PATH = PROJECT_ROOT / "PreTrainedModels" / "prolip"
PROLIP_PROCESSOR_PATH = PROJECT_ROOT / "PreTrainedModels" / "prolipProcessor"
PROLIP_TOKENIZER_PATH = PROJECT_ROOT / "PreTrainedModels" / "prolipTokenizer"
PROLIP_CONTEXT_LENGTH = 77

# ProLIP checkpoint and output paths
PROLIP_BEST_CKPT = CHECKPOINT_DIR / "prolip_best.pt"
PROLIP_LAST_CKPT = CHECKPOINT_DIR / "prolip_last.pt"
PROLIP_EVAL_RESULTS_PATH = OUTPUT_DIR / "prolip_eval_results.json"
PROLIP_ZERO_SHOT_EVAL_RESULTS_PATH = OUTPUT_DIR / "prolip_zero_shot_eval_results.json"
TRAIN_PROLIP_LOG_PATH = LOG_DIR / "train_prolip.log"
EVAL_PROLIP_LOG_PATH = LOG_DIR / "evaluate_prolip.log"
EVAL_PROLIP_ZERO_SHOT_LOG_PATH = LOG_DIR / "evaluate_prolip_zero_shot.log"

# ProLIP ViT-H/14 embedding dimension (mean / log-variance head output dim)
PROLIP_EMBED_DIM = 1024

# Fine-tuning hyperparameters (full fine-tuning of the whole ProLIP model with
# the ProLIP inclusion loss; see prolip.loss.ProLIPLoss).
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
# Baseline B4: GroVE Configuration
# =============================================================================
# GroVE adds GP posterior on top of frozen CLIP features
# Reference: kaaikai/grove

GROVE_BEST_CKPT = CHECKPOINT_DIR / "grove_best.pt"
GROVE_EVAL_RESULTS_PATH = OUTPUT_DIR / "grove_eval_results.json"
TRAIN_GROVE_LOG_PATH = LOG_DIR / "train_grove.log"
EVAL_GROVE_LOG_PATH = LOG_DIR / "evaluate_grove.log"

# GroVE hyperparameters
GROVE_NUM_INDUCING = 128          # Number of inducing points for GP
GROVE_LR = 1e-3
GROVE_EPOCHS = 10
GROVE_BATCH_SIZE = 32
GROVE_WEIGHT_DECAY = 1e-4
GROVE_TEMPERATURE = 0.07


# =============================================================================
# Flickr30K Dataset Configuration
# =============================================================================
FLICKR30K_ROOT = PROJECT_ROOT / "TrainDatasets" / "flickr30k"
FLICKR30K_IMAGES_DIR = FLICKR30K_ROOT / "images"
FLICKR30K_CAPTIONS_PATH = FLICKR30K_ROOT / "captions.txt"
FLICKR30K_NUM_CAPTIONS = 5


# =============================================================================
# Experiment 3: Uncertainty Calibration Configuration
# =============================================================================
CALIBRATION_RESULTS_DIR = OUTPUT_DIR / "calibration"
CALIBRATION_NUM_BINS = 15         # Number of bins for ECE computation
CALIBRATION_LOG_PATH = LOG_DIR / "calibration.log"

# Experiment 3 output paths
CALIBRATION_DIST_ALIGN_PATH = CALIBRATION_RESULTS_DIR / "dist_align_calibration.json"
CALIBRATION_PROLIP_PATH = CALIBRATION_RESULTS_DIR / "prolip_calibration.json"
CALIBRATION_GROVE_PATH = CALIBRATION_RESULTS_DIR / "grove_calibration.json"


# =============================================================================
# Experiment 4: OOD Detection Configuration
# =============================================================================
OOD_RESULTS_DIR = OUTPUT_DIR / "ood_detection"
OOD_LOG_PATH = LOG_DIR / "ood_detection.log"

# OOD datasets: will be downloaded via torchvision if not present
OOD_DATASETS = ["svhn", "cifar10", "tiny_imagenet"]
OOD_DATA_DIR = PROJECT_ROOT / "TrainDatasets" / "ood"


# =============================================================================
# Experiment 5: Ablation Study Configuration
# =============================================================================
ABLATION_RESULTS_DIR = OUTPUT_DIR / "ablation"
ABLATION_LOG_PATH = LOG_DIR / "ablation.log"

# Ablation configurations (each is a dict of overrides)
ABLATION_CONFIGS = {
    "full_model": {
        "lambda_ctr": 1.0, "lambda_mu": 0.5, "lambda_var": 1.0,
        "lambda_cover": 0.5, "lambda_cov": 0.01, "lambda_reg": 0.01,
        "cov_rank": MSDA_COV_RANK, "num_captions": 5,
        "use_uncertainty_sim": True,
        "description": "Full MSDA (set-NCE + mu + var + cover + cov + reg)",
    },
    "no_var": {
        "lambda_ctr": 1.0, "lambda_mu": 0.5, "lambda_var": 0.0,
        "lambda_cover": 0.5, "lambda_cov": 0.01, "lambda_reg": 0.01,
        "cov_rank": MSDA_COV_RANK, "num_captions": 5,
        "use_uncertainty_sim": True,
        "description": "w/o L_var (variance semantic consistency)",
    },
    "no_cover": {
        "lambda_ctr": 1.0, "lambda_mu": 0.5, "lambda_var": 1.0,
        "lambda_cover": 0.0, "lambda_cov": 0.01, "lambda_reg": 0.01,
        "cov_rank": MSDA_COV_RANK, "num_captions": 5,
        "use_uncertainty_sim": True,
        "description": "w/o L_cover (multi-caption coverage)",
    },
    "no_cov": {
        "lambda_ctr": 1.0, "lambda_mu": 0.5, "lambda_var": 1.0,
        "lambda_cover": 0.5, "lambda_cov": 0.0, "lambda_reg": 0.01,
        "cov_rank": MSDA_COV_RANK, "num_captions": 5,
        "use_uncertainty_sim": True,
        "description": "w/o L_cov (covariance direction)",
    },
    "no_mu": {
        "lambda_ctr": 1.0, "lambda_mu": 0.0, "lambda_var": 1.0,
        "lambda_cover": 0.5, "lambda_cov": 0.01, "lambda_reg": 0.01,
        "cov_rank": MSDA_COV_RANK, "num_captions": 5,
        "use_uncertainty_sim": True,
        "description": "w/o L_mu (mean-center alignment)",
    },
    "diagonal_only": {
        "lambda_ctr": 1.0, "lambda_mu": 0.5, "lambda_var": 1.0,
        "lambda_cover": 0.5, "lambda_cov": 0.0, "lambda_reg": 0.01,
        "cov_rank": 0, "num_captions": 5,
        "use_uncertainty_sim": True,
        "description": "Diagonal only (cov_rank=0, no covariance)",
    },
    "no_uncertainty_sim": {
        "lambda_ctr": 1.0, "lambda_mu": 0.5, "lambda_var": 1.0,
        "lambda_cover": 0.5, "lambda_cov": 0.01, "lambda_reg": 0.01,
        "cov_rank": MSDA_COV_RANK, "num_captions": 5,
        "use_uncertainty_sim": False,
        "description": "w/o uncertainty-discounted similarity (standard cosine)",
    },
    "k1": {
        "lambda_ctr": 1.0, "lambda_mu": 0.5, "lambda_var": 1.0,
        "lambda_cover": 0.5, "lambda_cov": 0.0, "lambda_reg": 0.01,
        "cov_rank": MSDA_COV_RANK, "num_captions": 1,
        "use_uncertainty_sim": True,
        "description": "K=1 caption (single-caption fairness)",
    },
    "k3": {
        "lambda_ctr": 1.0, "lambda_mu": 0.5, "lambda_var": 1.0,
        "lambda_cover": 0.5, "lambda_cov": 0.01, "lambda_reg": 0.01,
        "cov_rank": MSDA_COV_RANK, "num_captions": 3,
        "use_uncertainty_sim": True,
        "description": "K=3 captions",
    },
    "k5": {
        "lambda_ctr": 1.0, "lambda_mu": 0.5, "lambda_var": 1.0,
        "lambda_cover": 0.5, "lambda_cov": 0.01, "lambda_reg": 0.01,
        "cov_rank": MSDA_COV_RANK, "num_captions": 5,
        "use_uncertainty_sim": True,
        "description": "K=5 captions (full)",
    },
}

# Sensitivity analysis parameter grids
ABLATION_LAMBDA_VAR_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
ABLATION_LAMBDA_COVER_VALUES = [0.1, 0.5, 1.0, 2.0]
ABLATION_TAU_VALUES = [0.05, 0.07, 0.1, 0.2]


# =============================================================================
# Experiment 7: σ Semantic Analysis Configuration
# =============================================================================
SIGMA_ANALYSIS_RESULTS_DIR = OUTPUT_DIR / "sigma_analysis"
SIGMA_ANALYSIS_LOG_PATH = LOG_DIR / "sigma_analysis.log"


# =============================================================================
# Experiment 8: Modality Gap Visualization Configuration
# =============================================================================
VIS_GAP_RESULTS_DIR = OUTPUT_DIR / "modality_gap"
VIS_GAP_LOG_PATH = LOG_DIR / "visualize_gap.log"


# =============================================================================
# VQA Checkpoint Paths for Baselines
# =============================================================================
VQA_PROLIP_CKPT = CHECKPOINT_DIR / "vqa_prolip_best.pt"
VQA_GROVE_CKPT = CHECKPOINT_DIR / "vqa_grove_best.pt"


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
    print("GaussianImageDistribution Configuration")
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
