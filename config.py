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
    PROJECT_ROOT = Path("/home/your_name/code/causality/GaussianImageDistribution")


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
# Dataset Settings
# =============================================================================
# Number of captions to use per image (minimum 5, will pad if necessary)
NUM_CAPTIONS = 5

# Number of worker processes for data loading
# Set to 0 for single-process data loading (no multiprocessing/threads)
# This ensures maximum compatibility across all platforms
NUM_WORKERS = 0


# =============================================================================
# CLIP Baseline Training Hyperparameters
# =============================================================================
# Number of training epochs
CLIP_BASELINE_EPOCHS = 1

# Training batch size
CLIP_BASELINE_BATCH_SIZE = 16

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
EVAL_BATCH_SIZE = 32


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
DIST_ALIGN_MLP_LR = 1e-6   # Learning rate for MLP distribution heads
DIST_ALIGN_WEIGHT_DECAY = 1e-4
DIST_ALIGN_TEMPERATURE = 0.07
DIST_ALIGN_FREEZE_CLIP = True  # Whether to freeze CLIP parameters

# Loss function weights
DIST_ALIGN_LAMBDA_CONTRASTIVE = 1.0  # Weight for contrastive loss
DIST_ALIGN_LAMBDA_KL = 10.0          # Weight for KL divergence loss (primary optimization target)
DIST_ALIGN_LAMBDA_VAR = 0.1           # Weight for variance regularization (optional)

# Distribution configuration
DIST_ALIGN_DROPOUT_RATE = 0.1         # Dropout rate for MLP heads
DIST_ALIGN_DISTRIBUTION_MERGING = "moment_matching"  # Method: "moment_matching", "poe", "simple"
DIST_ALIGN_KL_TYPE = "symmetric"      # KL divergence type: "symmetric", "forward", "reverse", "wasserstein"
DIST_ALIGN_TARGET_VARIANCE = 0.5      # Target variance for regularization
DIST_ALIGN_USE_VARIANCE_LOSS = False # Whether to use variance regularization loss

# Distributional Contrastive Learning via OT configuration
DIST_ALIGN_USE_OT_CONTRASTIVE = False  # Use OT-based distributional contrastive loss
DIST_ALIGN_OT_TEMPERATURE = 10.0      # Temperature for W2-based similarity (τ), larger for high-dim W2
DIST_ALIGN_LAMBDA_OT = 1.0            # Weight for distributional contrastive loss
DIST_ALIGN_LAMBDA_VAR_OT = 0.1        # Weight for variance regularization in OT mode
DIST_ALIGN_MIN_SIGMA = 1e-3           # Minimum sigma to prevent numerical collapse

# Uncertainty-Calibrated Distributional Contrastive Learning configuration
DIST_ALIGN_USE_UC_CL = True           # Use Uncertainty-Calibrated Distributional Contrastive Learning
DIST_ALIGN_UC_TEMPERATURE = 0.07      # Temperature for uncertainty-calibrated similarity
DIST_ALIGN_LAMBDA_UC_CL = 1.0         # Weight for uncertainty-calibrated contrastive loss (λ_cl)
DIST_ALIGN_LAMBDA_CONSIST = 1.0       # Weight for distributional consistency loss (λ_consist)
DIST_ALIGN_LAMBDA_UC_VAR = 0.1        # Weight for variance regularization in UC-CL mode (λ_var)
DIST_ALIGN_UC_TARGET_VARIANCE = 0.5   # Target variance for regularization


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
VQA_BATCH_SIZE = 32
VQA_LR = 1e-3
VQA_WEIGHT_DECAY = 1e-4
VQA_HIDDEN_DIM = 512
VQA_DROPOUT = 0.1
VQA_NUM_WORKERS = 0
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
    "kimi-k2.5": OUTPUT_DIR / "llm_vqa_kimi-k2.5_results.json",
}

# Models to evaluate
LLM_MODELS = {
    "Qwen/Qwen3.5-4B": "qwen3.5-4b",
    "Pro/moonshotai/Kimi-K2.5": "kimi-k2.5",
}

# API call settings
LLM_API_DELAY = 0.5          # Delay between API calls (seconds)
LLM_API_MAX_RETRIES = 3      # Maximum retries for failed API calls
LLM_API_RETRY_WAIT = 5       # Base wait time for retries (seconds)
LLM_API_TIMEOUT = 60         # Request timeout (seconds)


# =============================================================================
# Baseline B3: ProLIP Configuration
# =============================================================================
# HuggingFace model identifier for ProLIP pretrained weights
PROLIP_MODEL_NAME = "thanossk/prolip-vit-b16-laion400m"
# If using local cache, specify the path; None means download from HF
PROLIP_LOCAL_PATH = None

# ProLIP checkpoint and output paths
PROLIP_BEST_CKPT = CHECKPOINT_DIR / "prolip_best.pt"
PROLIP_EVAL_RESULTS_PATH = OUTPUT_DIR / "prolip_eval_results.json"
TRAIN_PROLIP_LOG_PATH = LOG_DIR / "train_prolip.log"
EVAL_PROLIP_LOG_PATH = LOG_DIR / "evaluate_prolip.log"

# ProLIP uses its own ViT-B/16, embedding dimension is 512
PROLIP_EMBED_DIM = 512


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
# Baseline B5: ICPE Configuration
# =============================================================================
# ICPE is training-free: computes intra-class covariance on CLIP features

ICPE_EVAL_RESULTS_PATH = OUTPUT_DIR / "icpe_eval_results.json"
EVAL_ICPE_LOG_PATH = LOG_DIR / "evaluate_icpe.log"

# ICPE hyperparameters (no training needed)
ICPE_NUM_NEIGHBORS = 10           # k for k-NN based covariance estimation
ICPE_REGULARIZATION = 1e-6        # Regularization for covariance


# =============================================================================
# Baseline B6: D2P Configuration
# =============================================================================
# D2P: Distribution to Point matching

D2P_BEST_CKPT = CHECKPOINT_DIR / "d2p_best.pt"
D2P_EVAL_RESULTS_PATH = OUTPUT_DIR / "d2p_eval_results.json"
TRAIN_D2P_LOG_PATH = LOG_DIR / "train_d2p.log"
EVAL_D2P_LOG_PATH = LOG_DIR / "evaluate_d2p.log"

# D2P hyperparameters
D2P_EPOCHS = 10
D2P_BATCH_SIZE = 32
D2P_LR = 1e-4
D2P_WEIGHT_DECAY = 1e-4
D2P_TEMPERATURE = 0.07
D2P_NUM_SAMPLES = 10              # Number of distribution samples for matching
D2P_DROPOUT_RATE = 0.1


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
        "lambda_cl": 1.0, "lambda_consist": 1.0, "lambda_var": 0.1,
        "description": "Full model (UC-CL + Consist + Var)"
    },
    "no_consistency": {
        "lambda_cl": 1.0, "lambda_consist": 0.0, "lambda_var": 0.1,
        "description": "w/o Distributional Consistency (λ_c=0)"
    },
    "no_uc": {
        "lambda_cl": 1.0, "lambda_consist": 1.0, "lambda_var": 0.1,
        "use_uc_cl": False,
        "description": "w/o Uncertainty Calibration (standard cosine)"
    },
    "no_var_reg": {
        "lambda_cl": 1.0, "lambda_consist": 1.0, "lambda_var": 0.0,
        "description": "w/o Variance Regularization (λ_v=0)"
    },
    "no_distribution_merging": {
        "lambda_cl": 1.0, "lambda_consist": 1.0, "lambda_var": 0.1,
        "num_captions": 1,
        "description": "w/o Distribution Merging (use single caption)"
    },
    "only_consistency": {
        "lambda_cl": 0.0, "lambda_consist": 1.0, "lambda_var": 0.1,
        "description": "Only Consistency Loss (λ_cl=0)"
    },
}

# Sensitivity analysis parameter grids
ABLATION_LAMBDA_CONSIST_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
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
VQA_ICPE_CKPT = CHECKPOINT_DIR / "vqa_icpe_best.pt"
VQA_D2P_CKPT = CHECKPOINT_DIR / "vqa_d2p_best.pt"


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
