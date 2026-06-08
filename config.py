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

# =============================================================================
# VQA Checkpoint Paths
# =============================================================================
VQA_DIST_ALIGN_CKPT = CHECKPOINT_DIR / "vqa_dist_align_best.pt"
VQA_CLIP_BASELINE_CKPT = CHECKPOINT_DIR / "vqa_clip_baseline_best.pt"
VQA_FREEZE_ALIGN_CKPT = CHECKPOINT_DIR / "vqa_freeze_align_best.pt"
VQA_FATE_CKPT = CHECKPOINT_DIR / "vqa_fate_best.pt"
VQA_CLIP_AST_CKPT = CHECKPOINT_DIR / "vqa_clip_ast_best.pt"
VQA_LOG_PATH = LOG_DIR / "train_vqa.log"


# =============================================================================
# Freeze-Align Model Configuration
# =============================================================================
# Checkpoint paths
FREEZE_ALIGN_BEST_CKPT = CHECKPOINT_DIR / "freeze_align_best.pt"
FREEZE_ALIGN_LAST_CKPT = CHECKPOINT_DIR / "freeze_align_last.pt"

# Log paths
TRAIN_FREEZE_ALIGN_LOG_PATH = LOG_DIR / "train_freeze_align.log"
EVAL_FREEZE_ALIGN_LOG_PATH = LOG_DIR / "evaluate_freeze_align.log"

# Evaluation results path
FREEZE_ALIGN_EVAL_RESULTS_PATH = OUTPUT_DIR / "freeze_align_eval_results.json"

# Model hyperparameters
FREEZE_ALIGN_PROJ_DIM = 256        # Bottleneck dimension for projectors
FREEZE_ALIGN_DROPOUT_RATE = 0.1    # Dropout rate for projectors
FREEZE_ALIGN_STRUCTURE_WEIGHT = 0.1  # Weight for STRUCTURE regularization loss

# Stage 1 training hyperparameters (image-caption alignment)
FREEZE_ALIGN_EPOCHS = 10
FREEZE_ALIGN_BATCH_SIZE = 32
FREEZE_ALIGN_LR = 1e-3             # Learning rate for projectors
FREEZE_ALIGN_WEIGHT_DECAY = 1e-4
FREEZE_ALIGN_TEMPERATURE = 0.07


# =============================================================================
# FATE Model Configuration
# =============================================================================
# Checkpoint paths
FATE_BEST_CKPT = CHECKPOINT_DIR / "fate_best.pt"
FATE_LAST_CKPT = CHECKPOINT_DIR / "fate_last.pt"

# Log paths
TRAIN_FATE_LOG_PATH = LOG_DIR / "train_fate.log"
EVAL_FATE_LOG_PATH = LOG_DIR / "evaluate_fate.log"

# Evaluation results path
FATE_EVAL_RESULTS_PATH = OUTPUT_DIR / "fate_eval_results.json"

# Model hyperparameters
FATE_BOTTLENECK_DIM = 64   # Bottleneck dimension for projector
FATE_ALPHA = 0.001          # Scaling factor for vision perturbation

# Stage 1 training hyperparameters (image-caption alignment)
FATE_EPOCHS = 10
FATE_BATCH_SIZE = 32
FATE_LR = 2e-3              # Learning rate for projector (paper: SGD lr=0.002)
FATE_WEIGHT_DECAY = 1e-4
FATE_TEMPERATURE = 0.07


# =============================================================================
# CLIP-AST Model Configuration
# =============================================================================
# Checkpoint paths
CLIP_AST_BEST_CKPT = CHECKPOINT_DIR / "clip_ast_best.pt"
CLIP_AST_LAST_CKPT = CHECKPOINT_DIR / "clip_ast_last.pt"

# Log paths
TRAIN_CLIP_AST_LOG_PATH = LOG_DIR / "train_clip_ast.log"
EVAL_CLIP_AST_LOG_PATH = LOG_DIR / "evaluate_clip_ast.log"

# Evaluation results path
CLIP_AST_EVAL_RESULTS_PATH = OUTPUT_DIR / "clip_ast_eval_results.json"

# Model hyperparameters
CLIP_AST_SELECT_RATIO = 0.05  # Fraction of CLIP params to select for fine-tuning
CLIP_AST_CLIP_LR = 1e-6       # Learning rate for selected CLIP params
CLIP_AST_WARMUP_EPOCHS = 1    # Warmup epochs before parameter selection

# Stage 1 training hyperparameters (image-caption alignment)
CLIP_AST_EPOCHS = 10
CLIP_AST_BATCH_SIZE = 32
CLIP_AST_WEIGHT_DECAY = 1e-4
CLIP_AST_TEMPERATURE = 0.07

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
