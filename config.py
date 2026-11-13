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
    PROJECT_ROOT = Path("/mydata/home/yxzhou/kxiong/mcdispAlign/DistributionAlignment")


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

# Training log for CLIP Set-FT (rebuttal exp 1)
TRAIN_CLIP_SETFT_LOG_PATH = LOG_DIR / "train_clip_setft.log"

# Evaluation log for CLIP baseline
EVAL_CLIP_BASELINE_LOG_PATH = LOG_DIR / "evaluate_clip_baseline.log"

# Evaluation log for CLIP zero-shot
EVAL_CLIP_ZERO_SHOT_LOG_PATH = LOG_DIR / "evaluate_clip_zero_shot.log"

# Evaluation log for the all-hit@K / cover-rank multi-caption metrics (scripts/eval_allhit.py)
EVAL_ALLHIT_LOG_PATH = LOG_DIR / "eval_allhit.log"


# =============================================================================
# Checkpoint Paths
# =============================================================================
# Best checkpoint (lowest validation loss)
CLIP_BASELINE_BEST_CKPT = CHECKPOINT_DIR / "clip_baseline_coco_best.pt"

# Last checkpoint (end of training)
CLIP_BASELINE_LAST_CKPT = CHECKPOINT_DIR / "clip_baseline_coco_last.pt"

# CLIP Set-FT (rebuttal exp 1): same hyperparameters as CLIP_BASELINE_*
# (epochs/batch/lr/wd/temperature), only the text supervision differs --
# five captions per image averaged into one set vector.
CLIP_SETFT_BEST_CKPT = CHECKPOINT_DIR / "clip_setft_coco_best.pt"
CLIP_SETFT_LAST_CKPT = CHECKPOINT_DIR / "clip_setft_coco_last.pt"


# =============================================================================
# Evaluation Results Paths
# =============================================================================
# JSON file for evaluation results
CLIP_BASELINE_EVAL_RESULTS_PATH = OUTPUT_DIR / "clip_baseline_eval_results.json"

# JSON file for CLIP zero-shot evaluation results
CLIP_ZERO_SHOT_EVAL_RESULTS_PATH = OUTPUT_DIR / "clip_zero_shot_eval_results.json"

# JSON file for the all-hit@K / cover-rank multi-caption metrics (scripts/eval_allhit.py)
ALLHIT_EVAL_RESULTS_PATH = OUTPUT_DIR / "allhit_eval_results.json"


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
MCDISP_ALIGN_LAST_CKPT = CHECKPOINT_DIR / "mcdisp_align_coco_last.pt"

# Evaluation results path
MCDISP_ALIGN_EVAL_RESULTS_PATH = OUTPUT_DIR / "mcdisp_align_eval_results.json"
MCDISP_COVERAGE_EVAL_RESULTS_PATH = OUTPUT_DIR / "mcdisp_coverage_eval_results.json"

# I2T per-caption pair-count metric (separate file)
MCDISP_ALIGN_I2T_PAIR_COUNTS_PATH = OUTPUT_DIR / "mcdisp_align_i2t_pair_counts.json"

# Training hyperparameters
MCDISP_ALIGN_EPOCHS = 10
MCDISP_ALIGN_BATCH_SIZE = 32
MCDISP_ALIGN_CLIP_LR = 1e-6  # Learning rate for CLIP (if fine-tuning)
MCDISP_ALIGN_MLP_LR = 5e-5   # Learning rate for MLP distribution heads (trained from scratch; balanced for convergence vs overfitting)
MCDISP_ALIGN_WEIGHT_DECAY = 1e-4
MCDISP_ALIGN_FREEZE_CLIP = True  # Whether to freeze CLIP parameters

# Distribution configuration
MCDISP_ALIGN_DROPOUT_RATE = 0.1         # Dropout rate for MLP heads
MCDISP_ALIGN_DISTRIBUTION_MERGING = "moment_matching"  # Method: "moment_matching", "poe", "simple"
# Auxiliary per-caption NCE (loss_name="kl_capnce"): weight and its own
# fixed temperature. lambda_cap_nce=0 reduces exactly to the KL objective.
MCDISP_ALIGN_LAMBDA_CAP_NCE = 0.5
MCDISP_ALIGN_CAP_NCE_TAU = 0.07


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
MCDISP_ALIGN_STAGE_MAIN_FRAC = 0.6        # + L_var + L_cover
MCDISP_ALIGN_STAGE_FULL_FRAC = 0.2        # + L_cov (ramped 0 -> 1)


# =============================================================================
# VQA-as-Retrieval Dataset Paths
# =============================================================================
# Anchor path for the MSCOCO VQA questions split.
VQA_TRAIN_QUESTIONS = PROJECT_ROOT / "TrainDatasets" / "mscoco_captions" / "train" / "questions.txt"

# VQA images share the same directory as MSCOCO captions
VQA_IMAGES_DIR = IMAGES_DIR


# =============================================================================
# Caption-Generation Configuration (build_vqa_expansions.py)
# =============================================================================
# API configuration file path (stores API keys, never commit this file)
API_CONFIG_PATH = PROJECT_ROOT / "api_config.json"

# API call settings (used by the --backend api caption path)
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
PROLIP_LAST_CKPT = CHECKPOINT_DIR / "prolip_coco_last.pt"
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

# ProLIP backbone variants (B3). All variants share the fine-tuning pipeline
# (train_prolip.py, ProLIP loss); they differ in the pretrained checkpoint and
# its preprocessing/tokenization:
#   h14 : official ProLIP-ViT-H-14-FT-DC-1B-1_28B (legacy default, embed 1024)
#   l14 : self-assembled from the local OpenAI CLIP ViT-L/14 (built by
#         tools/build_prolip_vit_l14.py; SAME backbone+init as Ours/B1/B2/B4/B5,
#         but WITHOUT ProLIP's 1.28B-pair probabilistic pretraining -- label
#         honestly as "ProLIP-objective fine-tune from CLIP init", embed 768)
#   l16 : official ProLIP-ViT-L-16-FT-DC-1B-1_28B (SigLIP architecture: 256px
#         input, SigLIP norm, sentencepiece 32100-vocab tokenizer, context 64,
#         embed 1024)
# ckpt_prefix feeds resolve_checkpoint(): h14 keeps the legacy
# prolip_{dataset}_best.pt names; variants use prolip_{l14,l16}_{dataset}_*.pt.
PROLIP_VARIANTS = {
    "h14": dict(
        model_path=PROLIP_MODEL_PATH,
        processor="clip",
        processor_path=PROLIP_PROCESSOR_PATH,
        tokenizer_path=PROLIP_TOKENIZER_PATH,
        context_length=77,
        clean="whitespace",
        embed_dim=1024,
        ckpt_prefix="prolip",
        eval_results=PROLIP_EVAL_RESULTS_PATH,
        zero_shot_results=PROLIP_ZERO_SHOT_EVAL_RESULTS_PATH,
        label="ProLIP ViT-H/14 (official, fine-tuned)",
    ),
    "l14": dict(
        model_path=PROJECT_ROOT / "PreTrainedModels" / "prolip-vit-l-14",
        processor="clip",
        processor_path=PROLIP_PROCESSOR_PATH,   # same 224 CLIP processor
        tokenizer_path=PROLIP_TOKENIZER_PATH,   # same CLIP BPE tokenizer
        context_length=77,
        clean="whitespace",
        embed_dim=768,
        ckpt_prefix="prolip_l14",
        eval_results=OUTPUT_DIR / "prolip_l14_eval_results.json",
        zero_shot_results=OUTPUT_DIR / "prolip_l14_zero_shot_eval_results.json",
        label="ProLIP ViT-L/14 (CLIP init, no 1.28B probabilistic pretraining)",
    ),
    "l16": dict(
        model_path=PROJECT_ROOT / "PreTrainedModels" / "prolip-vit-l-16",
        processor="siglip",                    # constructed programmatically (256, 0.5/0.5)
        processor_path=None,
        tokenizer_path=PROJECT_ROOT / "PreTrainedModels" / "prolip-siglip-tokenizer",
        context_length=64,
        clean="canonicalize",
        embed_dim=1024,
        ckpt_prefix="prolip_l16",
        eval_results=OUTPUT_DIR / "prolip_l16_eval_results.json",
        zero_shot_results=OUTPUT_DIR / "prolip_l16_zero_shot_eval_results.json",
        label="ProLIP ViT-L/16 (official, SigLIP-arch, 1.28B pretrained)",
    ),
}


# =============================================================================
# Baseline B4: ProbVLM Configuration (Upadhyay et al., ICCV 2023)
# =============================================================================
# ProbVLM: two BayesCap MLP adapters (image + text) on top of the FROZEN CLIP
# ViT-L/14 projected embeddings (D=768; the official paper used ViT-B/32,
# D=512). Trained with the TempComb loss (T1*L1 + T2*GenGauss NLL) plus
# cross-modal terms against the frozen features; only the adapters are
# trainable. Official hyperparameters kept as defaults.
# Checkpoint names satisfy utils/eval_common.resolve_checkpoint("probvlm", dataset).
PROBVLM_BEST_CKPT = CHECKPOINT_DIR / "probvlm_coco_best.pt"
PROBVLM_LAST_CKPT = CHECKPOINT_DIR / "probvlm_coco_last.pt"
PROBVLM_EVAL_RESULTS_PATH = OUTPUT_DIR / "probvlm_eval_results.json"
TRAIN_PROBVLM_LOG_PATH = LOG_DIR / "train_probvlm.log"
EVAL_PROBVLM_LOG_PATH = LOG_DIR / "evaluate_probvlm.log"

# Adapter architecture (official BayesCap_for_CLIP defaults)
PROBVLM_EMBED_DIM = 768             # CLIP ViT-L/14 projection dim
PROBVLM_HIDDEN_DIM = 256
PROBVLM_NUM_LAYERS = 3
PROBVLM_DROPOUT = 0.1

# Training (official train_ProbVLM defaults: Adam over adapter params only,
# per-epoch cosine annealing, <=2000 batches per epoch). The official code
# steps its scheduler outside the epoch loop (indentation bug, effectively a
# constant LR); the intent -- cosine annealing per epoch -- is implemented
# via the shared apply_lr_for_epoch with warmup 0 / min_lr_ratio 0.0.
PROBVLM_EPOCHS = 100
PROBVLM_BATCH_SIZE = 32
PROBVLM_LR = 1e-4
PROBVLM_WEIGHT_DECAY = 0.0
PROBVLM_MAX_BATCHES_PER_EPOCH = 2000
# The official schedule never early-stops and selects by val MAE; the project
# pattern selects by val loss. Patience 10 (not the usual 3) because a
# 100-epoch cosine anneal improves slowly mid-run and patience 3 would
# truncate it; --no-early-stop reproduces the official full schedule.
PROBVLM_EARLY_STOP_PATIENCE = 10

# Loss weights (official TempComb / cross-modal defaults)
PROBVLM_T1 = 1.0
PROBVLM_T2 = 5e-2
PROBVLM_CROSS_MODAL_LAMBDA = 1e-4


# =============================================================================
# Baseline B5: PCME++ Configuration (Chun, ICLR 2024)
# =============================================================================
# PCME++ (official code github.com/naver-ai/pcmepp, vendored under pcmepp/):
# probabilistic CLIP whose last backbone layer splits into a mean branch and
# a freshly-initialized uncertainty branch; per-token features (no CLIP
# projection) are pooled by Linear+GPO into mu (l2-normalized) and
# log sigma^2. Trained with the closed-form probabilistic-distance BCE loss
# with pseudo-positive smoothing. Backbone: OpenAI CLIP ViT-L/14 (converted
# to the OpenAI state dict layout by tools/convert_hf_clip_to_openai.py).
# ALL hyperparameters below follow the official configs/pcmepp.yaml +
# README ViT-L/14 recipe (single-GPU batch 128 == the official 8x16).
PCMEPP_BEST_CKPT = CHECKPOINT_DIR / "pcmepp_coco_best.pt"
PCMEPP_LAST_CKPT = CHECKPOINT_DIR / "pcmepp_coco_last.pt"
PCMEPP_EVAL_RESULTS_PATH = OUTPUT_DIR / "pcmepp_eval_results.json"
TRAIN_PCMEPP_LOG_PATH = LOG_DIR / "train_pcmepp.log"
EVAL_PCMEPP_LOG_PATH = LOG_DIR / "evaluate_pcmepp.log"

# OpenAI-format CLIP ViT-L/14 state dict for the vendored loader (~1.7GB,
# generated by tools/convert_hf_clip_to_openai.py from CLIP_VIT_L_14_PATH)
PCMEPP_OPENAI_CLIP_PATH = PROJECT_ROOT / "PreTrainedModels" / "clip-vit-large-patch14-openai.pt"

# Model (official COCO recipe: n_unc_layers 1, sigma_ln_init null, embed 1024)
PCMEPP_EMBED_SIZE = 1024
PCMEPP_IMG_DIM = 1024              # ViT-L/14 transformer width (pre-projection tokens)
PCMEPP_N_UNC_LAYERS = 1
PCMEPP_INIT_SHIFT = 5.0
PCMEPP_INIT_NEGATIVE_SCALE = 5.0
PCMEPP_VIB_BETA = 1e-4
PCMEPP_SMOOTHNESS_ALPHA = 0.1      # pseudo-positive label smoothing
PCMEPP_PROB_DISTANCE = "csd"
PCMEPP_IMG_SIZE_AUGMENT = 0.2      # token-level SizeAugment inside the encoder

# Optimizer (official: AdamP + layerwise LR decay; the LLRD formula hardcodes
# 12 -- for the 24-layer ViT-L vision tower blocks >= 12 get factors > 1,
# faithfully kept as the official behavior for ViT-L/14)
PCMEPP_EPOCHS = 5                 # reduced from the official 25 (training budget)
PCMEPP_BATCH_SIZE = 128            # == official 8 GPUs x 16
PCMEPP_LR = 5e-4
PCMEPP_WEIGHT_DECAY = 1e-4
PCMEPP_LAYERWISE_LR_DECAY = 0.7
PCMEPP_TXT_BACKBONE_LR_DECAY = 0.1
PCMEPP_IMG_BACKBONE_LR_DECAY = 0.01
PCMEPP_GRAD_CLIP = 2.0
PCMEPP_LR_MILESTONE_EPOCH = 15     # lr x0.1 from epoch 15 (official SequentialLR)
PCMEPP_IMG_BACKBONE_FREEZE_EPOCHS = 2
PCMEPP_EMBEDDING_WARMUP_EPOCHS = 2  # intra-epoch loss warmup at epoch 2 (after freeze)

# Augmentation (official)
PCMEPP_MIXUP = 2.0
PCMEPP_CUTMIX = 2.0
PCMEPP_MIXUP_PROB = 1.0
PCMEPP_MIXUP_SWITCH_PROB = 0.5
PCMEPP_MIXUP_MODE = "elem"
PCMEPP_MIX_BATCH_RATIO = 0.25
PCMEPP_MIXUP_SMOOTHING = 1e-4
PCMEPP_LABEL_SMOOTHING = 1e-4
PCMEPP_TXT_SIZE_AUGMENT = 0.2      # caption token perturbation rate
PCMEPP_TXT_SIZE_AUGMENT_MASKING = 0.5
PCMEPP_TXT_SIZE_AUGMENT_ERASING = 0.1
PCMEPP_PERTURB_VOCAB_SIZE = 30522  # official uses len(BertTokenizer vocab)

# Precision: official trains fp32 ("32-true") -- the default again since
# 2026-11-10: bf16 backward produced non-finite gradients in the (officially
# unbounded) sigma branch during occasional mid-training runaways, which the
# official fp32 recipe survives via grad clipping (2026-11-10 root-cause
# investigation, experiments.md section 10). fp32 batch 128 fits the card
# thanks to gradient checkpointing over the two CLIP towers (numerically
# identical; see PCMEPPModel.enable_grad_checkpoint).
PCMEPP_PRECISION = "32-true"


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
# Experiment 5: Ablation Study Configuration
# =============================================================================
ABLATION_RESULTS_DIR = OUTPUT_DIR / "ablation"
ABLATION_LOG_PATH = LOG_DIR / "ablation.log"

# Ablation configurations per the rebuttal plan (MCDisp-Align_Rebuttal
# §3B 逐项消融矩阵, rows A0-A7) on the KL-objective model (loss_name="kl").
# Every row copies the SAME Full recipe -- the paper defaults
# alpha=1 (lambda_ctr), beta=0.5 (lambda_cover_pos), gamma=0.01 (lambda_cov =
# MCDISP_ALIGN_LAMBDA_COV), eta=0.01 (lambda_reg), r=4, v0=0.04, eps=1e-6 --
# and changes ONLY the knob named in the plan; every other weight, the model
# structure, the data split and the training settings stay identical.
# Row mapping (rebuttal-plan numbering; descriptions carry the row ID so
# results tables map back unambiguously):
#   A0 full_model   : Full recipe (all terms on)
#   A1 no_align     : w/o the whole L_align (kl/cover/cov/reg all 0); only
#                     L_match remains. Compare against A0.
#   A2 no_kl        : KL alignment            -> lambda_kl = 0
#   A3 no_cover     : caption-center coverage -> lambda_cover_pos = 0
#   A4 no_direction : subspace alignment      -> lambda_cov = 0 (low-rank
#                     branch KEPT so only the direction supervision is removed)
#   A5 no_reg       : variance regularization -> lambda_reg = 0
#   A6 diag_only    : full covariance vs diagonal-only -> cov_rank 4 vs 0 (the
#                     branch is removed entirely; compare against A4, both
#                     without direction loss, to isolate capacity)
#   A7 no_sg        : cancel sg[.] on the caption targets of the three
#                     alignment terms (detach_targets=False); the variance
#                     gradient schedule is unchanged
ABLATION_LOSS_NAME = "kl"
ABLATION_CONFIGS = {
    "full_model": {
        "loss_name": "kl",
        "lambda_ctr": 1.0, "lambda_kl": 1.0,
        "lambda_cover_pos": 0.5, "lambda_cover_neg": 0.0,
        "lambda_cov": MCDISP_ALIGN_LAMBDA_COV, "lambda_reg": 0.01,
        "cov_rank": MCDISP_ALIGN_COV_RANK, "num_captions": 5,
        "use_uncertainty_sim": True,
        "description": "A0: Full MCDisp-Align (KL + coverage + direction)",
    },
    "no_align": {
        "loss_name": "kl",
        "lambda_ctr": 1.0, "lambda_kl": 0.0,
        "lambda_cover_pos": 0.0, "lambda_cover_neg": 0.0,
        "lambda_cov": 0.0, "lambda_reg": 0.0,
        "cov_rank": MCDISP_ALIGN_COV_RANK, "num_captions": 5,
        "use_uncertainty_sim": True,
        "description": "A1: w/o L_align (only L_match; the whole alignment loss disabled)",
    },
    "no_kl": {
        "loss_name": "kl",
        "lambda_ctr": 1.0, "lambda_kl": 0.0,
        "lambda_cover_pos": 0.5, "lambda_cover_neg": 0.0,
        "lambda_cov": MCDISP_ALIGN_LAMBDA_COV, "lambda_reg": 0.01,
        "cov_rank": MCDISP_ALIGN_COV_RANK, "num_captions": 5,
        "use_uncertainty_sim": True,
        "description": "A2: w/o KL Alignment (no direct image<-caption-set center/range supervision)",
    },
    "no_cover": {
        "loss_name": "kl",
        "lambda_ctr": 1.0, "lambda_kl": 1.0,
        "lambda_cover_pos": 0.0, "lambda_cover_neg": 0.0,
        "lambda_cov": MCDISP_ALIGN_LAMBDA_COV, "lambda_reg": 0.01,
        "cov_rank": MCDISP_ALIGN_COV_RANK, "num_captions": 5,
        "use_uncertainty_sim": True,
        "description": "A3: w/o Caption Coverage (per-caption coverage loss removed)",
    },
    "no_direction": {
        "loss_name": "kl",
        "lambda_ctr": 1.0, "lambda_kl": 1.0,
        "lambda_cover_pos": 0.5, "lambda_cover_neg": 0.0,
        "lambda_cov": 0.0, "lambda_reg": 0.01,
        "cov_rank": MCDISP_ALIGN_COV_RANK, "num_captions": 5,
        "use_uncertainty_sim": True,
        "description": "A4: w/o Direction Alignment (L_cov=0; low-rank branch kept)",
    },
    "no_reg": {
        "loss_name": "kl",
        "lambda_ctr": 1.0, "lambda_kl": 1.0,
        "lambda_cover_pos": 0.5, "lambda_cover_neg": 0.0,
        "lambda_cov": MCDISP_ALIGN_LAMBDA_COV, "lambda_reg": 0.0,
        "cov_rank": MCDISP_ALIGN_COV_RANK, "num_captions": 5,
        "use_uncertainty_sim": True,
        "description": "A5: w/o Variance Regularization (weak variance prior removed)",
    },
    "diag_only": {
        "loss_name": "kl",
        "lambda_ctr": 1.0, "lambda_kl": 1.0,
        "lambda_cover_pos": 0.5, "lambda_cover_neg": 0.0,
        "lambda_cov": 0.0, "lambda_reg": 0.01,
        "cov_rank": 0,
        "num_captions": 5,
        "use_uncertainty_sim": True,
        "description": "A6: Diagonal-only (no low-rank covariance branch; r=0)",
    },
    "no_sg": {
        "loss_name": "kl",
        "lambda_ctr": 1.0, "lambda_kl": 1.0,
        "lambda_cover_pos": 0.5, "lambda_cover_neg": 0.0,
        "lambda_cov": MCDISP_ALIGN_LAMBDA_COV, "lambda_reg": 0.01,
        "cov_rank": MCDISP_ALIGN_COV_RANK, "num_captions": 5,
        "use_uncertainty_sim": True,
        "detach_targets": False,
        "description": "A7: no sg (L_align caption targets not detached; variance schedule unchanged)",
    },
}

# Sensitivity analysis parameter grids (KL-objective terms)
ABLATION_LAMBDA_KL_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
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

# Mu heads as identity-start residual MLPs (x + MLP(x), zero-init output
# layer). Fresh Xavier heads drop the frozen backbone ~13 pts below its own
# zero-shot retrieval; the residual start preserves it. (models/mcdisp_align_model.py)
MCDISP_ALIGN_MU_RESIDUAL = False
