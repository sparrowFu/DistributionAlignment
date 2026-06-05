# GaussianImageDistribution

A PyTorch project for image-text representation learning using CLIP fine-tuning and Gaussian distribution modeling.

## Overview

This project implements a modular framework for image-text representation learning with two approaches:

1. **Baseline CLIP Fine-tuning**: Standard contrastive learning baseline using CLIP ViT-Large-Patch14
2. **Distribution Alignment**: Novel method modeling embeddings as Gaussian distributions, addressing modality gap and one-to-many relationships

The distribution alignment method addresses:
- **Modality Gap**: Distribution differences between image and text embeddings
- **One-to-Many Relationships**: Multiple valid descriptions for a single image

## Project Structure

```
GaussianImageDistribution/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Centralized configuration (paths, hyperparameters)
├── main.py                      # Unified entry point for all tasks
├── data/                        # Data loading modules
│   ├── __init__.py
│   └── caption_dataset.py      # Image-caption dataset (MS-COCO format)
├── models/                      # Model definitions
│   ├── __init__.py
│   ├── clip_baseline.py         # CLIP fine-tuning baseline model
│   └── dist_align_model.py      # Distribution alignment model
├── losses/                      # Loss functions
│   ├── __init__.py
│   ├── clip_losses.py           # CLIP contrastive loss
│   └── dist_align_losses.py     # Distribution alignment losses (KL + contrastive + variance reg)
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── seed.py                  # Random seed setting (PyTorch, CUDA, NumPy)
│   ├── io_utils.py              # File I/O utilities (JSON, Parquet, checkpoints)
│   ├── image_utils.py           # Image processing utilities
│   ├── metrics.py               # Evaluation metrics (Recall@K)
│   └── logger.py                # Unified logging system
├── scripts/                     # Training and evaluation scripts
│   ├── train_clip_baseline.py   # Baseline training script
│   ├── evaluate_clip_baseline.py # Baseline evaluation script
│   ├── train_dist_align.py      # Distribution alignment training
│   └── evaluate_dist_align.py   # Distribution alignment evaluation
├── examples/                    # Example scripts and usage demos
│   ├── README.md                # Examples documentation
│   ├── quick_test.py            # Quick sanity check
│   └── test_dist_align.py       # Full pipeline example
├── checkpoints/                  # Model checkpoints (created automatically)
├── outputs/                      # Evaluation results (created automatically)
└── logs/                         # Log files (created automatically)
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `torch>=2.0.0`, `torchvision>=0.15.0`
- `transformers>=4.30.0`
- `pandas>=2.0.0`, `pyarrow>=12.0.0`, `Pillow>=9.5.0`
- `numpy>=1.24.0`, `scikit-learn>=1.2.0`, `tqdm>=4.65.0`

### 2. Configure Paths

Edit `config.py` to set `PROJECT_ROOT` to your project location:

**Windows:**
```python
PROJECT_ROOT = Path("D:/code/causality/GaussianImageDistribution")
```

**Ubuntu:**
```python
PROJECT_ROOT = Path("/home/your_name/code/causality/GaussianImageDistribution")
```

### 3. Verify Configuration

```bash
python config.py
```

This will print all paths and create necessary directories.

## Data Setup

### Required Data Structure

```
GaussianImageDistribution/
└── TrainDatasets/
    └── mscoco_captions/
        ├── captions/
        │   └── train-00000-of-00001.parquet
        └── images/
            ├── 000000000009.jpg
            ├── 000000000036.jpg
            └── ...
```

### Dataset Format

The parquet file should contain columns:
- `url`: Original image URL
- `caption`: List of text descriptions (List[str])
- `image_file_name`: Image filename (e.g., "000000000009.jpg")

Each image has at least 5 captions. If fewer exist, they will be duplicated.

## Model Setup

Download the CLIP ViT-Large-Patch14 model and place it in:

```
GaussianImageDistribution/
└── PreTrainedModels/
    └── clip-vit-large-patch14/
        ├── config.json
        ├── model.safetensors (or pytorch_model.bin)
        ├── preprocessor_config.json
        ├── tokenizer_config.json
        └── ...
```

**Important**: All models are loaded locally. No internet connection is required during training. The `local_files_only=True` parameter is enforced.

## Methods

### Method 1: CLIP Baseline Fine-tuning

Standard CLIP fine-tuning with bidirectional contrastive loss.

**Architecture:**
- CLIP ViT-Large-Patch14 encoder (image_dim=512, text_dim=512)
- Optional encoder freezing (`freeze_image`, `freeze_text`)
- Standard contrastive loss with temperature scaling

**Training:**
```bash
# Via main.py (default: 10 epochs, 10% val split, early stopping)
python main.py --task train_clip_baseline

# Direct script
python scripts/train_clip_baseline.py
```

**Evaluation:**
```bash
python main.py --task eval_clip_baseline
```

**Training Pipeline** (consistent with Distribution Alignment):
```
Full Dataset (118K samples)
    → 90% Train / 10% Validation (random_split, seed=42)
    → Each epoch: train → validate → save best checkpoint if improved
    → Early stopping if val loss doesn't improve for 3 epochs
```

**Data Processing Flow:**
```
Dataset → PIL Images + List[str] (5 captions per image)
    → Random select 1 caption per image
    → model.process_images(pil_images) → pixel_values tensor
    → model.process_text(captions) → input_ids, attention_mask
    → Forward pass → image_features, text_features
    → Contrastive loss + backprop
```

### Method 2: Distribution Alignment

Novel approach modeling embeddings as Gaussian distributions.

**Architecture:**
```
Input
├── Image [B, 3, 224, 224]
└── Captions [B, K, max_len] (K=5 descriptions)

↓

CLIP Encoder (frozen or fine-tuned)
├── Image features [B, 768]
└── Text features [B, K, 768]

↓

Distribution Modeling MLPs
├── Image: μ_img, logvar_img [B, 768]
└── Text: K distributions → Merge → μ_text, logvar_text [B, 768]

↓

Output: Gaussian distributions for alignment
```

**Distribution Merging Methods:**

| Method | Description |
|--------|-------------|
| `moment_matching` | Minimizes KL divergence: μ = Σwᵢμᵢ, σ² = Σwᵢ(σᵢ² + μᵢ²) - μ² |
| `poe` | Product of Experts: τ = Στᵢ, μ = (Στᵢμᵢ)/τ |
| `simple` | Direct averaging |

**Loss Function:**
```
L_total = λ_contrastive × L_contrastive + λ_kl × L_kl + λ_var × L_var
```

- **Contrastive Loss**: CLIP-style bidirectional contrastive learning
- **KL Divergence**: Distribution shape alignment (symmetric/forward/reverse/wasserstein)
- **Variance Regularization** (optional): Prevents collapsed distributions

**Training:**
```bash
# Via main.py
python main.py --task train_dist_align

# Direct script
python scripts/train_dist_align.py
```

**Evaluation:**
```bash
python main.py --task eval_dist_align
```

**Training Pipeline:**
```
Full Dataset (118K samples)
    → 90% Train / 10% Validation (random_split, seed=42)
    → Each epoch: train → validate → save best checkpoint if improved
    → Early stopping if val loss doesn't improve for 3 epochs
```

**Data Processing Flow:**
```
Dataset → PIL Images + List[List[str]] (K=5 captions per image)
    → model.process_images(pil_images) → pixel_values [B, 3, 224, 224]
    → Flatten all captions [B*K] → model.process_text() → [B*K, 77]
    → Reshape → input_ids [B, K, 77], attention_mask [B, K, 77]
    → Forward pass → img_mu, img_logvar, text_mu, text_logvar
    → Combined loss (contrastive + KL + optional variance reg)
```

## Usage

### Main.py Options

```bash
python main.py --task <task_name> [options]
```

**Available tasks:**
- `train_clip_baseline`: Train CLIP baseline model
- `eval_clip_baseline`: Evaluate CLIP baseline model
- `train_dist_align`: Train distribution alignment model
- `eval_dist_align`: Evaluate distribution alignment model

**Common options:**
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--lr` / `--clip-lr`: Learning rate
- `--seed`: Random seed
- `--device`: Device to use
- `--checkpoint`: Path to checkpoint (for evaluation)
- `--recall-at-k`: Recall@K values to compute (default: 1 5 10)

**Distribution Alignment options:**
- `--val-split`: Validation set ratio (default: 0.1)
- `--early-stop-patience`: Early stopping patience in epochs (default: 3)
- `--no-early-stop`: Disable early stopping

### Training Examples

```bash
# CLIP Baseline with custom settings
python main.py --task train_clip_baseline \
    --epochs 10 \
    --batch-size 16 \
    --lr 1e-6 \
    --freeze-image

# Distribution Alignment (default: 10 epochs, 10% val split, early stopping)
python main.py --task train_dist_align

# Distribution Alignment with custom settings
python main.py --task train_dist_align \
    --epochs 10 \
    --batch-size 32 \
    --val-split 0.1 \
    --early-stop-patience 3
```

### Evaluation Examples

```bash
# CLIP Baseline
python main.py --task eval_clip_baseline \
    --checkpoint checkpoints/clip_baseline_best.pt \
    --recall-at-k 1 5 10

# Distribution Alignment
python main.py --task eval_dist_align \
    --checkpoint checkpoints/dist_align_best.pt \
    --recall-at-k 1 5 10
```

## Configuration

All hyperparameters are defined in `config.py`:

### CLIP Baseline Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `CLIP_BASELINE_EPOCHS` | Training epochs | 10 |
| `CLIP_BASELINE_BATCH_SIZE` | Batch size | 16 |
| `CLIP_BASELINE_LR` | Learning rate | 1e-6 |
| `CLIP_BASELINE_WEIGHT_DECAY` | Weight decay | 1e-4 |
| `CLIP_BASELINE_TEMPERATURE` | Temperature for contrastive loss | 0.07 |
| `CLIP_BASELINE_FREEZE_IMAGE` | Freeze image encoder | False |
| `CLIP_BASELINE_FREEZE_TEXT` | Freeze text encoder | False |

### Distribution Alignment Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `DIST_ALIGN_EPOCHS` | Training epochs | 10 |
| `DIST_ALIGN_BATCH_SIZE` | Batch size | 32 |
| `DIST_ALIGN_CLIP_LR` | CLIP learning rate | 1e-6 |
| `DIST_ALIGN_MLP_LR` | MLP learning rate | 1e-4 |
| `DIST_ALIGN_FREEZE_CLIP` | Freeze CLIP | True |
| `DIST_ALIGN_LAMBDA_CONTRASTIVE` | Contrastive loss weight | 1.0 |
| `DIST_ALIGN_LAMBDA_KL` | KL loss weight (primary) | 10.0 |
| `DIST_ALIGN_LAMBDA_VAR` | Variance regularization weight | 0.1 |
| `DIST_ALIGN_DISTRIBUTION_MERGING` | Merging method | "moment_matching" |
| `DIST_ALIGN_KL_TYPE` | KL divergence type | "symmetric" |
| `DIST_ALIGN_DROPOUT_RATE` | MLP dropout rate | 0.1 |
| `DIST_ALIGN_TARGET_VARIANCE` | Target variance for reg | 0.5 |
| `DIST_ALIGN_USE_VARIANCE_LOSS` | Enable variance regularization | False |

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `SEED` | Random seed | 42 |
| `NUM_CAPTIONS` | Captions per image | 5 |
| `NUM_WORKERS` | Data loading workers | 0 |
| `RECALL_AT_K` | Recall@K values | [1, 5, 10] |
| `EVAL_BATCH_SIZE` | Evaluation batch size | 32 |

## Output Files

### Checkpoints
- `checkpoints/clip_baseline_best.pt`: Best baseline checkpoint (lowest validation loss)
- `checkpoints/clip_baseline_last.pt`: Last baseline checkpoint
- `checkpoints/dist_align_best.pt`: Best distribution alignment checkpoint (lowest validation loss)
- `checkpoints/dist_align_last.pt`: Last distribution alignment checkpoint

### Evaluation Results
- `outputs/clip_baseline_eval_results.json`: Baseline metrics
- `outputs/dist_align_eval_results.json`: Distribution alignment metrics

Format:
```json
{
  "image_to_text": {
    "recall@1": 0.XXX,
    "recall@5": 0.XXX,
    "recall@10": 0.XXX
  },
  "text_to_image": {
    "recall@1": 0.XXX,
    "recall@5": 0.XXX,
    "recall@10": 0.XXX
  }
}
```

### Log Files
- `logs/main.log`: Main application logs
- `logs/train_clip_baseline.log`: Baseline training logs
- `logs/evaluate_clip_baseline.log`: Baseline evaluation logs
- `logs/train_dist_align.log`: Distribution alignment training logs
- `logs/evaluate_dist_align.log`: Distribution alignment evaluation logs

## Methods Comparison

| Aspect | CLIP Baseline | Distribution Alignment |
|--------|--------------|------------------------|
| **Representation** | Point embeddings (512-dim) | Gaussian distributions (768-dim) |
| **Multi-caption** | Random select 1 | Distribution merging |
| **Loss Function** | Contrastive only | Contrastive + KL + optional var reg |
| **Modality Gap** | May persist | Addressed through distributions |
| **Parameters** | All CLIP fine-tuned | CLIP (frozen by default) + MLP heads |
| **Learning Rate** | Single LR (1e-6) | Dual LR (CLIP: 1e-6, MLP: 1e-4) |

## Cross-Platform Migration

To migrate between Windows and Ubuntu:

1. **Copy the entire project** to the new system
2. **Edit `config.py`**: Update `PROJECT_ROOT`
3. **Verify setup**: Run `python config.py`

All paths will automatically adjust based on `PROJECT_ROOT`.

## Testing

See `examples/README.md` for detailed testing documentation.

```bash
# Quick sanity check
python examples/quick_test.py

# Full pipeline test
python examples/test_dist_align.py
```

## License

This project is for research and educational purposes.

## Citation

If you use this code, please cite:

```
@software{gaussian_image_distribution,
  title = {GaussianImageDistribution: CLIP Fine-tuning and Distribution-Based Alignment},
  author = {Your Name},
  year = {2026},
  note = {Baseline and distribution alignment methods for image-text representation learning}
}
```
