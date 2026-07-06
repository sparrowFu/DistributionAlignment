# GaussianImageDistribution

**MSDA (Multi-caption Semantic Distribution Alignment)** for image-text representation learning — model each image and text as a Gaussian distribution on a frozen CLIP backbone, and ground the learned variance in real caption diversity.

## Overview

This project implements **MSDA**, which represents each sample as a **general Gaussian** `N(μ, Σ)` on top of a frozen CLIP ViT-L/14 backbone, where `Σ = diag(σ²) + UUᵀ` (low-rank, rank `r`). The core idea is to **ground uncertainty in semantics**: the image variance is constrained to approximate the spread across multiple captions (`σ²_img ≈ Var(μ_captions)`), and a covariance direction-alignment loss aligns the image covariance subspace with the caption-deviation subspace.

MSDA evolves the earlier diagonal variant **UC-CL** (`r=0`, `Σ = diag(σ²)`) into a full-covariance formulation with a 3-stage training schedule. Full method details, including the `L_cov` training-stability analysis and fix, are in [methods.md](methods.md).

### Key Features

- **General Gaussian embeddings**: `N(μ, Σ)` with `Σ = diag(σ²) + UUᵀ`; `r=0` recovers the diagonal UC-CL variant
- **Semantic variance**: `σ²_img ≈ Var(μ_captions)` — uncertainty reflects real caption diversity instead of a hand-set floor
- **Covariance direction alignment** (`L_cov`): aligns the image covariance subspace with the caption-deviation subspace
- **Uncertainty-calibrated similarity**: `sim(x,y) = μ_x·μ_y / (τ·√(1+var_x)·√(1+var_y))`
- **Moment-matching merge**: fuses K per-caption distributions into one set distribution
- **3-stage training**: Warmup → Main → Full, gradually activating loss terms
- **5 comparison baselines** (B1–B4 + Ours) plus LLM VQA (B7/B8)

## Documentation

| Document | Description |
|----------|-------------|
| [methods.md](methods.md) | Current method (MSDA) — model structure, losses, staged training, stability fix |
| [experiments.md](experiments.md) | Experiment log — UC-CL vs MSDA results, `L_cov` crash analysis, P0 fix |
| [examples/README.md](examples/README.md) | Example scripts (quick sanity check, full pipeline test) |

> MSDA is the current method, an evolution of the early UC-CL diagonal variant (`r=0`); see `methods.md`.

## Project Structure

```
GaussianImageDistribution/
├── main.py                          # Unified task entry point
├── config.py                        # All hyperparameters & paths
├── data/
│   ├── caption_dataset.py           # MSCOCO image-caption dataset
│   ├── flickr30k_dataset.py         # Flickr30K cross-dataset evaluation
│   └── vqa_dataset.py               # VQA question-answer dataset
├── models/
│   ├── dist_align_model.py          # Ours: MSDA distribution alignment model
│   ├── clip_baseline.py             # B2: CLIP fine-tuning baseline
│   ├── clip_zero_shot.py            # B1: CLIP zero-shot VQA
│   ├── prolip_model.py              # B3: ProLIP probabilistic embeddings
│   ├── grove_model.py               # B4: GroVE GP-based posterior
│   ├── vqa_model.py                 # Unified VQA classification head
│   └── baseline_utils.py            # Shared utilities (merging, encoding)
├── losses/
│   ├── clip_losses.py               # CLIP contrastive loss
│   └── dist_align_losses.py         # MSDA losses (set-NCE + mu + var + cover + cov + reg)
├── utils/
│   ├── seed.py, io_utils.py, logger.py, metrics.py
│   ├── image_preprocess.py, image_utils.py
│   ├── calibration.py               # ECE/NLL/Brier/AUROC metrics
│   ├── retrieval.py                 # Recall@K / distribution feature extraction
│   └── cpu_affinity.py              # Exclude faulty CPU cores before subprocess
├── scripts/
│   ├── train_dist_align.py          # Ours training
│   ├── evaluate_dist_align.py       # Ours evaluation
│   ├── train_clip_baseline.py       # B2 training
│   ├── evaluate_clip_baseline.py    # B2 evaluation
│   ├── evaluate_clip_zero_shot.py   # B1 evaluation
│   ├── train_prolip.py              # B3 training
│   ├── evaluate_prolip.py           # B3 evaluation
│   ├── train_grove.py               # B4 training
│   ├── evaluate_grove.py            # B4 evaluation
│   ├── train_vqa.py                 # VQA classification head training
│   ├── eval_llm_vqa.py              # B7/B8 LLM VQA querying
│   ├── evaluate_llm_vqa.py          # B7/B8 LLM VQA metrics
│   ├── eval_calibration.py          # Exp3: Uncertainty calibration
│   ├── eval_ood.py                  # Exp4: OOD detection
│   ├── run_ablation.py              # Exp5: Ablation study
│   ├── eval_flickr30k.py            # Exp6: Flickr30K generalization
│   ├── eval_sigma_analysis.py       # Exp7: σ semantic analysis
│   └── visualize_modality_gap.py    # Exp8: Modality gap visualization
├── examples/
│   ├── quick_test.py                # Quick sanity check
│   └── test_dist_align.py           # Full pipeline test
├── PreTrainedModels/                # CLIP ViT-L/14 weights (local)
├── TrainDatasets/                   # MSCOCO + Flickr30K data
├── checkpoints/                     # Model checkpoints (auto-created)
├── outputs/                         # Evaluation results (auto-created)
└── logs/                            # Log files (auto-created)
```

## Installation

```bash
# Option A: pip
pip install -r requirements.txt

# Option B: conda
conda env create -f environments_Linux.yml        # Linux
conda env create -f environments_windows.yml      # Windows
```

Required: `torch>=2.0.0`, `transformers>=4.30.0`, `scikit-learn`, `matplotlib`, `tqdm`, `Pillow`.

## Data Setup

### MSCOCO Captions (Stage 1 + VQA)

```
TrainDatasets/mscoco_captions/
├── captions/train-00000-of-00001.parquet   # url, caption, image_file_name
├── images/                                  # image files
├── train/                                   # VQA train split
│   ├── questions.txt, img_filenames.txt, types.txt, answers.txt
└── test/                                    # VQA test split
    ├── questions_filtered.txt, img_filenames_filtered.txt, ...
```

### Flickr30K (Exp6, optional)

```
TrainDatasets/flickr30k/
├── images/
└── captions.txt
```

### Pre-trained Model

Download CLIP ViT-Large-Patch14 into `PreTrainedModels/clip-vit-large-patch14/`. All models load locally (`local_files_only=True`).

## Comparison Methods

| ID | Method | σ source | Trainable? |
|----|--------|----------|------------|
| B1 | CLIP Zero-Shot | — | No |
| B2 | CLIP Fine-Tune | — | Yes (CLIP) |
| B3 | ProLIP | learned log-variance head | Yes (MLP) |
| B4 | GroVE | GP posterior variance | Yes (inducing pts) |
| **Ours** | **MSDA** | **explicit `σ²≈Var(captions)` + covariance** | **Yes (MLP heads)** |
| B7 | Qwen-VL | — (LLM) | No (API) |
| B8 | Kimi-K2.5 | — (LLM) | No (API) |

All five comparison models (B1–B4 + Ours) share a unified 768-dim feature interface and the same VQA downstream head, for fair comparison.

## Usage

All tasks run through `main.py`.

### Stage 1: Image-Text Alignment Training & Evaluation

```bash
# Ours (MSDA)
python main.py --task train_dist_align
python main.py --task eval_dist_align

# B2: CLIP Fine-Tune
python main.py --task train_clip_baseline
python main.py --task eval_clip_baseline

# B3: ProLIP
python main.py --task train_prolip
python main.py --task eval_prolip

# B4: GroVE
python main.py --task train_grove
python main.py --task eval_grove

# B1: CLIP Zero-Shot
python main.py --task eval_clip_zero_shot
```

### Stage 2: VQA Downstream

```bash
# Train VQA classification head (supports B1-B4 + Ours)
python main.py --task train_vqa --model-type dist_align

# B7/B8: LLM VQA evaluation
python main.py --task eval_llm_vqa
python main.py --task evaluate_llm_vqa
```

### Experiments

```bash
python main.py --task eval_calibration                 # Exp3: uncertainty calibration (ECE/NLL/Brier/AUROC)
python main.py --task eval_ood                         # Exp4: OOD detection (sigma-based)
python main.py --task run_ablation --config all        # Exp5: ablation study
python main.py --task eval_flickr30k --model-type dist_align   # Exp6: cross-dataset generalization
python main.py --task eval_sigma_analysis              # Exp7: σ semantic analysis
python main.py --task visualize_gap                    # Exp8: modality gap visualization
```

## Configuration

All hyperparameters live in `config.py`.

### MSDA loss weights & schedule

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MSDA_COV_RANK` | 4 | low-rank covariance rank `r` (0 = diagonal / UC-CL) |
| `MSDA_TAU` | 0.07 | temperature for set-NCE similarity |
| `MSDA_LAMBDA_CTR` | 1.0 | set-level contrastive loss (main driver) |
| `MSDA_LAMBDA_MU` | 0.5 | mean-center alignment |
| `MSDA_LAMBDA_VAR` | 1.0 | variance semantic consistency (core: `σ²≈Var(captions)`) |
| `MSDA_LAMBDA_COVER` | 0.5 | multi-caption coverage |
| `MSDA_LAMBDA_COV` | 0.01 | covariance direction alignment (down-tuned from 0.1; see methods.md §6) |
| `MSDA_LAMBDA_REG` | 0.01 | variance regularization |
| `MSDA_GRAD_CLIP_NORM` | 1.0 | global grad-norm clip (stability guard) |
| `MSDA_STAGE_WARMUP/MAIN/FULL_FRAC` | 0.2 / 0.6 / 0.2 | 3-stage schedule fractions |

The 6-term loss is `L = λ_ctr·L_set-NCE + λ_mu·L_mu + λ_var·L_var + λ_cover·L_cover + λ_cov·L_cov + λ_reg·L_reg`. See [methods.md](methods.md) for the full formulation and the staged activation schedule.

### Training defaults

| Model | Epochs | LR | Notes |
|-------|--------|----|-------|
| Ours (MSDA) | 10 | MLP 5e-5 / CLIP 1e-6 | frozen CLIP, staged schedule |
| B2 CLIP Fine-Tune | 5 | 1e-6 | CLIP unfrozen |
| B3 ProLIP | 10 | 5e-5 | frozen CLIP, shares dist_align heads |
| B4 GroVE | 10 | 1e-3 | frozen CLIP, `num_inducing=128` |

## Output Files

### Checkpoints
- `checkpoints/dist_align_best.pt`, `dist_align_last.pt`
- `checkpoints/clip_baseline_best.pt`
- `checkpoints/prolip_best.pt`, `grove_best.pt`
- `checkpoints/vqa_{model_type}_best.pt`

### Evaluation Results
- `outputs/{model}_eval_results.json` — R@K retrieval metrics
- `outputs/{experiment}_results.json` — experiment-specific metrics
