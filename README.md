# GaussianImageDistribution

**MCDisp_Align (Multi-Caption Semantic Dispersion Guided Distribution Alignment)** for image-text representation learning — model each image and text as a Gaussian distribution on a frozen CLIP backbone, and ground the learned variance in real caption diversity.

## Overview

This project implements **MCDisp_Align**, which represents each sample as a **general Gaussian** `N(μ, Σ)` on top of a frozen CLIP ViT-L/14 backbone, where `Σ = diag(σ²) + UUᵀ` (low-rank, rank `r`). The core idea is to **ground uncertainty in semantics**: the image variance is constrained to approximate the spread across multiple captions (`σ²_img ≈ Var(μ_captions)`), and a covariance direction-alignment loss aligns the image covariance subspace with the caption-deviation subspace.

MCDisp_Align has two modes: a **diagonal mode** (`r=0`, `Σ = diag(σ²)`) and a **full-covariance mode** (`r>0`, `Σ = diag(σ²) + UUᵀ`, the default) with a 3-stage training schedule. Full method details, including the `L_cov` training-stability analysis and fix, are in [methods.md](methods.md).

### Key Features

- **General Gaussian embeddings**: `N(μ, Σ)` with `Σ = diag(σ²) + UUᵀ`; `r=0` selects the diagonal mode, `r>0` adds low-rank covariance
- **Semantic variance**: `σ²_img ≈ Var(μ_captions)` — uncertainty reflects real caption diversity instead of a hand-set floor
- **Covariance direction alignment** (`L_cov`): aligns the image covariance subspace with the caption-deviation subspace
- **Uncertainty-calibrated similarity**: `sim(x,y) = μ_x·μ_y / (τ·√(1+var_x)·√(1+var_y))`
- **Moment-matching merge**: fuses K per-caption distributions into one set distribution
- **3-stage training**: Warmup → Main → Full, gradually activating loss terms
- **3 comparison baselines** (B1–B3) + Ours (MCDisp_Align)

## Documentation

| Document | Description |
|----------|-------------|
| [methods.md](methods.md) | Current method (MCDisp_Align) — model structure, losses, staged training, stability fix |
| [experiments.md](experiments.md) | Experiment log — diagonal-mode vs full-covariance-mode results, `L_cov` crash analysis, P0 fix |

> MCDisp_Align is the project's method; `r=0` is its diagonal mode and `r>0` its full-covariance mode (see `methods.md`).

## Project Structure

```
GaussianImageDistribution/
├── main.py                          # Unified task entry point
├── config.py                        # All hyperparameters & paths
├── data/
│   ├── caption_dataset.py           # MSCOCO image-caption dataset
│   ├── flickr30k_dataset.py         # Flickr30K cross-dataset evaluation
│   └── vqa_expansion_dataset.py     # VQA-as-retrieval gemma-expansion dataset
├── models/
│   ├── mcdisp_align_model.py          # Ours: MCDisp_Align distribution alignment model
│   ├── clip_baseline.py             # B2: CLIP fine-tuning baseline
│   └── prolip_model.py              # B3: ProLIP probabilistic embeddings
├── losses/
│   ├── clip_losses.py               # CLIP contrastive loss
│   └── mcdisp_align_losses.py         # MCDisp_Align losses (set-NCE + mu + var + cover + cov + reg)
├── utils/
│   ├── seed.py, logger.py
│   ├── image_preprocess.py
│   ├── calibration.py               # AUROC / FPR@TPR (OOD scoring)
│   ├── retrieval.py                 # Recall@K computation
│   ├── cpu_affinity.py              # Exclude faulty CPU cores before subprocess
│   ├── dataset_registry.py, dataset_factory.py, mcdisp_align_trainer.py
│   └── eval_common.py, eval_results.py, retrieval_metrics.py, vqa_retrieval_metrics.py
├── scripts/
│   ├── train_mcdisp_align.py          # Ours training
│   ├── evaluate_mcdisp_align.py       # Ours evaluation
│   ├── train_clip_baseline.py       # B2 training
│   ├── evaluate_clip_baseline.py    # B2 evaluation
│   ├── evaluate_clip_zero_shot.py   # B1 evaluation
│   ├── train_prolip.py              # B3 training
│   ├── evaluate_prolip.py           # B3 evaluation
│   ├── evaluate_prolip_zero_shot.py # B3 zero-shot evaluation
│   ├── build_vqa_expansions.py      # Stage-2: build gemma caption expansions
│   ├── eval_vqa_retrieval.py        # Stage-2: VQA-as-retrieval evaluation
│   ├── eval_ood.py                  # Exp4: OOD detection
│   ├── run_ablation.py              # Exp5: Ablation v2 (train/eval/report/all 子命令 + --variant)
│   ├── eval_flickr30k.py            # Exp6: Flickr30K generalization
│   ├── eval_sigma_analysis.py       # Exp7: σ semantic analysis
│   ├── visualize_modality_gap.py    # Exp8: Modality gap visualization
│   └── diagnostics/                 # Standalone ad-hoc diagnostic scripts
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
| **Ours** | **MCDisp_Align** | **explicit `σ²≈Var(captions)` + covariance** | **Yes (MLP heads)** |

All four methods share a unified feature interface for fair comparison.

## Usage

All tasks run through `main.py`.

### Stage 1: Image-Text Alignment Training & Evaluation

```bash
# Ours (MCDisp_Align)
python main.py --task train_mcdisp_align
python main.py --task eval_mcdisp_align

# B2: CLIP Fine-Tune
python main.py --task train_clip_baseline
python main.py --task eval_clip_baseline

# B3: ProLIP
python main.py --task train_prolip
python main.py --task eval_prolip
python main.py --task eval_prolip_zero_shot

# B1: CLIP Zero-Shot
python main.py --task eval_clip_zero_shot
```

### Stage 2: VQA-as-Retrieval Downstream

```bash
# Build the gemma caption-expansion dataset (local gemma by default; --no-batch required)
python main.py --task build_vqa_expansions --split test --limit 0 --no-batch

# VQA-as-retrieval evaluation (one model, or --model all for all four)
python main.py --task eval_vqa_retrieval --model mcdisp_align
```

### Experiments

```bash
python main.py --task eval_ood                         # Exp4: OOD detection (sigma-based)
python scripts/run_ablation.py all                      # Exp5: ablation v2 (或: python main.py --task run_ablation --command train --variant full)
python main.py --task eval_flickr30k --model-type mcdisp_align   # Exp6: cross-dataset generalization
python main.py --task eval_sigma_analysis              # Exp7: σ semantic analysis
python main.py --task visualize_gap                    # Exp8: modality gap visualization
```

## Configuration

All hyperparameters live in `config.py`.

### MCDisp_Align loss weights & schedule

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MCDISP_ALIGN_COV_RANK` | 4 | low-rank covariance rank `r` (0 = diagonal mode) |
| `MCDISP_ALIGN_TAU` | 0.07 | temperature for set-NCE similarity |
| `MCDISP_ALIGN_LAMBDA_CTR` | 1.0 | set-level contrastive loss (main driver) |
| `MCDISP_ALIGN_LAMBDA_MU` | 0.5 | mean-center alignment |
| `MCDISP_ALIGN_LAMBDA_VAR` | 1.0 | variance semantic consistency (core: `σ²≈Var(captions)`) |
| `MCDISP_ALIGN_LAMBDA_COVER_POS` | 0.5 | L_cover positive coverage (methodology canonical) |
| `MCDISP_ALIGN_LAMBDA_COVER_NEG` | 0.0 | L_cover optional negative repulsion (off by default; sweep once stable) |
| `MCDISP_ALIGN_LAMBDA_COV` | 0.01 | covariance direction alignment (STABILITY-RUN value; official target 0.2 — see methods.md §6.5) |
| `MCDISP_ALIGN_LAMBDA_REG` | 0.01 | variance regularization |
| `MCDISP_ALIGN_GRAD_CLIP_NORM` | 1.0 | global grad-norm clip (stability guard) |
| `MCDISP_ALIGN_STAGE_WARMUP/FULL_FRAC` | 0.2 / 0.2 | 5-stage schedule: Warmup → Var-Bootstrap → Pos-Coverage → Neg-Repulsion → Full (middle split into thirds) |

The loss is `L = λ_ctr·L_set-NCE + λ_mu·L_mu + λ_var·L_var + λ_cover_pos·L_cover_pos + λ_cover_neg·L_cover_neg + λ_cov·L_cov + λ_reg·L_reg`. See [methods.md](methods.md) for the full formulation and the 5-stage activation schedule (§6.5).

### Training defaults

| Model | Epochs | LR | Notes |
|-------|--------|----|-------|
| Ours (MCDisp_Align) | 10 | MLP 5e-5 / CLIP 1e-6 | frozen CLIP, staged schedule |
| B2 CLIP Fine-Tune | 5 | 1e-6 | CLIP unfrozen |
| B3 ProLIP | 5 | 1e-6 | full ProLIP fine-tune, inclusion loss |

## Output Files

### Checkpoints
- `checkpoints/mcdisp_align_coco_best.pt`, `mcdisp_align_coco_last.pt`
- `checkpoints/clip_baseline_coco_best.pt`
- `checkpoints/prolip_coco_best.pt`

### Evaluation Results
- `outputs/{model}_eval_results.json` — R@K retrieval metrics
- `outputs/{experiment}_results.json` — experiment-specific metrics
