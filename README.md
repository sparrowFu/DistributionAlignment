# GaussianImageDistribution

Uncertainty-Calibrated Distributional Contrastive Learning (UC-CL) for image-text representation learning.

## Overview

This project implements **UC-CL**, a method that models image and text embeddings as Gaussian distributions on top of a frozen CLIP ViT-L/14 backbone. The core innovation is the **uncertainty-calibrated similarity** that uses learned variance (σ²) to modulate retrieval sharpness, and a **distributional consistency constraint** that forces σ²_img to approximate the variance across multiple captions.

### Key Features

- **Distributional Embeddings**: Each sample is represented as N(μ, σ²I) in 768-dim space
- **Uncertainty-Calibrated Similarity**: `sim(x,y) = μ_x·μ_y / (τ·√(1+var_x)·√(1+var_y))`
- **Distributional Consistency**: σ²_img ≈ Var(μ_captions), grounding σ in semantic diversity
- **Distribution Merging**: Moment matching to merge K caption distributions into one
- **7 Baselines**: B1-B6 + LLM VQA (B7/B8) for comprehensive comparison
- **8 Experiments**: Training, evaluation, calibration, OOD, ablation, generalization, analysis, visualization

## Project Structure

```
GaussianImageDistribution/
├── config.py                        # Centralized configuration (paths, hyperparameters)
├── main.py                          # Unified entry point (21 tasks)
├── data/
│   ├── caption_dataset.py           # MSCOCO image-caption dataset
│   ├── flickr30k_dataset.py         # Flickr30K cross-dataset evaluation
│   └── vqa_dataset.py               # VQA question-answer dataset
├── models/
│   ├── dist_align_model.py          # Ours: UC-CL distribution alignment model
│   ├── clip_baseline.py             # B2: CLIP fine-tuning baseline
│   ├── clip_zero_shot.py            # B1: CLIP zero-shot VQA
│   ├── prolip_model.py              # B3: ProLIP probabilistic embeddings
│   ├── grove_model.py               # B4: GroVE GP-based posterior
│   ├── icpe_model.py                # B5: ICPE training-free k-NN covariance
│   ├── d2p_model.py                 # B6: D2P distribution-to-point
│   ├── vqa_model.py                 # Unified VQA classification head
│   └── baseline_utils.py           # Shared utilities (merging, encoding)
├── losses/
│   ├── clip_losses.py               # CLIP contrastive loss
│   └── dist_align_losses.py         # UC-CL losses (calibrated CL + consistency + var reg)
├── utils/
│   ├── seed.py, io_utils.py, logger.py, metrics.py
│   ├── image_utils.py               # Image processing
│   └── calibration.py               # ECE/NLL/Brier/AUROC metrics
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
│   ├── evaluate_icpe.py             # B5 evaluation (training-free)
│   ├── train_d2p.py                 # B6 training
│   ├── evaluate_d2p.py              # B6 evaluation
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
├── checkpoints/                     # Model checkpoints (auto-created)
├── outputs/                         # Evaluation results (auto-created)
└── logs/                            # Log files (auto-created)
```

## Installation

```bash
pip install -r requirements.txt
```

Required: `torch>=2.0.0`, `transformers>=4.30.0`, `scikit-learn`, `matplotlib`, `tqdm`, `Pillow`

## Data Setup

### MSCOCO Captions (Stage 1 + VQA)

```
TrainDatasets/mscoco_captions/
├── captions/train-00000-of-00001.parquet   # url, caption, image_file_name
├── images/                                  # Image files
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

Download CLIP ViT-Large-Patch14 to `PreTrainedModels/clip-vit-large-patch14/`. All models load locally (`local_files_only=True`).

## Methods

### Ours: UC-CL (Uncertainty-Calibrated Distributional Contrastive Learning)

**Architecture**: Frozen CLIP ViT-L/14 + 4 MLP heads → (μ_img, logvar_img, μ_text, logvar_text)

**Loss**:
```
L = λ_cl × L_calibrated_CL + λ_consist × L_consistency + λ_var × L_variance
```

- **L_calibrated_CL**: σ modulates similarity sharpness via uncertainty-calibrated similarity
- **L_consistency**: Forces σ²_img = Var(μ_captions) (distributional consistency)
- **L_variance**: Prevents σ collapse (||σ² - target||²)

**Distribution Merging** (K=5 captions → 1 distribution):
```
μ_c = (1/K)Σμ_k,   σ²_c = (1/K)Σ(σ²_k + μ²_k) - μ²_c
```

### Baselines

| ID | Method | σ Source | Trainable? |
|----|--------|----------|------------|
| B1 | CLIP Zero-Shot | None | No |
| B2 | CLIP Fine-Tune | None | Yes (CLIP) |
| B3 | ProLIP | Implicit (inclusion loss) | Yes (MLP) |
| B4 | GroVE | GP posterior variance | Yes (inducing pts) |
| B5 | ICPE | k-NN covariance | No (training-free) |
| B6 | D2P | Text-side only | Yes (MLP) |
| B7 | Qwen-VL | N/A (LLM) | No (API) |
| B8 | Kimi-K2.5 | N/A (LLM) | No (API) |

## Usage

### Stage 1: Image-Text Alignment Training & Evaluation

```bash
# Ours (UC-CL)
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

# B5: ICPE (training-free)
python main.py --task eval_icpe

# B6: D2P
python main.py --task train_d2p
python main.py --task eval_d2p

# B1: CLIP Zero-Shot
python main.py --task eval_clip_zero_shot
```

### Stage 2: VQA Downstream

```bash
# Train VQA classification head (supports all B1-B6 + Ours)
python main.py --task train_vqa --model-type dist_align

# B7/B8: LLM VQA evaluation
python main.py --task eval_llm_vqa
python main.py --task evaluate_llm_vqa
```

### Experiments

```bash
# Exp3: Uncertainty calibration (ECE/NLL/Brier/AUROC)
python main.py --task eval_calibration

# Exp4: OOD detection (sigma-based anomaly scoring)
python main.py --task eval_ood

# Exp5: Ablation study (6 configs + sensitivity analysis)
python main.py --task run_ablation --config all

# Exp6: Flickr30K cross-dataset generalization
python main.py --task eval_flickr30k --model-type dist_align

# Exp7: σ semantic analysis (Pearson/Spearman correlation)
python main.py --task eval_sigma_analysis

# Exp8: Modality gap visualization (t-SNE + bar chart + histograms)
python main.py --task visualize_gap
```

## Configuration

All hyperparameters are in `config.py`. Key settings:

### UC-CL Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DIST_ALIGN_EPOCHS` | 10 | Training epochs |
| `DIST_ALIGN_BATCH_SIZE` | 32 | Batch size |
| `DIST_ALIGN_FREEZE_CLIP` | True | Freeze CLIP backbone |
| `DIST_ALIGN_USE_UC_CL` | True | Enable UC-CL loss |
| `DIST_ALIGN_LAMBDA_UC_CL` | 1.0 | Calibrated CL weight (λ_cl) |
| `DIST_ALIGN_LAMBDA_CONSIST` | 1.0 | Consistency loss weight (λ_consist) |
| `DIST_ALIGN_LAMBDA_UC_VAR` | 0.1 | Variance reg weight (λ_var) |
| `DIST_ALIGN_UC_TEMPERATURE` | 0.07 | Similarity temperature (τ) |

### Baseline Parameters

| Baseline | Epochs | LR | Key Params |
|----------|--------|----|-----------|
| B2 CLIP | 1 | 1e-6 | freeze_image/text=False |
| B3 ProLIP | 10 | 1e-6 | Same arch, no consistency |
| B4 GroVE | 10 | 1e-3 | num_inducing=128 |
| B6 D2P | 10 | 1e-4 | num_samples=10 |

## Output Files

### Checkpoints
- `checkpoints/dist_align_best.pt`, `dist_align_last.pt`
- `checkpoints/clip_baseline_best.pt`
- `checkpoints/prolip_best.pt`, `grove_best.pt`, `d2p_best.pt`
- `checkpoints/vqa_{model_type}_best.pt`

### Evaluation Results
- `outputs/{model}_eval_results.json` — R@K metrics
- `outputs/calibration/` — ECE/NLL/Brier/AUROC
- `outputs/ood_detection/` — AUROC/FPR@95TPR
- `outputs/ablation/` — Ablation results
- `outputs/sigma_analysis/` — Correlation results
- `outputs/modality_gap/` — Visualization figures
- `outputs/flickr30k/` — Cross-dataset results

## Cross-Platform

Edit `config.py` → `PROJECT_ROOT` to match your system path. All other paths adjust automatically.

## License

Research and educational purposes.
