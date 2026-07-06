# Examples

Example scripts for validating the project's core functionality.

## Scripts

### `quick_test.py` — Quick Sanity Check

Fast validation of model creation, forward pass, and backward pass with dummy data.

```bash
python examples/quick_test.py
```

Verifies:
- DistributionAlignmentModel creation and parameter counting
- Forward pass with dummy tensors
- Loss computation and gradient flow

### `test_dist_align.py` — Full Pipeline Test

Complete training and evaluation pipeline demonstration:

```bash
python examples/test_dist_align.py
```

Tests:
1. Model creation
2. Loss function computation (contrastive + KL + variance reg)
3. Training step with backprop
4. Full training epoch
5. Evaluation with Recall@K
6. Checkpoint save/load

## Configuration

Both scripts use minimal resources:
- **Samples**: 20 (mock data if real dataset unavailable)
- **Batch size**: 4
- **Epochs**: 2
- **Device**: CUDA or CPU

## Troubleshooting

### Pre-trained Model Not Found

Ensure CLIP ViT-Large-Patch14 is at `PreTrainedModels/clip-vit-large-patch14/`.

### CUDA Out of Memory

```bash
set CUDA_VISIBLE_DEVICES=
python examples/test_dist_align.py
```

## Full Training

After examples pass, start production training:

```bash
# Stage 1: Train all models
python main.py --task train_dist_align
python main.py --task train_clip_baseline
python main.py --task train_prolip
python main.py --task train_grove

# Stage 1: Evaluate
python main.py --task eval_dist_align
python main.py --task eval_clip_baseline
python main.py --task eval_clip_zero_shot
python main.py --task eval_prolip
python main.py --task eval_grove

# Stage 2: VQA
python main.py --task train_vqa --model-type dist_align

# Experiments
python main.py --task eval_calibration
python main.py --task eval_ood
python main.py --task run_ablation --config all
python main.py --task eval_flickr30k --model-type dist_align
python main.py --task eval_sigma_analysis
python main.py --task visualize_gap
```
