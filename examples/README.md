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

## Configuration

The script uses minimal resources:
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
python examples/quick_test.py
```

## Full Training

After the example passes, start production training via `main.py` (see the project
[README.md](../README.md) for the full task list):

```bash
# Stage 1: Train + evaluate alignment models
python main.py --task train_dist_align
python main.py --task train_clip_baseline
python main.py --task train_prolip

python main.py --task eval_dist_align
python main.py --task eval_clip_baseline
python main.py --task eval_clip_zero_shot
python main.py --task eval_prolip

# Stage 2: VQA-as-retrieval
python main.py --task build_vqa_expansions --split test --limit 0 --no-batch
python main.py --task eval_vqa_retrieval --model dist_align

# Experiments
python main.py --task eval_ood
python main.py --task run_ablation --config all
python main.py --task eval_flickr30k --model-type dist_align
python main.py --task eval_sigma_analysis
python main.py --task visualize_gap
```
