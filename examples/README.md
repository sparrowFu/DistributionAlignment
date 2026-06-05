# Examples

This folder contains example scripts demonstrating how to use the project's models and pipeline.

## Example Scripts

### 1. `quick_test.py` - Quick Sanity Check

**Purpose**: Fast validation of basic model functionality

**What it does**:
- Creates CLIPFineTuneBaseline and DistributionAlignmentModel
- Runs forward pass with dummy data
- Computes loss and runs backward pass
- Verifies gradient flow

**Usage**:
```bash
python examples/quick_test.py
```

**When to use**:
- After code modifications, verify nothing is broken
- Before running full training, confirm environment is set up
- Quick sanity check that the pipeline works

### 2. `test_dist_align.py` - Complete Pipeline Example

**Purpose**: Full demonstration of the training and evaluation pipeline

**What it does**:
1. Model creation and parameter counting
2. Loss function computation (contrastive + KL + variance reg)
3. Single training step with backprop
4. Complete training epoch
5. Evaluation with Recall@K metrics
6. Checkpoint save/load

**Usage**:
```bash
python examples/test_dist_align.py
```

**When to use**:
- Before starting production training
- After major code changes
- To understand how the full pipeline works end-to-end

## Configuration

Both scripts use minimal resources:
- **Samples**: 20 mock samples (or real data if available)
- **Batch size**: 4
- **Epochs**: 2
- **Device**: CUDA (if available) or CPU

## Expected Output

### Quick Test

```
======================================================================
Quick Test - Distribution Alignment Model
======================================================================

Device: cuda
PyTorch version: 2.x.x

------------------------------------------------------------
Test 1: Model Creation
------------------------------------------------------------
  Model created successfully
  Total parameters: XXX,XXX,XXX
  Trainable parameters: X,XXX,XXX

------------------------------------------------------------
Test 2: Forward Pass
------------------------------------------------------------
  Input images shape: torch.Size([2, 3, 224, 224])
  Input captions shape: torch.Size([2, 5, 77])
  Forward pass successful
  Output shapes:
    img_features: torch.Size([2, 768])
    text_features: torch.Size([2, 768])
    img_mu: torch.Size([2, 768])
    img_logvar: torch.Size([2, 768])
    text_mu: torch.Size([2, 768])
    text_logvar: torch.Size([2, 768])

...

======================================================
All quick tests passed!
======================================================
```

### Complete Pipeline

```
======================================================================
Distribution Alignment - Test Suite
======================================================================
Device: cuda
PyTorch version: 2.x.x

Test Configuration:
  num_samples: 20
  batch_size: 4
  num_captions: 5
  epochs: 2
  device: cuda

...

======================================================
TEST SUMMARY
======================================================
  model_creation: PASS
  loss_function: PASS
  training_step: PASS
  training_epoch: PASS
  evaluation: PASS
  checkpoint: PASS
======================================================
Total: 6/6 tests passed
All tests passed!
```

## Troubleshooting

### CUDA Out of Memory

If you get CUDA OOM errors:
```bash
# Force CPU usage
set CUDA_VISIBLE_DEVICES=
python examples/test_dist_align.py
```

### Import Errors

Make sure you're running from the project root:
```bash
cd D:/code/causality/GaussianImageDistribution
python examples/test_dist_align.py
```

### Dataset Not Found

The scripts will automatically use mock data if the real dataset is not available. This is normal and expected for running examples.

### Pre-trained Model Not Found

If the CLIP model is not found at `PreTrainedModels/clip-vit-large-patch14/`, the scripts will fail. Download the model first (see main README.md for instructions).

## Checklist Before Production Training

Before running full training:
- [ ] Run `quick_test.py` - should pass
- [ ] Run `test_dist_align.py` - should pass
- [ ] Verify configuration with `python config.py`
- [ ] Check dataset is accessible at `TrainDatasets/mscoco_captions/`
- [ ] Ensure CLIP model is at `PreTrainedModels/clip-vit-large-patch14/`

## Next Steps

If the example scripts run successfully:
```bash
# Train CLIP Baseline
python main.py --task train_clip_baseline

# Train Distribution Alignment (with validation and early stopping)
python main.py --task train_dist_align

# Evaluate
python main.py --task eval_clip_baseline --checkpoint checkpoints/clip_baseline_best.pt
python main.py --task eval_dist_align --checkpoint checkpoints/dist_align_best.pt
```
