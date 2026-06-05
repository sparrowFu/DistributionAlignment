# GaussianImageDistribution - Changelog

> **Note**: Main documentation is in [README.md](README.md). This file contains the version changelog, documentation index, and migration notes.

---

## Documentation Index

| File | Description |
|------|-------------|
| [README.md](README.md) | Main project documentation (usage, configuration, architecture) |
| [CODE_ANALYSIS.md](CODE_ANALYSIS.md) | Code logic analysis (data flow, module details) |
| [PROJECT_REVIEW.md](PROJECT_REVIEW.md) | Engineering code review (quality assessment, optimization suggestions) |
| [examples/README.md](examples/README.md) | Example scripts documentation |

---

## Version Changelog

### v2.1 - Training Pipeline Improvements

**Changed:**
- `config.py` - `DIST_ALIGN_EPOCHS`: 50 → 10
- `config.py` - `DIST_ALIGN_LAMBDA_KL`: 0.5 → 10.0 (KL divergence as primary optimization target)
- `scripts/train_dist_align.py`:
  - Added validation set split (10% by default, configurable via `--val-split`)
  - Added early stopping mechanism (patience=3, configurable via `--early-stop-patience`)
  - Added `dist_align_best.pt` checkpoint saved on best validation loss
  - Training now logs both train and validation metrics per epoch
  - New CLI args: `--val-split`, `--early-stop-patience`, `--no-early-stop`
- `scripts/train_clip_baseline.py` - Aligned with dist_align style:
  - Added validation set split (`--val-split`) and early stopping (`--early-stop-patience`)
  - Added best checkpoint saving based on validation loss
  - Unified code structure: function naming, path handling, epoch counting
  - Removed redundant image_inputs loop, verbose logging
  - Logger name: `"train"` → `"train_clip_baseline"`
- `scripts/evaluate_clip_baseline.py` - Cleanup:
  - Removed redundant image_inputs loop
  - Simplified checkpoint loading
  - Logger name: `"eval"` → `"eval_clip_baseline"`
- `test_scripts/` renamed to `examples/` (as usage examples, not tests to ignore)
- `.gitignore` - Added Python/IDE/OS/Jupyter ignore rules

**Fixed:**
- `scripts/train_dist_align.py` - Fixed `args` not passed to `train_epoch`/`evaluate` functions
- `scripts/train_dist_align.py` / `scripts/evaluate_dist_align.py` - Fixed `str()` wrapping Path objects causing TypeError
- `losses/dist_align_losses.py` - Fixed KL divergence scale mismatch: changed `sum(dim=-1)` to `mean(dim=-1)` for dimensional averaging

### v2.0 - Distribution Alignment Module

**Added:**
- `models/dist_align_model.py` - DistributionAlignmentModel
  - Gaussian distribution modeling for image and text embeddings
  - Three distribution merging methods: moment_matching, poe, simple
  - Dual learning rate support (CLIP + MLP)
- `losses/dist_align_losses.py` - Distribution alignment losses
  - DistributionAlignmentLoss (contrastive + KL divergence)
  - VarianceRegularizationLoss
  - CombinedDistributionLoss
- `scripts/train_dist_align.py` - Distribution alignment training script
- `scripts/evaluate_dist_align.py` - Distribution alignment evaluation script
- `examples/quick_test.py` - Quick sanity check
- `examples/test_dist_align.py` - Full pipeline test

**Updated:**
- `config.py` - Added distribution alignment hyperparameters
- `main.py` - Added `train_dist_align` and `eval_dist_align` tasks
- `models/clip_baseline.py` - Added process_images/process_text methods

### v1.0 - CLIP Baseline Module

**Initial implementation:**
- `data/caption_dataset.py` - ImageCaptionDataset (MS-COCO format)
- `models/clip_baseline.py` - CLIPFineTuneBaseline
- `losses/clip_losses.py` - CLIP contrastive loss
- `scripts/train_clip_baseline.py` - Baseline training
- `scripts/evaluate_clip_baseline.py` - Baseline evaluation
- `utils/` - seed, io_utils, image_utils, metrics, logger
- `config.py` - Centralized configuration
- `main.py` - Unified entry point

---

## Migration Notes

### From v1.0 to v2.0

1. No breaking changes to existing CLIP Baseline functionality
2. New config parameters added (all have sensible defaults)
3. `main.py` supports 4 tasks (was 2)
4. All scripts now use `model.process_images()`/`model.process_text()` for data processing

### Cross-Platform Migration

1. Copy the entire project to the new system
2. Edit `config.py`: Update `PROJECT_ROOT`
3. Run `python config.py` to verify

See [README.md](README.md) for full details.
