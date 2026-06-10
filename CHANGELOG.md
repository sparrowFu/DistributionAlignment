# GaussianImageDistribution - Changelog

> Main documentation: [README.md](README.md)

---

## Documentation Index

| File | Description |
|------|-------------|
| [README.md](README.md) | Main project documentation (methods, usage, configuration) |
| [CODE_ANALYSIS.md](CODE_ANALYSIS.md) | Code logic analysis (data flow, module details) |
| [PROJECT_REVIEW.md](PROJECT_REVIEW.md) | Engineering code review (quality, coverage) |
| [examples/README.md](examples/README.md) | Example scripts documentation |

---

## Version Changelog

### v3.0 - Full Experiment Framework (UC-CL + 7 Baselines + 8 Experiments)

**Added - Core Method (UC-CL):**
- `losses/dist_align_losses.py` — UncertaintyCalibratedContrastiveLoss
  - L_calibrated_CL: σ-modulated similarity sharpness
  - L_consistency: σ²_img = Var(μ_captions)
  - L_variance: prevents σ collapse
- `models/dist_align_model.py` — Added `encode_image()`/`encode_text()` interface methods

**Added - Baselines (B3-B6):**
- `models/prolip_model.py` — B3: ProLIP probabilistic embeddings (implicit σ)
- `models/grove_model.py` — B4: GroVE GP-based posterior variance
- `models/icpe_model.py` — B5: ICPE training-free k-NN covariance
- `models/d2p_model.py` — B6: D2P distribution-to-point matching
- `models/baseline_utils.py` — Shared utilities (merge_distributions, encode_clip_features, init_heads_xavier)
- `scripts/train_prolip.py` — B3 training (contrastive + var reg, no consistency)
- `scripts/train_grove.py` — B4 training (contrastive on GP posterior mu)
- `scripts/train_d2p.py` — B6 training (MC distribution-to-point loss)
- `scripts/evaluate_prolip.py` — B3 R@K evaluation
- `scripts/evaluate_grove.py` — B4 R@K evaluation
- `scripts/evaluate_icpe.py` — B5 R@K evaluation (training-free)
- `scripts/evaluate_d2p.py` — B6 R@K evaluation

**Added - Experiments (Exp3-8):**
- `utils/calibration.py` — ECE/NLL/Brier Score/AUROC metrics
- `scripts/eval_calibration.py` — Exp3: Uncertainty calibration
- `scripts/eval_ood.py` — Exp4: OOD detection (sigma-based anomaly scoring)
- `scripts/run_ablation.py` — Exp5: 6 ablation configs + sensitivity analysis
- `scripts/eval_flickr30k.py` — Exp6: Flickr30K cross-dataset generalization
- `scripts/eval_sigma_analysis.py` — Exp7: σ semantic analysis (Pearson/Spearman)
- `scripts/visualize_modality_gap.py` — Exp8: t-SNE + gap bar chart + similarity histograms

**Added - Data:**
- `data/flickr30k_dataset.py` — Flickr30K dataset loader
- `data/vqa_dataset.py` — VQA question-answer dataset

**Added - VQA Pipeline:**
- `models/vqa_model.py` — Unified VQA classification head (all B1-B6 + Ours)
- `models/clip_zero_shot.py` — B1: CLIP zero-shot VQA
- `scripts/train_vqa.py` — VQA training script
- `scripts/eval_llm_vqa.py` — B7/B8: LLM VQA query via SiliconFlow API
- `scripts/evaluate_llm_vqa.py` — B7/B8: LLM VQA metrics

**Updated:**
- `config.py` — Added all B3-B6 configs, Flickr30K paths, Exp3-8 experiment configs, VQA paths
- `main.py` — 21 registered tasks (was 4), updated CLI choices and help
- `models/vqa_model.py` — Removed clip_zero_shot from VQAModel (handled separately), unified feature extraction via `encode_image()`/`encode_text()`

**Removed:**
- `models/freeze_align_model.py`, `fate_model.py`, `clip_ast_model.py` — Methods not in experiment plan
- `scripts/train_freeze_align.py`, `train_fate.py`, `train_clip_ast.py` — Corresponding training scripts
- `scripts/evaluate_freeze_align.py`, `evaluate_fate.py`, `evaluate_clip_ast.py` — Corresponding eval scripts
- `scripts/test_dataloader.py`, `test_image_load.py`, `verify_fix.py` — Debug scripts

### v2.1 - Training Pipeline Improvements

**Changed:**
- `config.py` — DIST_ALIGN_EPOCHS: 50→10, LAMBDA_KL: 0.5→10.0
- Training scripts — Added validation split, early stopping, best checkpoint saving

**Fixed:**
- `losses/dist_align_losses.py` — KL scale mismatch: `sum`→`mean` for dimensional averaging

### v2.0 - Distribution Alignment Module

**Added:**
- `models/dist_align_model.py` — DistributionAlignmentModel
- `losses/dist_align_losses.py` — DistributionAlignmentLoss, CombinedDistributionLoss
- `scripts/train_dist_align.py`, `evaluate_dist_align.py`

### v1.0 - CLIP Baseline Module

**Initial implementation:**
- `data/caption_dataset.py`, `models/clip_baseline.py`, `losses/clip_losses.py`
- `scripts/train_clip_baseline.py`, `evaluate_clip_baseline.py`
- `utils/` — seed, io_utils, image_utils, metrics, logger
- `config.py`, `main.py`

---

## Migration Notes

### From v2.x to v3.0

1. Removed models not in experiment plan (Freeze-Align, FATE, CLIP-AST)
2. New baselines B3-B6 have training and evaluation scripts
3. `main.py` now supports 21 tasks (was 4)
4. `VQAModel` no longer wraps `clip_zero_shot` (handled in `train_vqa.py` directly)
5. All models now have `encode_image(pixel_values)` and `encode_text(input_ids, attention_mask)` methods

### Cross-Platform

1. Edit `config.py`: Update `PROJECT_ROOT`
2. Run `python config.py` to verify
