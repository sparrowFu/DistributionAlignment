"""Ablation study package for MCDisp_Align (experiment plan:
``docs/mcdisp_align_ablation_experiment_plan.md``). Modules:

  experiments.py        Experiment definitions (F, A1–A5, K-fairness configs)
  data_audit.py         Phase 0: image-exclusive manifests + audit
  feature_extraction.py Checkpoint -> features on a manifest split
  h1_semantic_range.py  H1 metrics (variance semantic-range validation)
  h2_coverage.py        H2 metrics (set coverage)
  h3_subspace.py        H3 metrics (low-rank direction)
  gaussian_scorer.py    Full-Gaussian-likelihood retrieval scorer
  interventions.py      Inference-time scorer table + sigma/U interventions
  stats.py              Seed aggregation, paired bootstrap, Holm correction
  run.py                CLI driver (phases)
"""
