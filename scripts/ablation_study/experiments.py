"""Experiment definitions for the MCDisp_Align ablation study.

One entry per configuration in the experiment plan §4 (six core configs) and
§5 (caption-count sensitivity / supervision fairness), plus the §5.5 strict
rank-fixed companion. Every field maps onto ``MCDispAlignTrainConfig`` so all
variants share the SAME training code path, optimizer, schedule, manifests and
seeds (plan §8 unified training controls):

  * One knob per variant: leave-one-out loss weights, cov_rank, or the caption
    sampling regime -- nothing else moves.
  * Loss weights are NOT renormalized when a term is disabled (plan §8.7).
  * No per-variant tuning; no early stopping by default (fixed budget, §8.5).
  * Checkpoint selection is always the unified development multi-caption mR
    (``select_by="mr"``), on the same dev manifest with the same K for every
    config (plan §8 fallback clause).

``LAMBDA_COV_FULL`` is the single place to change the Full-model cov weight
(D1 decision: method target 0.2; drop to 0.01 here ONLY if the Full stage
proves unstable -- and then for every variant alike).
"""

from pathlib import Path
from typing import Dict, Optional

import config
from utils.mcdisp_align_trainer import MCDispAlignTrainConfig


# Common seed set (plan §9.1: 3 shared seeds; expand F/A1/A2/A3 to 5 later).
SEEDS = [42, 43, 44]
EXPAND_TO_5_SEEDS = ["full", "no_var", "no_cover", "no_cov"]

LAMBDA_COV_FULL = 0.2  # D1: single switch for the Full-model L_cov weight
K_EVAL = 5             # uniform dev protocol K for checkpoint selection


def _base_loss_weights() -> Dict[str, float]:
    """Full-model loss weights (plan §4.1 F): one source of truth."""
    return {
        "lambda_ctr": 1.0,
        "lambda_mu": 0.5,
        "lambda_var": 1.0,
        "lambda_cover_pos": 0.5,
        "lambda_cover_neg": 0.0,   # negative repulsion is NOT part of Full (§4.1)
        "lambda_cov": LAMBDA_COV_FULL,
        "lambda_reg": 0.01,
    }


def _mean_only_weights() -> Dict[str, float]:
    """A5/Mean-only: cosine set contrast + mean alignment, nothing else."""
    return {
        "lambda_ctr": 1.0,
        "lambda_mu": 0.5,
        "lambda_var": 0.0,
        "lambda_cover_pos": 0.0,
        "lambda_cover_neg": 0.0,
        "lambda_cov": 0.0,
        "lambda_reg": 0.0,
    }


# experiment name -> definition. ``K`` is the TRAINING caption count;
# ``sample`` is the caption sampling regime ("first" | "random").
EXPERIMENTS: Dict[str, Dict] = {
    # ---- §4 core matrix ----
    "full": {
        "desc": "F: Full model (all core losses, r=4)",
        "weights": _base_loss_weights(), "cov_rank": config.MCDISP_ALIGN_COV_RANK,
        "K": 5, "sample": "first", "uncertainty_sim": True,
    },
    "no_var": {
        "desc": "A1: w/o L_var (keep variance heads, cover, cov, reg, scorer)",
        "weights": {**_base_loss_weights(), "lambda_var": 0.0},
        "cov_rank": config.MCDISP_ALIGN_COV_RANK,
        "K": 5, "sample": "first", "uncertainty_sim": True,
    },
    "no_cover": {
        "desc": "A2: w/o L_cover_pos (negative repulsion already off)",
        "weights": {**_base_loss_weights(), "lambda_cover_pos": 0.0, "lambda_cover_neg": 0.0},
        "cov_rank": config.MCDISP_ALIGN_COV_RANK,
        "K": 5, "sample": "first", "uncertainty_sim": True,
    },
    "no_cov": {
        "desc": "A3: w/o L_cov but KEEP U (capacity-matched, U still in Mahalanobis)",
        "weights": {**_base_loss_weights(), "lambda_cov": 0.0},
        "cov_rank": config.MCDISP_ALIGN_COV_RANK,
        "K": 5, "sample": "first", "uncertainty_sim": True,
    },
    "diagonal_only": {
        "desc": "A4: diagonal covariance (remove U head entirely)",
        "weights": {**_base_loss_weights(), "lambda_cov": 0.0},
        "cov_rank": 0,
        "K": 5, "sample": "first", "uncertainty_sim": True,
    },
    "mean_only_kall": {
        "desc": "A5: Mean-only with all captions (cosine set contrast + mean alignment)",
        "weights": _mean_only_weights(),
        "cov_rank": 0,
        "K": 5, "sample": "first", "uncertainty_sim": False,
        "no_staged": True,  # every gated loss is 0; flat schedule is equivalent
    },
    # ---- §5 caption-count sensitivity / fairness ----
    "mean_only_k1": {
        "desc": "Mean-only-K1: one RANDOM valid caption per image per epoch",
        "weights": _mean_only_weights(),
        "cov_rank": 0,
        "K": 1, "sample": "random", "uncertainty_sim": False,
        "no_staged": True,
    },
    "full_k3": {
        "desc": "Full-K3: three RANDOM captions per image; r*=2 fixed for K-comparability (§5.5)",
        "weights": _base_loss_weights(), "cov_rank": 2,
        "K": 3, "sample": "random", "uncertainty_sim": True,
    },
    "full_kall_r2": {
        "desc": "Full with r*=2 (rank-fixed companion for full_k3; NOT the main Full)",
        "weights": _base_loss_weights(), "cov_rank": 2,
        "K": 5, "sample": "first", "uncertainty_sim": True,
    },
}


def build_train_config(
    experiment: str,
    seed: int,
    manifests_dir: Optional[Path] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: str = "cuda",
    ckpt_dir: Optional[Path] = None,
) -> MCDispAlignTrainConfig:
    """Map an experiment definition to a ``MCDispAlignTrainConfig``.

    All variants use the image-exclusive manifests, the same optimizer
    settings, no early stopping (fixed budget), and unified ``select_by="mr"``
    checkpoint selection on the dev manifest.
    """
    if experiment not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment {experiment!r}; known: {list(EXPERIMENTS)}")
    d = EXPERIMENTS[experiment]
    manifests_dir = Path(manifests_dir) if manifests_dir else (
        config.OUTPUT_DIR / "ablation_study" / "manifests")
    out_root = Path(ckpt_dir) if ckpt_dir else (
        config.OUTPUT_DIR / "ablation_study" / experiment)
    seed_dir = out_root / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    w = d["weights"]
    return MCDispAlignTrainConfig(
        dataset="coco",
        tag=f"{experiment}/s{seed}",
        epochs=epochs if epochs is not None else config.MCDISP_ALIGN_EPOCHS,
        batch_size=batch_size if batch_size is not None else config.MCDISP_ALIGN_BATCH_SIZE,
        freeze_clip=True,
        clip_lr=config.MCDISP_ALIGN_CLIP_LR,
        mlp_lr=config.MCDISP_ALIGN_MLP_LR,
        cov_rank=d["cov_rank"],
        # --- manifest-backed data (plan §3.1/§3.2) ---
        train_manifest=manifests_dir / "manifest_coco_train.json",
        dev_manifest=manifests_dir / "manifest_coco_dev.json",
        manifest_num_captions=d["K"],
        manifest_sample_mode=d["sample"],
        dev_num_captions=K_EVAL,
        # --- loss weights (NOT renormalized; §8.7) ---
        lambda_ctr=w["lambda_ctr"], lambda_mu=w["lambda_mu"],
        lambda_var=w["lambda_var"],
        lambda_cover_pos=w["lambda_cover_pos"], lambda_cover_neg=w["lambda_cover_neg"],
        lambda_cov=w["lambda_cov"], lambda_reg=w["lambda_reg"],
        tau=config.MCDISP_ALIGN_TAU,
        use_uncertainty_sim=d["uncertainty_sim"],
        # --- unified schedule/selection (§8) ---
        no_staged=d.get("no_staged", False),
        select_by="mr",
        no_early_stop=True,
        seed=seed,
        device=device,
        model_name=f"abl_{experiment}",
        checkpoint_dir=seed_dir,
        # final retrieval on the test manifest is run by the ablation eval
        # phase (full protocol), not by the trainer's built-in subset eval.
        eval_num_samples=None,
    )
