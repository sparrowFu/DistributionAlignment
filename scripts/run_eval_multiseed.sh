#!/bin/bash
# Multi-seed retrieval evaluation runner.
#
# Matrix: models {clip_baseline, prolip, mcdisp_align(std), mcdisp_align(KL)}
#         x datasets {coco, flickr} x seeds {42, 43, 44}
# Metrics per task:
#   evaluate_*.py    -> bidirectional Recall@1/5/10 (model-native score)
#   eval_allhit.py   -> AllHit@5 + cover-rank mean ("mean coverage": average
#                       retrieval depth to cover an image's whole caption set)
# Sequential, one eval at a time; light GPU-memory guard before each task
# (evals are cheap but coexist with the training runner). Checkpoints not yet
# trained are skipped (rerun this script after training finishes to pick them
# up). .done markers make reruns resume, not repeat.
#
# Usage:
#   bash scripts/run_eval_multiseed.sh                 # trained models, 3 seeds
#   INCLUDE_ZS=1 bash scripts/run_eval_multiseed.sh    # + zero-shot references
#   SEEDS="42" DATASETS="coco" bash scripts/run_eval_multiseed.sh
set -u

PY=/home/xpfu/.conda/envs/CudaVersion128Fuxp/bin/python
export PYTHONUNBUFFERED=1
cd /home/xpfu/WorkSpace/DistributionAlignment || exit 1

SEEDS=${SEEDS:-"42 43 44"}
DATASETS=${DATASETS:-"coco flickr"}
EVAL_BS=${EVAL_BS:-64}
INCLUDE_ZS=${INCLUDE_ZS:-0}

RUN_DIR=logs/run_eval_multiseed
OUT_DIR=outputs/eval_multiseed
mkdir -p "$RUN_DIR" "$OUT_DIR"

# Free-GPU thresholds (MiB) before each eval task (eval-only footprints).
CLIP_EVAL_THRESH=6000
MCDISP_EVAL_THRESH=6000
PROLIP_EVAL_THRESH=12000

wait_gpu() {  # $1 = threshold MiB
  local need=$1 free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "${free:-0}" -ge "$need" ] && break
    sleep 60
  done
}

run_eval() {  # name thresh cmd...
  local name=$1 thresh=$2; shift 2
  if [ -f "$RUN_DIR/${name}.done" ]; then
    echo "[$(date '+%F %T')] SKIP $name (done)" >> "$RUN_DIR/runner.log"
    return 0
  fi
  wait_gpu "$thresh"
  echo "[$(date '+%F %T')] START $name" >> "$RUN_DIR/runner.log"
  "$@" > "$RUN_DIR/${name}.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] END $name exit=$rc" >> "$RUN_DIR/runner.log"
  if [ "$rc" -eq 0 ]; then
    touch "$RUN_DIR/${name}.done"
  else
    echo "$rc" > "$RUN_DIR/${name}.exit"
  fi
  sleep 10
}

recall_and_allhit() {  # tag model_key eval_script allhit_model thresh ckpt ds
  local tag=$1 model_key=$2 eval_script=$3 ah_model=$4 thresh=$5 ckpt=$6 ds=$7
  if [ ! -f "$ckpt" ]; then
    echo "[$(date '+%F %T')] MISSING $tag (no checkpoint: $ckpt)" >> "$RUN_DIR/runner.log"
    return 0
  fi
  run_eval "recall_${model_key}" "$thresh" \
    "$PY" "scripts/${eval_script}" --dataset "$ds" --batch-size "$EVAL_BS" \
    --checkpoint "$ckpt" \
    --output-path "$OUT_DIR/recall_${model_key}.json"
  run_eval "allhit_${model_key}" "$thresh" \
    "$PY" scripts/eval_allhit.py --model "$ah_model" --dataset "$ds" \
    --batch-size "$EVAL_BS" --checkpoint "$ckpt" \
    --output-path "$OUT_DIR/allhit_${model_key}.json"
}

for SEED in $SEEDS; do
  for DS in $DATASETS; do
    CK="checkpoints/seed${SEED}"

    recall_and_allhit "clip/${DS}/s${SEED}" "clip_${DS}_seed${SEED}" \
      evaluate_clip_baseline.py clip_baseline "$CLIP_EVAL_THRESH" \
      "$CK/clip_baseline_${DS}_best.pt" "$DS"

    recall_and_allhit "prolip/${DS}/s${SEED}" "prolip_${DS}_seed${SEED}" \
      evaluate_prolip.py prolip "$PROLIP_EVAL_THRESH" \
      "$CK/prolip_${DS}_best.pt" "$DS"

    recall_and_allhit "mcdisp/${DS}/s${SEED}" "mcdisp_${DS}_seed${SEED}" \
      evaluate_mcdisp_align.py mcdisp_align "$MCDISP_EVAL_THRESH" \
      "$CK/mcdisp_align_${DS}_best.pt" "$DS"

    recall_and_allhit "mcdisp_kl/${DS}/s${SEED}" "mcdisp_kl_${DS}_seed${SEED}" \
      evaluate_mcdisp_align.py mcdisp_align "$MCDISP_EVAL_THRESH" \
      "$CK/mcdisp_align_kl_${DS}_best.pt" "$DS"
  done
done

# Zero-shot references (deterministic -- one run per dataset, no seed loop)
if [ "$INCLUDE_ZS" = "1" ]; then
  for DS in $DATASETS; do
    run_eval "allhit_clip_zero_shot_${DS}" "$CLIP_EVAL_THRESH" \
      "$PY" scripts/eval_allhit.py --model clip_zero_shot --dataset "$DS" \
      --batch-size "$EVAL_BS" --output-path "$OUT_DIR/allhit_clip_zero_shot_${DS}.json"
    run_eval "allhit_prolip_zero_shot_${DS}" "$PROLIP_EVAL_THRESH" \
      "$PY" scripts/eval_allhit.py --model prolip_zero_shot --dataset "$DS" \
      --batch-size "$EVAL_BS" --output-path "$OUT_DIR/allhit_prolip_zero_shot_${DS}.json"
    run_eval "recall_clip_zero_shot_${DS}" "$CLIP_EVAL_THRESH" \
      "$PY" scripts/evaluate_clip_zero_shot.py --dataset "$DS" \
      --batch-size "$EVAL_BS" --output-path "$OUT_DIR/recall_clip_zero_shot_${DS}.json"
    run_eval "recall_prolip_zero_shot_${DS}" "$PROLIP_EVAL_THRESH" \
      "$PY" scripts/evaluate_prolip_zero_shot.py --dataset "$DS" \
      --batch-size "$EVAL_BS" --output-path "$OUT_DIR/recall_prolip_zero_shot_${DS}.json"
  done
fi

echo "[$(date '+%F %T')] AGGREGATE" >> "$RUN_DIR/runner.log"
"$PY" scripts/aggregate_eval_multiseed.py "$OUT_DIR" 2>&1 | tee "$OUT_DIR/summary.txt"
echo "[$(date '+%F %T')] ALL_EVAL_DONE (done markers: $(ls "$RUN_DIR"/*.done 2>/dev/null | wc -l))" >> "$RUN_DIR/runner.log"
