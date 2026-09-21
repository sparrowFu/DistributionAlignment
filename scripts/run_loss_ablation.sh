#!/bin/bash
# Loss ablation per paper/main.md Eq. (overall objective):
#   L = L_match + L_align,  L_align = KL + cover + direction + var-reg
# Variants (frozen backbone, COCO, seed42, select-by cosine_mr, 10 epochs):
#   full      : all defaults (ctr=1 kl=1 cover=0.5 cov=0.01 reg=0.01)
#   w/o match : --lambda-ctr 0
#   w/o align : --lambda-kl 0 --lambda-cover-pos 0 --lambda-cov 0 --lambda-reg 0
#   w/o kl    : --lambda-kl 0
#   w/o cover : --lambda-cover-pos 0
#   w/o dir   : --lambda-cov 0
#   w/o reg   : --lambda-reg 0
# Each run trains, then evaluates the full metric suite (retrieval + AllHit
# + coverage + dispersion). Sequential; .done markers make reruns resume.
set -u
PY=/home/xpfu/.conda/envs/CudaVersion128Fuxp/bin/python
export PYTHONUNBUFFERED=1
cd /home/xpfu/WorkSpace/DistributionAlignment || exit 1

RUN_DIR=logs/loss_ablation
CKPT_ROOT=/mnt/disk1/xpfu/checkpoints
mkdir -p "$RUN_DIR"
DS=coco
SEED=42

run_variant() {  # tag extra-args...
  local tag=$1; shift
  local dir="$CKPT_ROOT/abl_${tag}_${DS}_s${SEED}"
  if [ -f "$RUN_DIR/${tag}.done" ]; then
    echo "[$(date '+%F %T')] SKIP $tag (done)" >> "$RUN_DIR/runner.log"
    return 0
  fi
  mkdir -p "$dir"
  echo "[$(date '+%F %T')] START $tag" >> "$RUN_DIR/runner.log"
  "$PY" scripts/train_mcdisp_align.py --loss kl --dataset $DS --seed $SEED \
    --freeze-clip --select-by cosine_mr --checkpoint-dir "$dir" "$@" \
    > "$RUN_DIR/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')] TRAIN FAIL $tag rc=$rc" >> "$RUN_DIR/runner.log"
    return 1
  fi
  "$PY" scripts/eval_full_metrics.py --tag abl_${tag} --model mcdisp \
    --checkpoint "$dir/mcdisp_align_kl_${DS}_best.pt" --dataset $DS \
    >> "$RUN_DIR/${tag}.log" 2>&1
  echo "[$(date '+%F %T')] END $tag eval_rc=$?" >> "$RUN_DIR/runner.log"
  touch "$RUN_DIR/${tag}.done"
}

# the two headline losses only (user scope)
run_variant w_match --lambda-ctr 0
run_variant w_align --lambda-kl 0 --lambda-cover-pos 0 --lambda-cov 0 --lambda-reg 0

echo "[$(date '+%F %T')] ALL_DONE" >> "$RUN_DIR/runner.log"
