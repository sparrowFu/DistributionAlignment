#!/bin/bash
# Chain: wait for the two Flickr w/o-L_align trainings -> evaluate them ->
# train COCO UNFROZEN w/o-L_align (frozen COCO variant already exists from
# the loss_ablation queue) -> evaluate it.
set -u
PY=/home/xpfu/.conda/envs/CudaVersion128Fuxp/bin/python
export PYTHONUNBUFFERED=1
cd /home/xpfu/WorkSpace/DistributionAlignment || exit 1
RUN_DIR=logs/ablation_fl
CKPT=/mnt/disk1/xpfu/checkpoints
echo "[$(date '+%F %T')] waiting for flickr trainings" >> "$RUN_DIR/chain.log"

while pgrep -f "train_mcdisp_align.py.*abl_w_align_flickr" > /dev/null; do
  sleep 120
done
echo "[$(date '+%F %T')] flickr trainings done; evaluating" >> "$RUN_DIR/chain.log"

"$PY" scripts/eval_full_metrics.py --tag abl_walign_fl_fz --model mcdisp \
  --checkpoint "$CKPT/abl_w_align_flickr_fz_s42/mcdisp_align_kl_flickr_best.pt" \
  --dataset flickr >> "$RUN_DIR/eval_fl.log" 2>&1
echo "[$(date '+%F %T')] fl_fz eval rc=$?" >> "$RUN_DIR/chain.log"

"$PY" scripts/eval_full_metrics.py --tag abl_walign_fl_unf --model mcdisp \
  --checkpoint "$CKPT/abl_w_align_flickr_unf_s42/mcdisp_align_kl_flickr_best.pt" \
  --dataset flickr >> "$RUN_DIR/eval_fl.log" 2>&1
echo "[$(date '+%F %T')] fl_unf eval rc=$?" >> "$RUN_DIR/chain.log"

# frozen single-model Full reference on flickr (missing CSLS/dispersion cells)
if [ ! -f outputs/full_metrics/oldkl_s42_flickr.json ]; then
  "$PY" scripts/eval_full_metrics.py --tag oldkl_s42 --model mcdisp \
    --checkpoint checkpoints/seed42/mcdisp_align_kl_flickr_best.pt \
    --dataset flickr >> "$RUN_DIR/eval_fl.log" 2>&1
  echo "[$(date '+%F %T')] oldkl fl eval rc=$?" >> "$RUN_DIR/chain.log"
fi

echo "[$(date '+%F %T')] START coco unfrozen w/o L_align" >> "$RUN_DIR/chain.log"
mkdir -p "$CKPT/abl_w_align_coco_unf_s42"
"$PY" scripts/train_mcdisp_align.py --loss kl --dataset coco --seed 42 \
  --no-freeze-clip --lambda-kl 0 --lambda-cover-pos 0 --lambda-cov 0 --lambda-reg 0 \
  --select-by cosine_mr \
  --checkpoint-dir "$CKPT/abl_w_align_coco_unf_s42" \
  > "$RUN_DIR/w_align_coco_unf.log" 2>&1
echo "[$(date '+%F %T')] coco unf train rc=$?" >> "$RUN_DIR/chain.log"

"$PY" scripts/eval_full_metrics.py --tag abl_walign_coco_unf --model mcdisp \
  --checkpoint "$CKPT/abl_w_align_coco_unf_s42/mcdisp_align_kl_coco_best.pt" \
  --dataset coco >> "$RUN_DIR/eval_fl.log" 2>&1
echo "[$(date '+%F %T')] coco unf eval rc=$? ALL_DONE" >> "$RUN_DIR/chain.log"
