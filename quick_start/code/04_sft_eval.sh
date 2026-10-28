#!/bin/bash
# Step 04 (GPU, ~3 min per run): greedy GSM8K test accuracy (1319 problems).
# The paper evaluates without any SV at test time (alpha = 0). Set
# INFER_ALPHA=0.5 to additionally try inference-time steering (Fig. 2c: it does not help).
# Without a step-03 adapter in $WORK, the bundled one in checkpoints/ is used
# (GSM8K, mean_sv_add and slerp only), so this step also runs without training.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
METHOD="${METHOD:-mean_sv_add}"
BUNDLED="$CKPT/sft_${METHOD}"
[ "$DATASET" = gsm8k ] || BUNDLED=""
ADAPTER="$(pick "$WORK/sft_${METHOD}" "$BUNDLED")"
if [ -z "$ADAPTER" ] || [ ! -d "$ADAPTER" ]; then
  echo "no adapter for METHOD=$METHOD: run 03_sft_train.sh first" >&2; exit 1
fi

run $SV_TRAIN eval --model "$ADAPTER" --base_model "$STUDENT" --dataset "$DATASET" \
    --out_dir "$WORK/eval/sft_${METHOD}_infer0"

if [ -n "${INFER_ALPHA:-}" ]; then
  run $SV_TRAIN eval --model "$ADAPTER" --base_model "$STUDENT" --dataset "$DATASET" \
      --sv "$(pick "$WORK/sft_sv_L${LAYER}_mean.pt" "$SFT_SV_MEAN")" --layer "$LAYER" \
      --alpha "$INFER_ALPHA" --out_dir "$WORK/eval/sft_${METHOD}_infer${INFER_ALPHA}"
fi
