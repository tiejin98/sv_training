#!/bin/bash
# Step 07 (GPU, ~3 min): greedy GSM8K test accuracy of the distilled student.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
METHOD="${METHOD:-mean_sv_add}"

run $SV_TRAIN eval --model "$WORK/distill_${METHOD}" --base_model "$STUDENT" \
    --dataset "$DATASET" --out_dir "$WORK/eval/distill_${METHOD}_infer0"
