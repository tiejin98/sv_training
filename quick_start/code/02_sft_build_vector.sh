#!/bin/bash
# Step 02 (optional for Mean-SV, needed for Per-sample Slerp; GPU, ~20 min):
# SFT steering vector = student hidden state on the GT answer minus on its own
# bootstrap prediction, mean-pooled over answer tokens (paper Eq. 6-7).
# The bundled $SFT_SV_MEAN is the output of this step for Mean-SV.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PREDS=$(pick "$WORK/student_bootpreds.json" "$BOOTPREDS")

run $SV_TRAIN acts --model "$STUDENT" --dataset "$DATASET" \
    --preds_json "$PREDS" --out "$WORK/sft_acts.pt"           # [7473, 28, 1536] x 2, ~2.5 GB

run $SV_TRAIN vector --acts "$WORK/sft_acts.pt" --layer "$LAYER" --kind mean \
    --out "$WORK/sft_sv_L${LAYER}_mean.pt"
run $SV_TRAIN vector --acts "$WORK/sft_acts.pt" --layer "$LAYER" --kind per_sample \
    --out "$WORK/sft_sv_L${LAYER}_per_sample.pt"
