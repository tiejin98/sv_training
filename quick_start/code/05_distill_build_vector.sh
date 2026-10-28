#!/bin/bash
# Step 05 (GPU, ~30 min): distillation steering vector, in the STUDENT's
# hidden space: student on the teacher's (thinking) output minus student on
# its own bootstrap prediction. Uses both bundled JSON files for $DATASET.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PREDS=$(pick "$WORK/student_bootpreds.json" "$BOOTPREDS")
TJSON=$(pick "$WORK/teacher_think.json" "$TEACHER_JSON")

run $SV_TRAIN acts --model "$STUDENT" --dataset "$DATASET" \
    --preds_json "$PREDS" --reference_json "$TJSON" --out "$WORK/distill_acts.pt"

run $SV_TRAIN vector --acts "$WORK/distill_acts.pt" --layer "$LAYER" --kind mean \
    --out "$WORK/distill_sv_L${LAYER}_mean.pt"
run $SV_TRAIN vector --acts "$WORK/distill_acts.pt" --layer "$LAYER" --kind per_sample \
    --out "$WORK/distill_sv_L${LAYER}_per_sample.pt"
