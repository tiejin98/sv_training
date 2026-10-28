#!/bin/bash
# Step 06 (GPU, ~30 min): distil Qwen3-8B (thinking) into Qwen2.5-1.5B-Instruct
# on GSM8K. Trains on teacher outputs that pass the verifier and fit in 2048
# tokens: 4829 of 7473.
#   METHOD=mean_sv_add (default)   Table 3: 67.55
#   METHOD=slerp                   Table 3: 67.40
#   METHOD=plain                   Table 3: 65.81
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
METHOD="${METHOD:-mean_sv_add}"
TJSON=$(pick "$WORK/teacher_think.json" "$TEACHER_JSON")
OUT="$WORK/distill_${METHOD}"

case "$METHOD" in
  plain)       SV_ARGS=() ;;
  mean_sv_add) SV_ARGS=(--sv "$WORK/distill_sv_L${LAYER}_mean.pt" --op add --alpha 0.5) ;;
  slerp)       SV_ARGS=(--sv "$WORK/distill_sv_L${LAYER}_per_sample.pt" --op slerp --alpha 0.1) ;;
  *) echo "unknown METHOD=$METHOD" >&2; exit 1 ;;
esac
[ "$METHOD" = plain ] || SV_ARGS+=(--layer "$LAYER")

run $SV_TRAIN train --model "$STUDENT" --dataset "$DATASET" --teacher_json "$TJSON" \
    "${SV_ARGS[@]}" --output_dir "$OUT" ${EXTRA_ARGS:-}
