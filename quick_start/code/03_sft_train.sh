#!/bin/bash
# Step 03 (GPU, ~25 min): SV-SFT of Qwen2.5-1.5B-Instruct on GSM8K (the bundled vector is GSM8K).
#   METHOD=mean_sv_add (default): bundled mean vector, h <- h + 0.5 v   (Table 1: 62.85)
#   METHOD=slerp: per-sample Slerp, alpha 0.1, needs step 02           (Table 1: 64.14)
#   METHOD=plain: no steering vector                                    (Table 1: 59.14)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
METHOD="${METHOD:-mean_sv_add}"
OUT="$WORK/sft_${METHOD}"

case "$METHOD" in
  plain)       SV_ARGS=() ;;
  mean_sv_add) SV_ARGS=(--sv "$(pick "$WORK/sft_sv_L${LAYER}_mean.pt" "$SFT_SV_MEAN")"
                        --op add --alpha 0.5) ;;
  slerp)       SV_ARGS=(--sv "$WORK/sft_sv_L${LAYER}_per_sample.pt" --op slerp --alpha 0.1) ;;
  *) echo "unknown METHOD=$METHOD" >&2; exit 1 ;;
esac
[ "$METHOD" = plain ] || SV_ARGS+=(--layer "$LAYER")

run $SV_TRAIN train --model "$STUDENT" --dataset "$DATASET" "${SV_ARGS[@]}" \
    --output_dir "$OUT" ${EXTRA_ARGS:-}
