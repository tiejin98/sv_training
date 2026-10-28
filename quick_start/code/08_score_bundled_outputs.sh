#!/bin/bash
# Step 08 (CPU, seconds): re-score the paper's GSM8K test outputs in eval_outputs/
# for all six runs (plain / mean_sv_add / slerp, for SFT and distillation).
# To score your own run instead: sv-train score --predictions <out_dir>/predictions.json --dataset gsm8k
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for run_dir in "$REF_EVAL"/*; do
  echo "== $(basename "$run_dir")"
  run $SV_TRAIN score --predictions "$run_dir/predictions.json" --dataset gsm8k
done
