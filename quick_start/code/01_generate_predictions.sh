#!/bin/bash
# Step 01 (optional, GPU + vLLM): regenerate the two bundled prediction files.
#   student bootstrap predictions  ~5 min   (Qwen2.5-1.5B-Instruct, greedy)
#   teacher outputs                ~1-2 h   (Qwen3-8B, thinking mode, greedy)
# Skip this step to use quick_start/data/*.json as-is.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

run $SV_TRAIN generate --model "$STUDENT" --dataset "$DATASET" --split train \
    --max_new_tokens 512 --out "$WORK/student_bootpreds.json"

run $SV_TRAIN generate --model "$TEACHER" --dataset "$DATASET" --split train \
    --enable_thinking 1 --max_new_tokens 4096 --out "$WORK/teacher_think.json"
