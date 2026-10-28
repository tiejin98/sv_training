#!/bin/bash
# Step 00 (CPU, seconds): look at the bundled intermediate results.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

run $SV_TRAIN inspect "$SFT_SV_MEAN" "$BOOTPREDS" "$TEACHER_JSON"
