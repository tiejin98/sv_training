#!/bin/bash
# Step 00 (CPU, seconds): look at the bundled intermediate results for $DATASET.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

FILES=()
for f in "$SFT_SV_MEAN" "$BOOTPREDS" "$TEACHER_JSON"; do [ -f "$f" ] && FILES+=("$f"); done
run $SV_TRAIN inspect "${FILES[@]}"
