#!/bin/bash
# SV-SFT quick start end to end (~30 min on one GPU with the bundled vector).
D="$(dirname "${BASH_SOURCE[0]}")"
set -e
bash "$D/00_inspect_data.sh"
[ "${METHOD:-mean_sv_add}" = slerp ] && bash "$D/02_sft_build_vector.sh"
bash "$D/03_sft_train.sh"
bash "$D/04_sft_eval.sh"
