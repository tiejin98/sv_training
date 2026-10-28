#!/bin/bash
# SV-distillation quick start end to end (~65 min on one GPU).
# DATASET=gsm8k (default) or DATASET=math.
D="$(dirname "${BASH_SOURCE[0]}")"
set -e
bash "$D/00_inspect_data.sh"
bash "$D/05_distill_build_vector.sh"
bash "$D/06_distill_train.sh"
bash "$D/07_distill_eval.sh"
