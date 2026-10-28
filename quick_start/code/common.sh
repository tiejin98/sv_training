# Shared settings for the quick-start steps. Override any variable from the
# environment, e.g.  WORK=/scratch/sv LAYER=15 bash quick_start/code/03_sft_train.sh
set -euo pipefail

QS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "$QS_DIR/.." && pwd)"
DATA="$QS_DIR/data"
WORK="${WORK:-$QS_DIR/generated}"
mkdir -p "$WORK"

STUDENT="${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}"
TEACHER="${TEACHER:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-gsm8k}"
LAYER="${LAYER:-15}"            # L/2 + 1 for the 28-layer Qwen2.5-1.5B

# Bundled intermediate results (see quick_start/README.md)
BOOTPREDS="$DATA/student_Qwen2.5-1.5B_gsm8k_bootpreds.json"
TEACHER_JSON="$DATA/teacher_Qwen3-8B_gsm8k_think.json"
SFT_SV_MEAN="$DATA/sv_Qwen2.5-1.5B_gsm8k_L15_mean.pt"
CKPT="$QS_DIR/checkpoints"        # trained SFT adapters: sft_mean_sv_add, sft_slerp
REF_EVAL="$QS_DIR/eval_outputs"   # paper eval outputs for all six runs

# Use the installed CLI, or fall back to the source tree.
if [ -z "${SV_TRAIN:-}" ]; then
  if command -v sv-train >/dev/null 2>&1; then
    SV_TRAIN="sv-train"
  else
    export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
    SV_TRAIN="python -m sv_training"
  fi
fi

# Prefer a freshly generated file in $WORK, else the bundled one.
pick() { if [ -e "$1" ]; then echo "$1"; else echo "$2"; fi; }

# Echo each command before running it, so it can be copied and adapted.
run() { echo "+ $*" >&2; "$@"; }
