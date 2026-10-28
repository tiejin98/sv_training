# Shared settings for the quick-start steps. Override any variable from the
# environment, e.g.  WORK=/scratch/sv LAYER=15 bash quick_start/code/03_sft_train.sh
set -euo pipefail

QS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "$QS_DIR/.." && pwd)"
DATA="$QS_DIR/data"
STUDENT="${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}"
TEACHER="${TEACHER:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-gsm8k}"     # gsm8k (SFT + distillation) or math (distillation)
LAYER="${LAYER:-15}"            # L/2 + 1 for the 28-layer Qwen2.5-1.5B
PYTHON="${PYTHON:-python}"

WORK="${WORK:-$QS_DIR/generated/$DATASET}"
mkdir -p "$WORK"

# Bundled intermediate results (see quick_start/README.md)
BOOTPREDS="$DATA/student_Qwen2.5-1.5B_${DATASET}_bootpreds.json"
TEACHER_JSON="$DATA/teacher_Qwen3-8B_${DATASET}_think.json"
SFT_SV_MEAN="$DATA/sv_Qwen2.5-1.5B_${DATASET}_L${LAYER}_mean.pt"   # gsm8k only
CKPT="$QS_DIR/checkpoints"        # trained GSM8K SFT adapters: sft_mean_sv_add, sft_slerp
REF_EVAL="$QS_DIR/eval_outputs"   # paper GSM8K eval outputs for all six runs

# The MATH teacher file is bundled zipped (121 MB unzipped); extract it into $WORK once.
if [ ! -f "$TEACHER_JSON" ] && [ -f "$TEACHER_JSON.zip" ]; then
  UNZIPPED="$WORK/$(basename "$TEACHER_JSON")"
  if [ ! -f "$UNZIPPED" ]; then
    echo "[setup] extracting $TEACHER_JSON.zip -> $UNZIPPED" >&2
    "$PYTHON" -c 'import sys, zipfile; z = zipfile.ZipFile(sys.argv[1]); open(sys.argv[2], "wb").write(z.read(z.namelist()[0]))' \
      "$TEACHER_JSON.zip" "$UNZIPPED"
  fi
  TEACHER_JSON="$UNZIPPED"
fi

# Use the installed CLI, or fall back to the source tree.
if [ -z "${SV_TRAIN:-}" ]; then
  if command -v sv-train >/dev/null 2>&1; then
    SV_TRAIN="sv-train"
  else
    export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
    SV_TRAIN="$PYTHON -m sv_training"
  fi
fi

# Prefer a freshly generated file in $WORK, else the bundled one.
pick() { if [ -e "$1" ]; then echo "$1"; else echo "$2"; fi; }

# Echo each command before running it, so it can be copied and adapted.
run() { echo "+ $*" >&2; "$@"; }
