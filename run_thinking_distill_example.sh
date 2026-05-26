#!/bin/bash
# End-to-end example: distill Qwen3-8B (thinking) → Qwen2.5-1.5B-Instruct on
# MATH, with Mean-SV-Add steering injection during SFT (raw v_mean × α=0.5
# at layer 15). Then evaluate the resulting student on MATH, both plain and
# with inference-time SV.
#
# Time estimate (single 80 GB GPU):
#   Step A. compute_distill_acts.py (student forward on teacher + bootstrap) ~15 min
#   Step B. SFT with mean-SV-add α=0.5 at L15 (1 epoch)                       ~30 min
#   Step C. eval (greedy, MATH-500 test)                                       ~3 min
#   Step D. inference-time SV eval (α=0.1)                                     ~3 min
#                                                              total ~ ~50 min
#
# Required environment:
#   - HuggingFace cache: HF_HOME=/path/to/large/disk (≥40 GB free for 1.5B + Qwen3-8B + dataset)
#   - HF_TOKEN with Qwen3 access if downloading
#
# This script uses the bundled assets:
#   teacher_Qwen3-8B-math-think.json  (Qwen3-8B thinking-mode teacher gen on MATH train, ~7.5K)
#   student_1.5B-math-bootpreds.json  (Qwen2.5-1.5B bootstrap preds on MATH train)

set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS=$SCRIPT_DIR
WORK=${WORK:-$SCRIPT_DIR/example_workdir_distill}
mkdir -p "$WORK"

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONPATH=$SCRIPTS:${PYTHONPATH:-}

STUDENT="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="math"
LAYER=15
ALPHA_TRAIN=0.5
ALPHA_INFER=0.1

TEACHER_JSON="$SCRIPT_DIR/teacher_Qwen3-8B-math-think.json"
BOOTPREDS_JSON="$SCRIPT_DIR/student_1.5B-math-bootpreds.json"

# Auto-unzip teacher json if only the .zip is present (bundled compressed to save space)
if [ ! -f "$TEACHER_JSON" ] && [ -f "${TEACHER_JSON}.zip" ]; then
  echo "[setup] unzipping ${TEACHER_JSON}.zip"
  unzip -o "${TEACHER_JSON}.zip" -d "$SCRIPT_DIR"
fi
ACTS_OUT="$WORK/distill_1.5B__Qwen3-8B-math-think_acts.pt"
ADAPTER_OUT="$WORK/student-1.5B-math-think-meansv-a${ALPHA_TRAIN}"

# ─────────────────────────────────────────────────────────────────
# Step A. Build student-space acts cache from the teacher JSON +
#         the bundled student bootstrap predictions.
# ─────────────────────────────────────────────────────────────────
echo "[A] computing distill acts cache → $ACTS_OUT"
python3 "$SCRIPTS/compute_distill_acts.py" \
    --student "$STUDENT" \
    --teacher_json "$TEACHER_JSON" \
    --student_preds_json "$BOOTPREDS_JSON" \
    --dataset "$DATASET" \
    --out "$ACTS_OUT"

# ─────────────────────────────────────────────────────────────────
# Step B. SFT the student via Mean-SV-Add at layer 15, α=0.5,
#         learning the teacher's thinking-mode outputs.
# ─────────────────────────────────────────────────────────────────
echo "[B] training SFT with raw mean-SV add α=$ALPHA_TRAIN @ layer $LAYER"
python3 "$SCRIPTS/sft.py" \
    --dataset "$DATASET" \
    --model_path "$STUDENT" \
    --method mean_sv_add \
    --alpha "$ALPHA_TRAIN" \
    --layer "$LAYER" \
    --acts_cache "$ACTS_OUT" \
    --distill_teacher_json "$TEACHER_JSON" \
    --output_dir "$ADAPTER_OUT" \
    --num_train_epochs 1

# ─────────────────────────────────────────────────────────────────
# Step C. Greedy eval (no inference SV) — plain accuracy.
# ─────────────────────────────────────────────────────────────────
echo "[C] eval @ α=0 (no inference SV)"
python3 "$SCRIPTS/infer_sv_vllm.py" \
    --model_path "$ADAPTER_OUT" \
    --base_model_name "$STUDENT" \
    --acts_cache "$ACTS_OUT" \
    --dataset "$DATASET" \
    --layer "$LAYER" \
    --alpha 0.0 \
    --out_name example-thinking-distill-math-meansv-a${ALPHA_TRAIN}-infer0

# ─────────────────────────────────────────────────────────────────
# Step D. Inference-time SV demo (same direction as training).
# ─────────────────────────────────────────────────────────────────
echo "[D] eval @ α=$ALPHA_INFER (inference-time SV)"
python3 "$SCRIPTS/infer_sv_vllm.py" \
    --model_path "$ADAPTER_OUT" \
    --base_model_name "$STUDENT" \
    --acts_cache "$ACTS_OUT" \
    --dataset "$DATASET" \
    --layer "$LAYER" \
    --alpha "$ALPHA_INFER" \
    --out_name example-thinking-distill-math-meansv-a${ALPHA_TRAIN}-infer${ALPHA_INFER}

echo "DONE. Predictions/metrics in outputs/main_paper_eval/example-thinking-distill-math-*"
