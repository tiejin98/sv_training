#!/bin/bash
# End-to-end example: distill Qwen3-8B (thinking) → Qwen2.5-1.5B-Instruct on
# GSM8K, with Mean-SV-Add steering injection during SFT (raw v_mean × α=0.5
# at layer 15). Then evaluate the resulting student on GSM8K, both plain and
# with inference-time SV.
#
# Time estimate (single 80 GB GPU):
#   Step A. acts cache (student bootstraps + teacher acts) ............ ~25 min
#   Step B. SFT with mean-SV-add α=0.5 at L15 (1 epoch) ............... ~25 min
#   Step C. eval (greedy, full GSM8K test) ............................ ~3 min
#   Step D. inference-time SV eval (α=0.1) ............................ ~3 min
#                                                              total ~ ~55 min
#
# Required environment:
#   - HuggingFace cache: HF_HOME=/path/to/large/disk (≥30 GB free for 1.5B + Qwen3-8B)
#   - HF_TOKEN with Llama+Qwen3 access if downloading those models
#
# This script uses the bundled assets:
#   examples/teacher_Qwen3-8B-gsm8k-think.json  (teacher generations, ~7K problems)
#   examples/sv_1.5b_gsm8k_layer15_meansv.pt    (precomputed v_mean for inference SV)

set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE=$(cd "$SCRIPT_DIR/.." && pwd)
SCRIPTS=$SCRIPT_DIR
WORK=${WORK:-$RELEASE/example_workdir}
mkdir -p "$WORK"

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONPATH=$SCRIPTS:${PYTHONPATH:-}

STUDENT="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="gsm8k"
LAYER=15
ALPHA_TRAIN=0.5         # training-time SV-Add scale
ALPHA_INFER=0.1         # inference-time SV scale (for the demo eval)

TEACHER_JSON="$SCRIPT_DIR/teacher_Qwen3-8B-gsm8k-think.json"
ACTS_OUT="$WORK/distill_1.5B__Qwen3-8B-gsm8k-think_acts.pt"
ADAPTER_OUT="$WORK/student-1.5B-gsm8k-think-meansv-a${ALPHA_TRAIN}"

# ─────────────────────────────────────────────────────────────────
# Step A. Build student-space acts cache from the teacher JSON
# ─────────────────────────────────────────────────────────────────
echo "[A] computing distill acts cache → $ACTS_OUT"
python3 "$SCRIPTS/compute_distill_acts.py" \
    --student "$STUDENT" \
    --teacher_json "$TEACHER_JSON" \
    --student_preds_json "$SCRIPT_DIR/student_1.5B-gsm8k-bootpreds.json" \
    --dataset "$DATASET" \
    --out "$ACTS_OUT"

# ─────────────────────────────────────────────────────────────────
# Step B. SFT the student via Mean-SV-Add at layer 15, α=0.5
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
# Step C. Greedy eval (no inference SV) — plain accuracy
# ─────────────────────────────────────────────────────────────────
echo "[C] eval @ α=0 (no inference SV)"
python3 "$SCRIPTS/infer_sv_vllm.py" \
    --model_path "$ADAPTER_OUT" \
    --base_model_name "$STUDENT" \
    --acts_cache "$ACTS_OUT" \
    --dataset "$DATASET" \
    --layer "$LAYER" \
    --alpha 0.0 \
    --out_name example-thinking-distill-meansv-a${ALPHA_TRAIN}-infer0

# ─────────────────────────────────────────────────────────────────
# Step D. Inference-time SV demo (use bundled v_mean .pt for the same cell)
#         This shows whether adding the same-direction v at inference helps.
# ─────────────────────────────────────────────────────────────────
echo "[D] eval @ α=$ALPHA_INFER (inference-time SV using bundled v_mean)"
python3 "$SCRIPTS/infer_sv_vllm.py" \
    --model_path "$ADAPTER_OUT" \
    --base_model_name "$STUDENT" \
    --acts_cache "$ACTS_OUT" \
    --dataset "$DATASET" \
    --layer "$LAYER" \
    --alpha "$ALPHA_INFER" \
    --out_name example-thinking-distill-meansv-a${ALPHA_TRAIN}-infer${ALPHA_INFER}

echo "DONE. Results in $RELEASE/outputs/main_paper_eval/example-thinking-distill-*"
