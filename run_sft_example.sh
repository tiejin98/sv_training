#!/bin/bash
# End-to-end example: SV-SFT (no distillation) on Qwen2.5-1.5B-Instruct +
# GSM8K via Mean-SV-Add (raw v_mean × α=0.5 at layer 15). Then evaluate.
#
# Time on 1× 80 GB GPU:
#   Step A (skipped if --skip_acts): compute_acts.py student bootpreds + acts cache  ~25 min
#   Step B: sft.py mean_sv_add α=0.5 L15 (1 epoch)                                   ~25 min
#   Step C: infer_sv_vllm.py greedy eval (no inference SV)                            ~3 min
#   Step D: infer_sv_vllm.py inference-time SV at α=0.5 (same as training)            ~3 min
#
# Bundled asset that lets you skip Step A:
#   sv_1.5b_gsm8k_layer15_meansv_direction.pt  — precomputed v_mean
#                                               (for Mean-SV-Add via --mean_dir_override)

set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK=${WORK:-$SCRIPT_DIR/example_workdir_sft}
mkdir -p "$WORK"

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONPATH=$SCRIPT_DIR:${PYTHONPATH:-}

STUDENT="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="gsm8k"
LAYER=15
ALPHA=0.5
ACTS_OUT="$WORK/1.5B-gsm8k_acts.pt"
DIR_OVERRIDE="$SCRIPT_DIR/sv_1.5b_gsm8k_layer15_meansv_direction.pt"
ADAPTER_OUT="$WORK/student-1.5B-gsm8k-meansv-a${ALPHA}"

# ─────────────────────────────────────────────────────────────────
# Step A. (optional) acts cache. Skip with SKIP_ACTS=1 to use bundled v_mean.
# ─────────────────────────────────────────────────────────────────
if [ "${SKIP_ACTS:-0}" != "1" ]; then
  echo "[A] computing acts cache + bootpreds → $ACTS_OUT"
  python3 "$SCRIPT_DIR/compute_acts.py" \
      --model "$STUDENT" --dataset "$DATASET" --out_path "$ACTS_OUT"
  ACTS_ARG="--acts_cache $ACTS_OUT"
else
  echo "[A] SKIPPED (using bundled $DIR_OVERRIDE)"
  ACTS_ARG="--mean_dir_override $DIR_OVERRIDE"
fi

# ─────────────────────────────────────────────────────────────────
# Step B. SV-SFT: mean-SV-Add at layer 15, α=0.5.
# ─────────────────────────────────────────────────────────────────
echo "[B] training SV-SFT mean_sv_add α=$ALPHA @ layer $LAYER"
python3 "$SCRIPT_DIR/sft.py" \
    --dataset "$DATASET" --model_path "$STUDENT" \
    --method mean_sv_add --alpha "$ALPHA" --layer "$LAYER" \
    $ACTS_ARG \
    --output_dir "$ADAPTER_OUT" \
    --num_train_epochs 1

# ─────────────────────────────────────────────────────────────────
# Step C. Plain eval (no inference-time SV).
# ─────────────────────────────────────────────────────────────────
# infer_sv_vllm.py needs an acts cache to compute v_mean for inference SV.
# If we skipped acts in Step A, build a minimal one only for eval.
if [ ! -f "$ACTS_OUT" ]; then
  echo "[A'] building acts cache for eval (one-time)"
  python3 "$SCRIPT_DIR/compute_acts.py" \
      --model "$STUDENT" --dataset "$DATASET" --out_path "$ACTS_OUT"
fi

echo "[C] eval @ inference α=0 (plain)"
python3 "$SCRIPT_DIR/infer_sv_vllm.py" \
    --model_path "$ADAPTER_OUT" --base_model_name "$STUDENT" \
    --acts_cache "$ACTS_OUT" --dataset "$DATASET" \
    --layer "$LAYER" --alpha 0.0 \
    --out_name example-sft-meansv-a${ALPHA}-infer0

# ─────────────────────────────────────────────────────────────────
# Step D. Inference-time SV at the same α used in training.
# ─────────────────────────────────────────────────────────────────
echo "[D] eval @ inference α=$ALPHA (same direction as training)"
python3 "$SCRIPT_DIR/infer_sv_vllm.py" \
    --model_path "$ADAPTER_OUT" --base_model_name "$STUDENT" \
    --acts_cache "$ACTS_OUT" --dataset "$DATASET" \
    --layer "$LAYER" --alpha "$ALPHA" \
    --out_name example-sft-meansv-a${ALPHA}-infer${ALPHA}

echo "DONE. Predictions/metrics in outputs/main_paper_eval/example-sft-*"
