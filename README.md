# SV_training


## What's here

```
README.md                              this file
sft.py                                 train a student via plain SFT or SV-SFT
                                       (also supports distillation via --distill_teacher_json)
compute_acts.py                        student bootstrap predictions + per-layer hidden caches → .pt
compute_distill_acts.py                build a teacher-vs-student acts cache for SV-distillation
teacher_gen.py                         generate a teacher's predictions on a dataset (vLLM)
infer_sv_vllm.py                       evaluate a (LoRA or merged) model with optional inference-time
                                       SV at one layer; supports GSM8K / MATH / MBPP / StrategyQA /
                                       AIME-24 / AIME-25

dataset_utils.py                       unified dataset adapters
_steering_helpers.py                   slerp_rotate + Trainer subclass that exposes labels to forward hooks
beegfs_safe_save.py                    /tmp staged write helper for flaky network filesystems (optional)
exploration_code/_dataset_utils_base.py  legacy base loader (wrapped by dataset_utils.py)

run_sft_example.sh                     end-to-end driver for the SFT example
run_thinking_distill_example.sh        end-to-end driver for the distillation example

# bundled working samples
sv_1.5b_gsm8k_layer15_meansv_direction.pt   precomputed v_mean for the SFT example
                                            (lets you skip Step A in run_sft_example.sh)
teacher_Qwen3-8B-math-think.json.zip        ~7.5K Qwen3-8B (thinking-mode) generations on MATH train
                                            (compressed; auto-unzipped by the example script,
                                             or `unzip teacher_Qwen3-8B-math-think.json.zip`)
student_1.5B-math-bootpreds.json            Qwen2.5-1.5B-Instruct bootstrap predictions on MATH train
                                            (used by compute_distill_acts.py in the distillation example)
sv_1.5b_gsm8k_layer15_meansv.pt             same v as the direction file but in
                                            {"raw_mean_vectors": {"block_15": v}} format
```


## Dependencies

Python 3.12 with `torch`, `transformers`, `peft`, `datasets`, `accelerate`,
`vllm`, `tqdm`. Optional `math-verify` for stricter LaTeX equivalence on
MATH/AIME. For `transformers`, we use a version <5.0.

---

## SFT

Plain SFT or SV-SFT. The model trains on the dataset GT answers, optionally with a steering vector injected during training.

### Run order

```
       compute_acts.py            (one run per student × dataset)
              │
              ▼  *_acts.pt + *_bootpreds.json
       sft.py                     --method ∈ {plain, slerp, mean_sv_add, per_inst_add,
              │                              random_gauss, sv_first_k, sv_multi_layer}
              ▼  output adapter
       infer_sv_vllm.py           (--alpha 0 = inference without SV (paper setting); --alpha >0 = inference-time SV)
```

### Quick start

```bash
# Full pipeline
bash run_sft_example.sh

# Skip Step A by reusing the bundled v_mean
SKIP_ACTS=1 bash run_sft_example.sh
```

This trains 1.5B-Instruct on GSM8K via Mean-SV-Add (α=0.5, layer 15), then
evaluates at inference α=0 and α=0.5. Output adapter:
`example_workdir_sft/student-1.5B-gsm8k-meansv-0.5/`.

### Per-script reference (SFT)

**1. `compute_acts.py` — student bootstraps + hidden caches**

```bash
python compute_acts.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset gsm8k \
    --out_path 1.5B-gsm8k_acts.pt
```

Produces `1.5B-gsm8k_acts.pt` with keys `gt_acts`, `pred_acts` of shape
`[N, num_layers, hidden]`, and writes `1.5B-gsm8k_acts_bootpreds.json` next
to it (the student's own predictions on the train set, used as the "Pred"
anchor for SV-SFT). The mean SV is `(gt_acts − pred_acts).mean(dim=0)[layer]`.

**2. `sft.py` — train**

```bash
# Plain SFT (no SV)
python sft.py --dataset gsm8k --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --method plain --output_dir outputs/plain-1.5B-gsm8k

# Mean-SV-Add (raw v_mean × α at one layer)
python sft.py --dataset gsm8k --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --method mean_sv_add --alpha 0.5 --layer 15 \
    --acts_cache 1.5B-gsm8k_acts.pt \
    --output_dir outputs/meansv-1.5B-gsm8k

# Per-instance Slerp SV-SFT (rotate h toward each instance's GT-direction)
python sft.py --dataset gsm8k --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --method slerp --alpha 0.1 --layer 15 \
    --acts_cache 1.5B-gsm8k_acts.pt \
    --output_dir outputs/slerp-1.5B-gsm8k
```

For Mean-SV-Add you can skip the acts cache and pass a precomputed
direction: `--mean_dir_override sv_1.5b_gsm8k_layer15_meansv_direction.pt`.

**Per-instance Slerp (`--method slerp`) needs the full per-instance deltas, so
the bundled direction file is not sufficient — you must run `compute_acts.py`
first to produce `*_acts.pt`.** Then:

```bash
python compute_acts.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset gsm8k --out_path 1.5B-gsm8k_acts.pt
python sft.py --dataset gsm8k --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --method slerp --alpha 0.1 --layer 15 \
    --acts_cache 1.5B-gsm8k_acts.pt \
    --output_dir outputs/slerp-1.5B-gsm8k
```

**3. `infer_sv_vllm.py` — evaluate**

```bash
# Plain (no inference-time SV)
python infer_sv_vllm.py \
    --model_path outputs/meansv-1.5B-gsm8k \
    --base_model_name Qwen/Qwen2.5-1.5B-Instruct \
    --acts_cache 1.5B-gsm8k_acts.pt \
    --dataset gsm8k --layer 15 --alpha 0.0 \
    --out_name meansv-1.5B-gsm8k-eval-a0

# Inference-time SV in the same direction
python infer_sv_vllm.py \
    --model_path outputs/meansv-1.5B-gsm8k \
    --base_model_name Qwen/Qwen2.5-1.5B-Instruct \
    --acts_cache 1.5B-gsm8k_acts.pt \
    --dataset gsm8k --layer 15 --alpha 0.5 \
    --out_name meansv-1.5B-gsm8k-eval-a0.5
```

Computes `v_mean` from the acts cache on the fly and adds `α · v_mean` to
the layer's residual stream during decode.

---

## Distillation

The student trains on a teacher's outputs,  with a steering vector.

### Run order

```
       teacher_gen.py             (one run per teacher × dataset)
              │
              ▼  teacher_*.json
       compute_acts.py            (one run per student × dataset; produces *_bootpreds.json)
              │
              ▼  *_bootpreds.json
       compute_distill_acts.py    (teacher_json + student bootpreds → distill acts cache)
              │
              ▼  distill_*_acts.pt
       sft.py  --distill_teacher_json TEACHER_JSON  --acts_cache DISTILL_ACTS
              │
              ▼  output adapter
       infer_sv_vllm.py           (--alpha 0 = plain; --alpha >0 = inference-time SV)
```

### Quick start

```bash
unzip teacher_Qwen3-8B-math-think.json.zip    # one-time (the example script also does this automatically)
bash run_thinking_distill_example.sh
```

Distills Qwen3-8B (thinking-mode) into Qwen2.5-1.5B-Instruct on **MATH** with
Mean-SV-Add (α=0.5, layer 15). Uses the bundled `teacher_Qwen3-8B-math-think.json`
and `student_1.5B-math-bootpreds.json`, so only the last three steps run:
`compute_distill_acts.py` → `sft.py` → `infer_sv_vllm.py` (×2). 


### Per-script reference (distillation-only)

`compute_acts.py`, `sft.py`, `infer_sv_vllm.py` work the same as in the SFT
section. Two extras:

**1. `teacher_gen.py` — generate teacher data (skip if you already have a teacher json)**

```bash
python teacher_gen.py \
    --model Qwen/Qwen3-8B \
    --dataset math \
    --split train \
    --enable_thinking 1 \
    --out teacher_Qwen3-8B--math--think.json
```

Output: a JSON list of `{"question": ..., "pred": ..., "idx": ...}` records.
For thinking-mode teachers the `<think>...</think>` chain is preserved in
`pred` so the student can learn the chain.

**2. `compute_distill_acts.py` — distillation acts cache**

```bash
python compute_distill_acts.py \
    --student Qwen/Qwen2.5-1.5B-Instruct \
    --teacher_json teacher_Qwen3-8B--math--think.json \
    --student_preds_json 1.5B-math_acts_bootpreds.json \
    --dataset math \
    --out distill_1.5B__Qwen3-8B--math--think.pt
```

Same shape as `compute_acts.py`'s output, but `gt_acts` is on
(prompt + teacher token) and `pred_acts` is on (prompt + student bootstrap).
Distillation SV = "shift the student's residual toward the teacher's manifold".

**3. `sft.py` with `--distill_teacher_json`**

Add `--distill_teacher_json PATH` to any `sft.py` invocation to switch from
GT-token SFT to teacher-token distillation. The acts cache passed to
`--acts_cache` should be the distillation cache (from
`compute_distill_acts.py`), not the plain SFT one.

```bash
# Plain distillation (no SV; learn teacher tokens only)
python sft.py --dataset math --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --method plain \
    --distill_teacher_json teacher_Qwen3-8B--math--think.json \
    --output_dir outputs/plain-distill-1.5B-math

# Mean-SV-Add distillation
python sft.py --dataset math --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --method mean_sv_add --alpha 0.5 --layer 15 \
    --acts_cache distill_1.5B__Qwen3-8B--math--think.pt \
    --distill_teacher_json teacher_Qwen3-8B--math--think.json \
    --output_dir outputs/meansv-distill-1.5B-math

# Per-instance Slerp distillation
python sft.py --dataset math --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --method slerp --alpha 0.1 --layer 15 \
    --acts_cache distill_1.5B__Qwen3-8B--math--think.pt \
    --distill_teacher_json teacher_Qwen3-8B--math--think.json \
    --output_dir outputs/slerp-distill-1.5B-math
```


Eval is the same `infer_sv_vllm.py` as in the SFT section; just point
`--acts_cache` at the distillation cache when you want inference-time SV.

---

## Notes

- Layer indices are 0-based and exclude the embedding (`block_0` = output of
  the first transformer block). Default = 15 (~middle of a 28-layer model).
- `sft.py` uses HuggingFace `peft` LoRA (rank 8, α 16, all-linear) by default.
- `beegfs_safe_save.py` exists because our shared scratch is BeeGFS and
  intermittently returns EREMOTEIO on large `torch.save`/`load`. On local SSD
  those calls are equivalent to plain `torch.save` / `torch.load`.
