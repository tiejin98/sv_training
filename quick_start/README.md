# Quick Start

This folder contains intermediate results from the paper runs and one short script per pipeline step.
With them you can start the pipeline partway through instead of from the beginning.
Run every command from the repository root after installing the package:

```bash
pip install -e ".[vllm]"
```

The scripts call `sv-train`. If it is not installed, they fall back to `python -m sv_training` from this checkout.
Each script prints the exact command it runs, so you can copy that command and change it.

## Settings

| Setting | Student | Reference completions | Steering | Paper result |
| --- | --- | --- | --- | --- |
| SV-SFT | Qwen2.5-1.5B-Instruct | GSM8K ground-truth solutions | Mean-SV, α = 0.5, layer 15 | 62.85 (Plain 59.14, Slerp 64.14), Table 1 |
| SV-distillation | Qwen2.5-1.5B-Instruct | Qwen3-8B outputs, thinking mode | Mean-SV, α = 0.5, layer 15 | 67.55 (Plain 65.81, Slerp 67.40), Table 3 |

All numbers are greedy GSM8K test accuracy (%, 1,319 problems) with no SV at test time. The main tables use a
single seed (42). Across three seeds, SFT on 1.5B GSM8K gives Plain 58.81 ± 0.37 and Per-sample Slerp 62.75 ± 0.50
(Table 4), which gives a sense of run-to-run variation.
The bundled Slerp SFT adapter and its outputs come from a re-run on a different machine, which scored 64.06 instead of 64.14.

## Bundled files (`data/`)

| File | Content | Used by |
| --- | --- | --- |
| `sv_Qwen2.5-1.5B_gsm8k_L15_mean.pt` | `{"direction": [1536], "meta": ...}`, the mean over 7,473 train problems of `h(GT) − h(own prediction)` at layer 15, `‖v‖ = 11.45` | 03 |
| `student_Qwen2.5-1.5B_gsm8k_bootpreds.json` | `[{"idx", "pred"}]`: greedy Qwen2.5-1.5B-Instruct output for each GSM8K train problem | 02, 05 |
| `teacher_Qwen3-8B_gsm8k_think.json` | `[{"idx", "question", "gt_answer", "pred", "pred_correct", "n_tokens"}]`: Qwen3-8B thinking-mode outputs. `pred` keeps the `<think>` chain, and 5,150 of the 7,473 outputs are correct | 05, 06 |

Run `bash quick_start/code/00_inspect_data.sh` to print shapes, norms and counts for these files. It runs on CPU in a few seconds.

## Trained adapters (`checkpoints/`) and paper outputs (`eval_outputs/`)

`checkpoints/` holds the two SV-SFT LoRA adapters (r = 8, fp16, about 18 MB each; load the tokenizer from the base model):

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "quick_start/checkpoints/sft_mean_sv_add")
```

`eval_outputs/<run>/` holds `predictions.json` (`[{"question", "gt", "pred", "correct", "n_tokens"}]`) and
`metrics.json` for the GSM8K test runs reported in the paper:

| Run | Adapter in `checkpoints/` | Accuracy (`metrics.json`) |
| --- | --- | --- |
| `sft_plain` | — | 59.14 |
| `sft_mean_sv_add` | `sft_mean_sv_add/` | 62.85 |
| `sft_slerp` | `sft_slerp/` | 64.06 |
| `distill_plain` | — | 65.81 |
| `distill_mean_sv_add` | — | 67.55 |
| `distill_slerp` | — | 67.40 |

`bash quick_start/code/08_score_bundled_outputs.sh` re-scores all six files on CPU in seconds and gives the same numbers.

## Steps

Outputs go to `quick_start/generated/`. To write them somewhere else, set `WORK=/path` (an acts cache takes about 2.5 GB).
Other settings you can change: `STUDENT`, `TEACHER`, `LAYER`, `METHOD` (`mean_sv_add` | `slerp` | `plain`) and
`EXTRA_ARGS` (passed to `sv-train train`, e.g. `EXTRA_ARGS="--report_to wandb"`).

| Step | Script | Needs | Output | Time (1× 80 GB GPU) |
| --- | --- | --- | --- | --- |
| 00 | `00_inspect_data.sh` | bundled data | console summary | seconds, CPU |
| 01 | `01_generate_predictions.sh` *(optional)* | — | `student_bootpreds.json`, `teacher_think.json` (replace the bundled ones) | ~5 min + 1–2 h |
| 02 | `02_sft_build_vector.sh` *(optional)* | bootstrap preds | `sft_acts.pt`, `sft_sv_L15_mean.pt`, `sft_sv_L15_per_sample.pt` | ~20 min |
| 03 | `03_sft_train.sh` | SFT vector (bundled) | LoRA adapter `sft_<method>/` | ~25 min |
| 04 | `04_sft_eval.sh` | adapter from 03, or the bundled one for `mean_sv_add` / `slerp` | `eval/sft_<method>_infer0/metrics.json` | ~3 min |
| 05 | `05_distill_build_vector.sh` | bootstrap preds + teacher outputs (bundled) | `distill_acts.pt`, `distill_sv_L15_{mean,per_sample}.pt` | ~30 min |
| 06 | `06_distill_train.sh` | distillation vector | LoRA adapter `distill_<method>/` | ~30 min |
| 07 | `07_distill_eval.sh` | adapter | `eval/distill_<method>_infer0/metrics.json` | ~3 min |
| 08 | `08_score_bundled_outputs.sh` | bundled outputs | console accuracies | seconds, CPU |

If a step finds a regenerated file in `WORK` (e.g. from step 01, 02 or 03), it uses that file instead of the bundled one.

### Setting 1: SV-SFT

To evaluate the bundled adapters without training (one GPU, about 3 minutes each):

```bash
bash quick_start/code/04_sft_eval.sh                 # checkpoints/sft_mean_sv_add
METHOD=slerp bash quick_start/code/04_sft_eval.sh    # checkpoints/sft_slerp
```

To train from scratch:

```bash
bash quick_start/code/03_sft_train.sh     # Mean-SV with the bundled vector
bash quick_start/code/04_sft_eval.sh
```

The training step runs:

```bash
sv-train train --model Qwen/Qwen2.5-1.5B-Instruct --dataset gsm8k \
    --sv quick_start/data/sv_Qwen2.5-1.5B_gsm8k_L15_mean.pt --op add --alpha 0.5 --layer 15 \
    --output_dir quick_start/generated/sft_mean_sv_add
```

Other methods:

```bash
METHOD=plain bash quick_start/code/03_sft_train.sh && METHOD=plain bash quick_start/code/04_sft_eval.sh

bash quick_start/code/02_sft_build_vector.sh   # per-sample vectors need the hidden states
METHOD=slerp bash quick_start/code/03_sft_train.sh && METHOD=slerp bash quick_start/code/04_sft_eval.sh
```

To also test inference-time steering on the SV-trained adapter, run `INFER_ALPHA=0.5 bash quick_start/code/04_sft_eval.sh`.
Fig. 2(c) of the paper shows that this lowers accuracy compared with no SV at test time.

### Setting 2: SV-distillation

```bash
bash quick_start/code/05_distill_build_vector.sh
bash quick_start/code/06_distill_train.sh
bash quick_start/code/07_distill_eval.sh
```

The vector lives in the student's hidden space. It is the student's hidden state on the teacher's thinking-mode
output minus the student's hidden state on its own output. Training keeps the 5,150 teacher outputs that pass
the verifier (`--keep_only_correct 1`, the default). It then drops the 321 of those that are longer than 2,048
tokens (`--max_context_len`), which leaves 4,829 training examples. The vector itself is averaged over all 7,473 problems,
as in the paper runs.

`METHOD=plain` and `METHOD=slerp` work the same way as in Setting 1.
