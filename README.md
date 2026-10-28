# Steering Vectors as a Training Signal in LLM Post-Training

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/sv-training.svg)](https://pypi.org/project/sv-training/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/tiejin98/sv_training/actions/workflows/ci.yml/badge.svg)](https://github.com/tiejin98/sv_training/actions/workflows/ci.yml)

Code for the NeurIPS 2026 paper **"Steering Vectors as a Training Signal in LLM
Post-Training"** (Tiejin Chen, Maunil Vyas, Huaiyuan Yao, Hua Wei; Arizona State University).

Steering vectors (SVs) are usually added to a frozen model's hidden states at
inference time. This repository uses them **during training**: an SV is
injected into the residual stream at one layer, on answer tokens only, while
the model is fine-tuned with teacher forcing. The direction points from where
the model's own outputs sit to where the reference outputs (dataset answers or
a teacher's outputs) sit, which reduces the off-policy gap of SFT and
distillation. Nothing is injected at test time, so any gain is stored in the weights.

The `sv-training` package covers the supervised regime of the paper (SFT and
distillation). It works with any steering vector at any layer(s), so you can
also train with directions you computed yourself.

## Installation

```bash
pip install "sv-training[vllm]"     # training + vLLM generation/evaluation
sv-train --help
```

`pip install sv-training` without `[vllm]` installs only what training needs
(`torch`, `transformers<5`, `peft`, `datasets`, `accelerate`). Optional extras:
`[math]` adds `math-verify` for stricter AIME scoring, `[wandb]` adds logging, `[dev]` adds test and build tools.

The quick-start data lives in this repository, not in the PyPI package. To run the quick start, clone the repo:

```bash
git clone https://github.com/tiejin98/sv_training.git
cd sv_training
pip install -e ".[vllm]"
```

## Method in one picture

```text
             teacher forcing, frozen model                      training (LoRA)
prompt + reference answer ──► h_ref  (layer l, answer-token mean)
prompt + own prediction   ──► h_pred                    prompt + label ──► block l ──► ... ──► loss
                                │                                          │
          v = h_ref − h_pred ───┴──────────── inject on answer tokens ─────┘
          (per sample, or its mean)          add:   h ← h + α·v
                                             slerp: h ← Slerp(h, v̂; α)  (norm-preserving)
```

| Paper name | Direction | Operation | α | CLI |
| --- | --- | --- | --- | --- |
| Mean-SV | dataset mean of `h_ref − h_pred` | add | 0.5 | `--method mean_sv_add` or `--sv mean.pt --op add` |
| Per-sample Slerp | per-sample unit `h_ref − h_pred` | slerp | 0.1 | `--method slerp` or `--sv per_sample.pt --op slerp` |
| Per-sample additive | per-sample unit `h_ref − h_pred` | add | 0.1 | `--method per_inst_add` |
| Multi-layer | per-sample, at several layers | slerp | 0.1 | `--method sv_multi_layer --multi_layers 10,15,20` |
| Random control | random Gaussian per sample | slerp | 0.1 | `--method random_gauss` |

The default layer is `l = L/2 + 1`, which is 15 for the 28-layer Qwen2.5 models.

## Quick Start

The [`quick_start/`](quick_start/) folder holds intermediate results from the
paper runs, so you can start in the middle of the pipeline instead of
regenerating everything. It covers these settings:

| Setting | Student | Reference | Bundled files | Paper result (test acc.) |
| --- | --- | --- | --- | --- |
| **SV-SFT** | Qwen2.5-1.5B-Instruct | GSM8K GT answers | Mean-SV at layer 15, student bootstrap predictions | Plain 59.14 → Mean-SV **62.85**, Slerp **64.14** (Table 1) |
| **SV-distillation, GSM8K** | Qwen2.5-1.5B-Instruct | Qwen3-8B, thinking mode | teacher outputs (7,473; 5,150 pass the verifier), student bootstrap predictions | Plain 65.81 → Mean-SV **67.55**, Slerp **67.40** (Table 3) |
| **SV-distillation, MATH** | Qwen2.5-1.5B-Instruct | Qwen3-8B, thinking mode | teacher outputs (12,000; 5,772 pass the verifier), student bootstrap predictions | Plain 21.20 → Mean-SV **31.20**, Slerp **33.60** (Table 3, MATH-500) |

Bundled files (`quick_start/data/`):

| File | What it is | Produced by |
| --- | --- | --- |
| `sv_Qwen2.5-1.5B_gsm8k_L15_mean.pt` | Mean-SV for SFT, layer 15, `‖v‖ = 11.45`, shape `[1536]` | step 02 |
| `student_Qwen2.5-1.5B_gsm8k_bootpreds.json` | greedy Qwen2.5-1.5B-Instruct outputs on the 7,473 GSM8K train problems | step 01 |
| `teacher_Qwen3-8B_gsm8k_think.json` | Qwen3-8B thinking-mode outputs on GSM8K train, with `<think>` chains and `pred_correct` flags | step 01 |
| `student_Qwen2.5-1.5B_math_bootpreds.json`, `teacher_Qwen3-8B_math_think.json.zip` | the same two files for the 12,000 MATH train problems (the teacher file is zipped and extracted on first use) | step 01 |

The folder also holds the trained SV-SFT LoRA adapters for Mean-SV and Slerp in `quick_start/checkpoints/`, and the
GSM8K test predictions for all six runs in the table in `quick_start/eval_outputs/`. The Slerp SFT adapter and its outputs
come from a re-run on a different machine, which scored 64.06 instead of 64.14.

Each step is a short script in [`quick_start/code/`](quick_start/code/). Run the scripts from the repository root.
Outputs go to `quick_start/generated/<dataset>/`; set `WORK=...` to change that.

```bash
# 00  (CPU, seconds) inspect the bundled files
bash quick_start/code/00_inspect_data.sh
# 08  (CPU, seconds) re-score the paper's test outputs for all six runs
bash quick_start/code/08_score_bundled_outputs.sh

# ---- Setting 1: SV-SFT --------------------------------------------------
# 01  (optional) regenerate the bundled predictions        -> skip, bundled
# 02  (optional) build the SFT vector from hidden states   -> skip, bundled
bash quick_start/code/03_sft_train.sh      # ~25 min: LoRA SFT with bundled Mean-SV, α=0.5, layer 15
bash quick_start/code/04_sft_eval.sh       # ~3 min:  GSM8K test accuracy (no SV at test time);
                                           #          without step 03 it evaluates the bundled adapter

# ---- Setting 2: SV-distillation ------------------------------------------
bash quick_start/code/05_distill_build_vector.sh  # ~30 min: student acts on teacher vs. own outputs
bash quick_start/code/06_distill_train.sh         # ~30 min: distil with Mean-SV, α=0.5, layer 15
bash quick_start/code/07_distill_eval.sh          # ~3 min
# On MATH: prefix each step with DATASET=math (evaluated on MATH-500)
```

Times are for one 80 GB GPU. Each script is about five lines. They print the exact
`sv-train` commands, so you can also copy those commands and run them yourself. `METHOD=plain` or
`METHOD=slerp` switches the method (Slerp needs step 02 or 05 first to build per-sample vectors),
for example `METHOD=plain bash quick_start/code/03_sft_train.sh`. `run_sft.sh` and
`run_distill.sh` run a whole setting end to end. See [`quick_start/README.md`](quick_start/README.md) for
each step's inputs, outputs and expected numbers.

## Train with your own steering vector

`sv-train train` injects **any vector file at any layer**:

```bash
# one vector for every training example, added at layer 15
sv-train train --model Qwen/Qwen2.5-1.5B-Instruct --dataset gsm8k \
    --sv my_vector.pt --layer 15 --op add --alpha 0.5 \
    --output_dir outputs/my-run

# one vector per training example ([N, D] tensor, row i = example i), Slerp
sv-train train ... --sv my_per_sample.pt --layer 15 --op slerp --alpha 0.1

# several vectors / layers at once: repeat --inject
sv-train train ... \
    --inject path=a.pt,layer=10,alpha=0.1 \
    --inject path=b.pt,layer=15,alpha=0.5 \
    --inject path=c.pt,layer=20,op=slerp,alpha=0.1,first_k=32
```

Accepted vector files: `.pt` / `.npy` / `.safetensors` with

- a `[D]` tensor: the same vector for every example,
- an `[N, D]` tensor: row `i` is used for training example `i` (rows with norm 0 are skipped),
- `{"direction": tensor}`, or `{"vectors": {layer: tensor}}` (also `"block_15"` keys),
- an acts cache from `sv-train acts`. `--sv_kind mean` gives the dataset mean and `--sv_kind per_sample` keeps one vector per example.

Other injection options: `--normalize` makes vectors unit length, `--first_k K` / `--skip_first_k K` steer only part of the
answer, `--all_tokens` also steers prompt tokens, and `--alpha` can be negative. `--dataset` also accepts a
`.json` / `.jsonl` file of `{"question": ..., "answer": ...}` records, so you can train on your own data. Distillation is
the same command plus `--teacher_json teacher.json`.

The same mechanism is available from Python, with or without the HF `Trainer`:

```python
import torch
from sv_training import Injection, SteeringContext, attach, detach, load_vector

v = load_vector("my_vector.pt")                          # [D] or [N, D]
ctx = SteeringContext()
handles = attach(model, [Injection(v, layer=15, alpha=0.5, op="add")], ctx)

model.train()
with ctx.batch(labels=labels, sample_idx=sample_idx):  # labels == -100 marks prompt tokens
    loss = model(input_ids=input_ids, labels=labels).loss
loss.backward()
detach(handles)
```

`sv_training.trainer.SteeringTrainer` is a drop-in `transformers.Trainer` that does the `ctx.batch(...)`
call for you, and `sv_training.train.train(TrainConfig(...))` is the full LoRA pipeline behind the CLI.

## Full pipeline on a new model or dataset

```bash
M=Qwen/Qwen2.5-7B-Instruct; D=math

# 1. the student's own greedy predictions on the train split
sv-train generate --model $M --dataset $D --max_new_tokens 512 --out preds.json
#    (distillation only) teacher outputs
sv-train generate --model Qwen/Qwen3-8B --dataset $D --enable_thinking 1 \
    --max_new_tokens 4096 --out teacher.json

# 2. answer-token hidden states on reference vs. own predictions (all layers)
sv-train acts --model $M --dataset $D --preds_json preds.json --out acts.pt
#    distillation: add  --reference_json teacher.json

# 3. extract a vector at one layer (or pass acts.pt directly to --sv)
sv-train vector --acts acts.pt --layer 15 --kind mean       --out sv_mean.pt
sv-train vector --acts acts.pt --layer 15 --kind per_sample --out sv_per_sample.pt

# 4. train (paper defaults: LoRA r=8, lr 5e-5, batch 1 x grad-accum 8, 1 epoch)
sv-train train --model $M --dataset $D --sv sv_per_sample.pt --layer 15 \
    --op slerp --alpha 0.1 --output_dir outputs/slerp-7B-$D
#    distillation: add  --teacher_json teacher.json

# 5. evaluate (greedy, vLLM); --alpha > 0 with --sv adds inference-time SV
sv-train eval --model outputs/slerp-7B-$D --dataset $D
```

| Command | What it does | Hardware |
| --- | --- | --- |
| `sv-train generate` | greedy vLLM outputs on a split, with `pred_correct` flags | GPU |
| `sv-train acts` | `gt_acts`, `pred_acts` of shape `[N, L, D]` (answer-token means, all layers) | GPU |
| `sv-train vector` | mean or per-sample vector at one layer from an acts cache | CPU |
| `sv-train train` | LoRA (or `--full_finetune`) SFT / distillation with SV injection | GPU |
| `sv-train eval` | test accuracy; GSM8K, MATH(-500), MBPP, StrategyQA, AIME-24/25 | GPU |
| `sv-train score` | re-score a `predictions.json` | CPU |
| `sv-train inspect` | summarise vector / acts / prediction files | CPU |

Built-in datasets: `gsm8k`, `math` (MATH-500 test split), `mbpp`, `strategyqa`, plus `aime24` / `aime25` for evaluation only.

## Notes

- Layer indices are 0-based and exclude the embedding: layer `l` is the output of transformer block `l`
  (`hidden_states[l + 1]` in HF). Blocks are found automatically for Qwen2/3, Llama, Gemma, Mistral and similar models.
- Steering is applied only in training mode and only to answer tokens (`labels != -100`). In eval mode the hooks do nothing.
- The best layer depends on the model family. In the paper, `l = 15` works for Qwen, while Llama-3.2-3B peaks at layers 10 and 19
  (Section 4.2.3). For a new backbone, run a short layer sweep on held-out training data.
- Code generation (MBPP) is a known failure case: the reference direction conflicts with the direction that makes tests pass.
- The hook implementation is checked against the code used for the paper runs: `tests/test_steering.py`
  asserts bit-identical logits and gradients for every method in fp32 and bf16.

## Repository layout

```text
sv_training/      package: steering hooks, vector I/O, trainer, data, CLI
quick_start/      bundled intermediate results, SFT adapters, paper eval outputs + step-by-step scripts
tests/            CPU unit tests (tiny random models, no downloads)
```

Run `make test` to run the tests and `make check` to also check the CLI and the bundled data.

## Citation

```bibtex
@inproceedings{chen2026steering,
  title     = {Steering Vectors as a Training Signal in {LLM} Post-Training},
  author    = {Chen, Tiejin and Vyas, Maunil and Yao, Huaiyuan and Wei, Hua},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
