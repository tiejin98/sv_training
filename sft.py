#!/usr/bin/env python3
"""Generic SFT trainer supporting Plain SFT and per-instance-Slerp SV-SFT.

Works across 4 datasets (gsm8k / math / mbpp / strategyqa) and any HF Qwen
model (1.5B, 7B, Qwen3-4B, etc.).

SV-SFT requires an acts cache file produced by `main_compute_acts.py`
(`gsm8k_train_acts_cache_<model>_<dataset>.pt`).

Usage:
    # Plain
    python main_sft.py --dataset gsm8k --model_path Qwen/Qwen2.5-1.5B-Instruct \
                       --method plain --output_dir outputs/main/plain_1.5B_gsm8k

    # SV (per-instance Slerp)
    python main_sft.py --dataset gsm8k --model_path Qwen/Qwen2.5-1.5B-Instruct \
                       --method slerp --acts_cache /scratch/.../acts_cache.pt \
                       --alpha 0.1 --layer 15 --output_dir outputs/main/sv_1.5B_gsm8k
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

from dataset_utils import load_main_split, main_build_messages

# Reuse slerp_rotate + SteerAnswerOnlyTrainer from existing infrastructure
from _steering_helpers import slerp_rotate  # type: ignore
from _steering_helpers import (  # type: ignore
    SteerAnswerOnlyTrainer, unwrap_ddp,
)
from beegfs_safe_save import torch_load_with_retry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))


# ---------------------------------------------------------------------------
# Dataset record building
# ---------------------------------------------------------------------------


def build_training_records(tokenizer, dataset_name: str, split: str = "train",
                           cache_dir: str = CACHE_DIR,
                           max_context_len: int = 2048,
                           enable_thinking: bool = False,
                           distill_teacher_json: Optional[str] = None,
                           distill_keep_only_correct: bool = True,
                           append_student_eos: bool = False) -> List[Dict]:
    """Build SFT records: prompt + full answer + <|im_end|>\\n.

    If `distill_teacher_json` is given, the assistant content is the teacher's
    `pred` instead of `ex["answer"]` (hard distillation). When
    `distill_keep_only_correct=True`, records where the teacher's pred did NOT
    satisfy correctness (pred_correct=False) are skipped.

    `distill_teacher_json` may be either:
      - aligned: list with same length as examples (one record per dataset idx)
      - filtered: shorter list, each record carries an `idx` field; missing idx
        are skipped (used for Gemma re-gen + filter pipeline).

    If `append_student_eos=True`, the student's eos_token is appended to the
    teacher response before chat-template assembly so the student learns to
    terminate at the answer.

    Returns list of records with keys: context, labels, orig_sample_idx.
    """
    examples = load_main_split(dataset_name, split, cache_dir=cache_dir)
    teacher_by_idx = None
    if distill_teacher_json is not None:
        import json as _json
        with open(distill_teacher_json) as _f:
            teacher_raw = _json.load(_f)
        if len(teacher_raw) == len(examples) and all('idx' not in r or r['idx'] == i for i, r in enumerate(teacher_raw)):
            teacher_by_idx = {i: r for i, r in enumerate(teacher_raw)}
        else:
            teacher_by_idx = {int(r['idx']): r for r in teacher_raw}
        print(f"  distill teacher: loaded {len(teacher_by_idx)} records (filtered={len(teacher_raw) != len(examples)})")
    records = []
    skipped_long = 0
    skipped_wrong_teacher = 0
    skipped_no_teacher = 0
    eos_str = tokenizer.eos_token if (append_student_eos and tokenizer.eos_token) else ""
    for i, ex in enumerate(examples):
        q = ex["question"]
        if teacher_by_idx is not None:
            t = teacher_by_idx.get(i)
            if t is None:
                skipped_no_teacher += 1
                continue
            if distill_keep_only_correct and not t.get("pred_correct", False):
                skipped_wrong_teacher += 1
                continue
            a = (t.get("pred") or t.get("answer") or "").strip()
            if not a:
                continue
        else:
            a = ex["answer"]
        if eos_str:
            a = a.rstrip() + eos_str
        # Build full chat including assistant turn
        user_msgs = main_build_messages(q, dataset_name)
        # Tokenize prompt-only and full-with-answer
        kw_thinking = {}
        try:
            _ = tokenizer.apply_chat_template(user_msgs, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=enable_thinking)
            kw_thinking = {"enable_thinking": enable_thinking}
        except TypeError:
            kw_thinking = {}
        prompt_text = tokenizer.apply_chat_template(
            user_msgs, tokenize=False, add_generation_prompt=True, **kw_thinking
        )
        full_msgs = user_msgs + [{"role": "assistant", "content": a}]
        full_text = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False, **kw_thinking
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        if len(full_ids) <= len(prompt_ids):
            continue
        if len(full_ids) > max_context_len:
            skipped_long += 1; continue
        answer_ids = full_ids[len(prompt_ids):]
        context = full_ids
        labels = [-100] * len(prompt_ids) + list(answer_ids)
        records.append({
            "context": context,
            "labels": labels,
            "orig_sample_idx": i,
        })
    if skipped_long:
        print(f"  skipped {skipped_long} records exceeding max_context_len={max_context_len}")
    if teacher_by_idx is not None:
        print(f"  distill: kept {len(records)} records (skipped {skipped_wrong_teacher} wrong-teacher, {skipped_no_teacher} no-teacher-record)")
    return records


# ---------------------------------------------------------------------------
# Dataset + collator
# ---------------------------------------------------------------------------


class SFTDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, idx): return self.records[idx]


class SFTCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, batch):
        # batch=1 only
        assert len(batch) == 1
        rec = batch[0]
        input_ids = torch.tensor([rec["context"]], dtype=torch.long)
        labels = torch.tensor([rec["labels"]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "sample_idx": torch.tensor([rec["orig_sample_idx"]], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# SV hook — per-instance Slerp on answer tokens (labels != -100)
# ---------------------------------------------------------------------------


class SVState:
    def __init__(self):
        self.sample_idx = -1
        self.sv_skip_first_k_answer = 0  # Phase 7: skip first K answer tokens from SV hook


def make_per_inst_slerp_hook(deltas: torch.Tensor, state: SVState, alpha: float):
    """Per-instance Slerp on answer tokens. deltas: [N, D] unit-normalised."""
    def hook(module, inputs, output):
        if not module.training:
            return output
        h = output[0] if isinstance(output, tuple) else output  # [B, T, D]
        assert h.shape[0] == 1
        idx = state.sample_idx
        if idx < 0 or idx >= deltas.shape[0]:
            return output
        # labels mask: set on the wrapped model by PerSampleTrainer before forward
        base_ref = unwrap_ddp(module) if hasattr(module, 'module') else None
        labels = getattr(module, "_sv_labels", None)
        if labels is None:
            # look up via shared closure attribute set by Trainer
            labels = state.current_labels  # set by trainer
        if labels is None:
            return output
        mask = (labels[0] != -100)  # [T]
        if not mask.any():
            return output
        v = deltas[idx].to(h.dtype).to(h.device)  # [D]
        # Skip slerp for zero-norm deltas (e.g. correct-only filter zeros out
        # rows where bootstrap pred was already correct). slerp_rotate with
        # v=0 gives cos_θ=0 → scales h by sin((1-α)π/2) ≈ 0.988 instead of
        # leaving it unchanged; bypass entirely.
        if v.norm() < 1e-6:
            return output
        T = h.shape[1]
        idx_positions = mask.nonzero(as_tuple=True)[0]
        # Phase 7: optionally skip first-K answer tokens (code-aware mask: drop def line)
        sk = getattr(state, "sv_skip_first_k_answer", 0)
        if sk > 0 and idx_positions.numel() > sk:
            idx_positions = idx_positions[sk:]  # answer-token positions
        if idx_positions.numel() == 0:
            return output
        h_new = h.clone()
        h_slice = h_new[0, idx_positions, :]          # [K, D]
        h_slice = slerp_rotate(h_slice, v, float(alpha))
        h_new[0, idx_positions, :] = h_slice
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        return h_new
    return hook


def make_mean_additive_hook(mean_dir: torch.Tensor, state: SVState, alpha: float):
    """Mean-SV additive on answer tokens: h ← h + α·v_mean.

    mean_dir: [D] — IMPORTANT: passed AS-IS (no re-normalisation inside hook).
    v2 calibration used RAW (unnormalised) mean vector with α=0.5; caller must
    pass the raw delta-mean (shape [D]) to match v2 algorithmically. Applying
    to answer tokens only via labels != -100 mask. Not per-sample — single
    global direction for all training samples.
    """
    v_mean = mean_dir.to(torch.float32)

    def hook(module, inputs, output):
        if not module.training:
            return output
        h = output[0] if isinstance(output, tuple) else output  # [B, T, D]
        labels = state.current_labels
        if labels is None:
            return output
        mask = (labels[0] != -100)
        if not mask.any():
            return output
        v = v_mean.to(h.dtype).to(h.device)  # [D] — raw, not unit-normalised
        idx_positions = mask.nonzero(as_tuple=True)[0]
        # Phase 7: optionally skip first-K answer tokens (code-aware mask: drop def line)
        sk = getattr(state, "sv_skip_first_k_answer", 0)
        if sk > 0 and idx_positions.numel() > sk:
            idx_positions = idx_positions[sk:]
        if idx_positions.numel() == 0:
            return output
        h_new = h.clone()
        h_new[0, idx_positions, :] = h_new[0, idx_positions, :] + alpha * v
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        return h_new
    return hook


def make_per_inst_slerp_first_k_hook(deltas: torch.Tensor, state: SVState,
                                      alpha: float, k: int):
    """Per-instance Slerp on FIRST K answer tokens (front-loaded scope)."""
    def hook(module, inputs, output):
        if not module.training:
            return output
        h = output[0] if isinstance(output, tuple) else output
        assert h.shape[0] == 1
        idx = state.sample_idx
        if idx < 0 or idx >= deltas.shape[0]:
            return output
        labels = state.current_labels
        if labels is None:
            return output
        mask = (labels[0] != -100)
        if not mask.any():
            return output
        v = deltas[idx].to(h.dtype).to(h.device)
        if v.norm() < 1e-6:
            return output
        idx_positions = mask.nonzero(as_tuple=True)[0]
        # Phase 7: optionally skip first-K answer tokens (code-aware mask: drop def line)
        sk = getattr(state, "sv_skip_first_k_answer", 0)
        if sk > 0 and idx_positions.numel() > sk:
            idx_positions = idx_positions[sk:]   # all answer positions
        if idx_positions.numel() == 0:
            return output
        # Keep only the FIRST k of those
        idx_positions_k = idx_positions[:k]
        h_new = h.clone()
        h_slice = h_new[0, idx_positions_k, :]
        h_slice = slerp_rotate(h_slice, v, float(alpha))
        h_new[0, idx_positions_k, :] = h_slice
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        return h_new
    return hook


def make_per_inst_additive_hook(deltas: torch.Tensor, state: SVState, alpha: float):
    """Per-instance additive: h ← h + α·v_i  where v_i = delta_i / ‖delta_i‖."""
    def hook(module, inputs, output):
        if not module.training:
            return output
        h = output[0] if isinstance(output, tuple) else output
        assert h.shape[0] == 1
        idx = state.sample_idx
        if idx < 0 or idx >= deltas.shape[0]:
            return output
        labels = state.current_labels
        if labels is None:
            return output
        mask = (labels[0] != -100)
        if not mask.any():
            return output
        v = deltas[idx].to(h.dtype).to(h.device)
        if v.norm() < 1e-6:
            return output
        idx_positions = mask.nonzero(as_tuple=True)[0]
        # Phase 7: optionally skip first-K answer tokens (code-aware mask: drop def line)
        sk = getattr(state, "sv_skip_first_k_answer", 0)
        if sk > 0 and idx_positions.numel() > sk:
            idx_positions = idx_positions[sk:]
        if idx_positions.numel() == 0:
            return output
        h_new = h.clone()
        h_new[0, idx_positions, :] = h_new[0, idx_positions, :] + alpha * v
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        return h_new
    return hook


def make_random_gaussian_per_sample_hook(n_samples: int, hidden_dim: int,
                                          state: SVState, alpha: float,
                                          seed: int = 0):
    """Control: per-sample random Gaussian direction (unit-norm). Drawn once at init,
    kept fixed per sample_idx. Use to verify SV gain isn't just "any perturbation helps"."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n_samples, hidden_dim, generator=g)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def hook(module, inputs, output):
        if not module.training:
            return output
        h = output[0] if isinstance(output, tuple) else output
        assert h.shape[0] == 1
        idx = state.sample_idx
        if idx < 0 or idx >= dirs.shape[0]:
            return output
        labels = state.current_labels
        if labels is None:
            return output
        mask = (labels[0] != -100)
        if not mask.any():
            return output
        v = dirs[idx].to(h.dtype).to(h.device)
        idx_positions = mask.nonzero(as_tuple=True)[0]
        # Phase 7: optionally skip first-K answer tokens (code-aware mask: drop def line)
        sk = getattr(state, "sv_skip_first_k_answer", 0)
        if sk > 0 and idx_positions.numel() > sk:
            idx_positions = idx_positions[sk:]
        if idx_positions.numel() == 0:
            return output
        h_new = h.clone()
        h_slice = h_new[0, idx_positions, :]
        h_slice = slerp_rotate(h_slice, v, float(alpha))
        h_new[0, idx_positions, :] = h_slice
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        return h_new
    return hook


class PerSampleSFTTrainer(SteerAnswerOnlyTrainer):
    def __init__(self, *args, sv_state: SVState = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sv_state = sv_state
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_idx = inputs.pop("sample_idx", None)
        if self.sv_state is not None:
            self.sv_state.sample_idx = int(sample_idx[0].item()) if sample_idx is not None else -1
            self.sv_state.current_labels = inputs.get("labels", None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def resolve_mid_layer(model) -> int:
    """Match v2 recipe: DEFAULT_BLOCK_IDX = 15 for Qwen2.5 28-layer models
    (just past the middle). Scale proportionally for other depths:
    layer = num_layers // 2 + 1  →  15 for 28, 19 for 36, 17 for 32, …
    """
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Cannot determine num_hidden_layers from model config")
    return num_layers // 2 + 1


def get_target_layer_module(model, layer_idx: int):
    try:
        return model.base_model.model.model.layers[layer_idx]  # PEFT-wrapped
    except AttributeError:
        return model.model.layers[layer_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math", "mbpp", "strategyqa"])
    ap.add_argument("--model_path", required=True, help="HF name or local dir")
    ap.add_argument("--method", required=True,
                    choices=["plain", "slerp", "mean_sv_add", "per_inst_add",
                             "random_gauss", "sv_first_k", "sv_multi_layer"])
    ap.add_argument("--k", type=int, default=32,
                    help="(method=slerp_first_k) # of leading answer tokens to steer")
    ap.add_argument("--multi_layers", default="10,15,20",
                    help="(method=slerp_multi_layer) comma-sep layers for per-inst Slerp")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--cache_dir", default=CACHE_DIR)

    # SFT hyperparams
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_context_len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enable_thinking", type=int, default=0)

    # SV hyperparams
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--layer", type=int, default=-1, help="L/2 if -1 (auto-resolve)")
    ap.add_argument("--acts_cache", default=None,
                    help="path to .pt file with {'gt_acts': [N, L, D], 'pred_acts': [N, L, D]} "
                         "or {'delta': [N, D]} at chosen layer. Required for method=slerp.")
    ap.add_argument("--distill_teacher_json", default=None,
                    help="If given, use teacher_json's pred text as assistant content "
                         "instead of ex['answer']. Hard distillation.")
    ap.add_argument("--distill_keep_only_correct", type=int, default=1,
                    help="(distill) keep only teacher records with pred_correct=True")
    ap.add_argument("--append_student_eos", type=int, default=0,
                    help="If 1, append tokenizer.eos_token to teacher pred / GT answer "
                         "before chat-template assembly (forces student to learn termination).")
    ap.add_argument("--gradient_checkpointing", type=int, default=0,
                    help="Trade compute for memory; required when training with "
                         "long contexts (e.g. max_context_len=8000) on 3B+ models.")
    ap.add_argument("--mean_dir_override", default=None,
                    help="(method=mean_sv_add only) path to .pt file with {'direction': [D]} "
                         "to use instead of averaging acts_cache deltas. For filtered-mean "
                         "variants (calc-wrong-only, cos-weighted).")
    ap.add_argument("--sv_skip_first_k_answer", type=int, default=0,
                    help="(Phase 7 / code-aware mask) skip first K answer tokens from SV hook "
                         "while keeping them in loss. Approximates skipping def signature on MBPP.")

    # W&B
    ap.add_argument("--wandb_project", default="steering_vector")
    ap.add_argument("--wandb_run_name", default=None)
    args = ap.parse_args()

    os.environ["WANDB_PROJECT"] = args.wandb_project

    # ---- Load model + LoRA ----
    print(f"Loading tokenizer + model from {args.model_path}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.cache_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    num_layers = model.config.num_hidden_layers
    # Match v2 recipe: DEFAULT_BLOCK_IDX = 15 for 28-layer Qwen2.5.
    # Generalise: num_layers // 2 + 1  →  15 for 28, 19 for 36.
    layer_idx = args.layer if args.layer >= 0 else num_layers // 2 + 1
    hidden_dim = model.config.hidden_size
    print(f"  num_hidden_layers={num_layers}, hidden_dim={hidden_dim}, using layer={layer_idx}")

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---- Build training records ----
    print(f"Building training records for {args.dataset}...", flush=True)
    records = build_training_records(
        tok, args.dataset, split="train", cache_dir=args.cache_dir,
        max_context_len=args.max_context_len,
        enable_thinking=bool(args.enable_thinking),
        distill_teacher_json=args.distill_teacher_json,
        distill_keep_only_correct=bool(args.distill_keep_only_correct),
        append_student_eos=bool(args.append_student_eos),
    )
    print(f"  built {len(records)} records")

    # ---- Load acts cache if method uses SV direction ----
    sv_state = None
    hook_handle = None

    def _load_deltas_raw(return_valid: bool = False):
        """Load per-instance deltas at chosen layer, RAW (unnormalised).

        If `return_valid=True`, also returns a boolean mask [N] flagging rows
        where BOTH gt_acts and pred_acts are non-zero (i.e. valid teacher and
        student preds existed when the cache was built). This is needed for
        filtered/curated caches where dropped samples leave gt_acts as zeros
        — including those rows in `delta = gt - pred = -pred` would contaminate
        the mean direction (mean_sv_add) and produce spurious unit deltas (sv).
        """
        assert args.acts_cache, f"--acts_cache required for method={args.method}"
        print(f"Loading acts cache from {args.acts_cache}...")
        cache = torch_load_with_retry(args.acts_cache)
        valid = None
        if "delta" in cache:
            d = cache["delta"]
            if return_valid:
                # No gt/pred breakdown — fall back to delta-norm > 0 (cannot
                # distinguish dropped vs zero-delta rows in this format).
                valid = d.norm(dim=-1) > 1e-6
        elif "gt_acts" in cache and "pred_acts" in cache:
            gt = cache["gt_acts"]
            pd = cache["pred_acts"]
            if gt.dim() == 3:
                gt = gt[:, layer_idx, :]
                pd = pd[:, layer_idx, :]
            d = gt - pd
            if return_valid:
                gt_norm = gt.norm(dim=-1)
                pd_norm = pd.norm(dim=-1)
                valid = (gt_norm > 1e-6) & (pd_norm > 1e-6)
        else:
            raise ValueError(f"acts_cache missing keys; got {list(cache.keys())}")
        if return_valid:
            return d.to(torch.float32), valid
        return d.to(torch.float32)

    if args.method == "slerp":
        deltas = _load_deltas_raw()
        # Unit-normalise per-instance (v2 per-inst Slerp semantics)
        norms = deltas.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        deltas = deltas / norms
        print(f"  deltas shape={tuple(deltas.shape)} (per-inst unit)")
        max_idx = max(r["orig_sample_idx"] for r in records)
        if max_idx >= deltas.shape[0]:
            raise RuntimeError(f"acts cache has {deltas.shape[0]} rows but dataset idx up to {max_idx}")
        target_block = get_target_layer_module(model, layer_idx)
        sv_state = SVState()
        sv_state.sv_skip_first_k_answer = args.sv_skip_first_k_answer
        sv_state.current_labels = None
        hook_handle = target_block.register_forward_hook(
            make_per_inst_slerp_hook(deltas, sv_state, args.alpha)
        )
    elif args.method == "mean_sv_add":
        if args.mean_dir_override:
            print(f"  loading override mean direction: {args.mean_dir_override}")
            obj = torch_load_with_retry(args.mean_dir_override)
            mean_dir_raw = obj["direction"].to(torch.float32)
        else:
            deltas_raw, valid = _load_deltas_raw(return_valid=True)
            n_total = deltas_raw.shape[0]
            n_valid = int(valid.sum().item()) if valid is not None else n_total
            if valid is not None and n_valid < n_total:
                print(f"  mean_sv_add: masking out {n_total - n_valid} dropped rows (gt or pred zero); using {n_valid}/{n_total}")
                mean_dir_raw = deltas_raw[valid].mean(dim=0)
            else:
                mean_dir_raw = deltas_raw.mean(dim=0)
        print(f"  mean_dir_raw shape={tuple(mean_dir_raw.shape)} ‖v‖={mean_dir_raw.norm().item():.4f}")
        target_block = get_target_layer_module(model, layer_idx)
        sv_state = SVState()
        sv_state.sv_skip_first_k_answer = args.sv_skip_first_k_answer
        sv_state.current_labels = None
        hook_handle = target_block.register_forward_hook(
            make_mean_additive_hook(mean_dir_raw, sv_state, args.alpha)
        )
    elif args.method == "per_inst_add":
        deltas = _load_deltas_raw()
        # Unit-normalise per-instance, apply additively (not Slerp)
        norms = deltas.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        deltas = deltas / norms
        print(f"  deltas shape={tuple(deltas.shape)} (per-inst unit, additive)")
        max_idx = max(r["orig_sample_idx"] for r in records)
        if max_idx >= deltas.shape[0]:
            raise RuntimeError(f"acts cache rows={deltas.shape[0]} vs idx up to {max_idx}")
        target_block = get_target_layer_module(model, layer_idx)
        sv_state = SVState()
        sv_state.sv_skip_first_k_answer = args.sv_skip_first_k_answer
        sv_state.current_labels = None
        hook_handle = target_block.register_forward_hook(
            make_per_inst_additive_hook(deltas, sv_state, args.alpha)
        )
    elif args.method == "sv_first_k":
        deltas = _load_deltas_raw()
        norms = deltas.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        deltas = deltas / norms
        print(f"  deltas shape={tuple(deltas.shape)} (per-inst unit, first-K slerp K={args.k})")
        max_idx = max(r["orig_sample_idx"] for r in records)
        if max_idx >= deltas.shape[0]:
            raise RuntimeError(f"acts cache rows={deltas.shape[0]} vs idx up to {max_idx}")
        target_block = get_target_layer_module(model, layer_idx)
        sv_state = SVState()
        sv_state.sv_skip_first_k_answer = args.sv_skip_first_k_answer
        sv_state.current_labels = None
        hook_handle = target_block.register_forward_hook(
            make_per_inst_slerp_first_k_hook(deltas, sv_state, args.alpha, args.k)
        )
    elif args.method == "sv_multi_layer":
        # Per-instance Slerp at multiple layers simultaneously. Each layer gets a
        # fresh hook reading its own slice of the [N, L, D] acts cache.
        assert args.acts_cache, "--acts_cache required"
        layers_list = [int(x) for x in args.multi_layers.split(",")]
        print(f"  multi-layer: applying per-inst Slerp α={args.alpha} at layers {layers_list}")
        print(f"Loading acts cache from {args.acts_cache}...")
        cache = torch_load_with_retry(args.acts_cache)
        gt3 = cache["gt_acts"]   # [N, L, D]
        pd3 = cache["pred_acts"]
        sv_state = SVState()
        sv_state.sv_skip_first_k_answer = args.sv_skip_first_k_answer
        sv_state.current_labels = None
        hook_handle = []  # list of handles
        for li in layers_list:
            d = (gt3[:, li, :] - pd3[:, li, :]).to(torch.float32)
            d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            target = get_target_layer_module(model, li)
            hh = target.register_forward_hook(
                make_per_inst_slerp_hook(d, sv_state, args.alpha)
            )
            hook_handle.append(hh)
    elif args.method == "random_gauss":
        # Per-sample random Gaussian unit direction (control).
        n_samples_total = max(r["orig_sample_idx"] for r in records) + 1
        print(f"  random-gauss: n_samples={n_samples_total} hidden_dim={hidden_dim}")
        target_block = get_target_layer_module(model, layer_idx)
        sv_state = SVState()
        sv_state.sv_skip_first_k_answer = args.sv_skip_first_k_answer
        sv_state.current_labels = None
        hook_handle = target_block.register_forward_hook(
            make_random_gaussian_per_sample_hook(
                n_samples_total, hidden_dim, sv_state, args.alpha, seed=args.seed)
        )

    # ---- Train (save to /tmp then cp to beegfs) ----
    tmp_train_dir = tempfile.mkdtemp(prefix="mainsft_", dir="/tmp")
    run_name = args.wandb_run_name or f"mainsft-{args.method}-{args.dataset}-{Path(args.model_path).name}"
    training_args = TrainingArguments(
        output_dir=tmp_train_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=20,
        save_strategy="no",
        bf16=True,
        report_to=["wandb"],
        run_name=run_name,
        seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=bool(getattr(args, "gradient_checkpointing", 0)),
    )
    if bool(getattr(args, "gradient_checkpointing", 0)):
        # Required for grad checkpointing + LoRA with PEFT-wrapped model
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # Methods needing per-sample sample_idx routing: sv / per_inst_add / random_gauss /
    # sv_first_k / sv_multi_layer. mean_sv_add uses a global direction — still needs
    # labels but no per-sample idx; use PerSampleSFTTrainer anyway to populate
    # state.current_labels.
    _sv_like = {"slerp", "mean_sv_add", "per_inst_add", "random_gauss",
                "sv_first_k", "sv_multi_layer"}
    TrainerCls = PerSampleSFTTrainer if args.method in _sv_like else SteerAnswerOnlyTrainer
    kwargs = dict(
        model=model, args=training_args,
        train_dataset=SFTDataset(records),
        tokenizer=tok,
        data_collator=SFTCollator(tok),
    )
    if args.method in _sv_like:
        kwargs["sv_state"] = sv_state
    trainer = TrainerCls(**kwargs)
    trainer.train()

    # Save adapter to /tmp, cp to beegfs target
    final_tmp = tempfile.mkdtemp(prefix="mainsft_adp_", dir="/tmp")
    trainer.save_model(final_tmp)
    os.makedirs(args.output_dir, exist_ok=True)
    for attempt in range(5):
        try:
            subprocess.run(["cp", "-a", f"{final_tmp}/.", args.output_dir], check=True)
            print(f"Saved adapter to {args.output_dir}", flush=True); break
        except Exception as e:
            print(f"cp attempt {attempt+1} failed: {e}", flush=True)
            import time; time.sleep(5)

    if hook_handle is not None:
        if isinstance(hook_handle, list):
            for hh in hook_handle:
                hh.remove()
        else:
            hook_handle.remove()
    shutil.rmtree(tmp_train_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
