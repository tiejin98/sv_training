"""Compute student-space acts cache for distillation SV direction.

For each training example i with teacher output t_i and student bootstrap pred ŷ_i:
  gt_acts[i, ℓ, :]   = mean-pooled student hidden state on prompt + t_i  (answer tokens)
  pred_acts[i, ℓ, :] = mean-pooled student hidden state on prompt + ŷ_i  (answer tokens)
So delta = gt_acts − pred_acts = "from student-pred-direction to teacher-pred-direction"
in student hidden space. Used to train SV-distillation: student learns to imitate teacher
outputs with steering direction = student-space delta.

Usage:
    python main_compute_distill_acts.py \\
        --student Qwen/Qwen2.5-1.5B-Instruct \\
        --teacher_json teacher_gen/Qwen2.5-Math-7B--gsm8k--nothink.json \\
        --student_preds_json acts_cache/1.5B-gsm8k_acts_bootpreds.json \\
        --dataset gsm8k \\
        --out acts_cache/distill_1.5B__Qwen2.5-Math-7B--gsm8k--nothink.pt
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
from typing import Optional
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataset_utils import load_main_split, main_build_messages  # type: ignore
from beegfs_safe_save import save_with_retry  # type: ignore

CACHE_DIR = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))


@torch.no_grad()
def mean_hidden_all_layers(model, tokenizer, prompt_text: str, answer_text: str,
                            max_length: int) -> Optional[torch.Tensor]:
    prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                            add_special_tokens=False).to(model.device)
    prompt_len = prompt_ids["input_ids"].shape[-1]
    full_text = prompt_text + answer_text
    full = tokenizer(full_text, return_tensors="pt", add_special_tokens=False,
                      truncation=True, max_length=max_length).to(model.device)
    seq_len = full["input_ids"].shape[-1]
    if seq_len <= prompt_len:
        return None
    out = model(**full, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states
    num_layers = model.config.num_hidden_layers
    per_layer = []
    for li in range(1, num_layers + 1):
        hs = hidden[li][:, prompt_len:seq_len, :]
        per_layer.append(hs.mean(dim=1).squeeze(0).to(torch.float32).cpu())
    return torch.stack(per_layer, dim=0)  # [L, D]


def build_prompt_text(tok, user_msgs, enable_thinking: bool) -> str:
    kw = {}
    try:
        _ = tok.apply_chat_template(user_msgs, tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=enable_thinking)
        kw = {"enable_thinking": enable_thinking}
    except TypeError:
        pass
    return tok.apply_chat_template(user_msgs, tokenize=False,
                                    add_generation_prompt=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True, help="HF model id for the STUDENT")
    ap.add_argument("--teacher_json", required=True,
                    help="teacher_gen/*.json — list of {'idx','pred',...} or similar")
    ap.add_argument("--student_preds_json", required=True,
                    help="student bootstrap preds JSON (from main_compute_acts)")
    ap.add_argument("--dataset", required=True,
                    choices=["gsm8k", "math", "mbpp", "strategyqa"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--enable_thinking", type=int, default=0,
                    help="student enable_thinking mode (for Qwen3 students)")
    ap.add_argument("--max_context_len", type=int, default=2048)
    args = ap.parse_args()

    print(f"[distill_acts] dataset={args.dataset} student={args.student}")
    examples = load_main_split(args.dataset, "train", cache_dir=CACHE_DIR)
    with open(args.teacher_json) as f:
        teacher_raw = json.load(f)
    with open(args.student_preds_json) as f:
        student_preds = json.load(f)
    assert len(examples) == len(student_preds), \
        f"{len(examples)} vs student={len(student_preds)}"
    # Allow teacher_raw to be either aligned (length == examples) or filtered
    # (shorter list with explicit `idx` field per record).
    if len(teacher_raw) == len(examples) and all(('idx' not in r) or (r['idx'] == i) for i, r in enumerate(teacher_raw)):
        teacher_by_idx = {i: r for i, r in enumerate(teacher_raw)}
    else:
        teacher_by_idx = {int(r['idx']): r for r in teacher_raw}
    print(f"[distill_acts] teacher: {len(teacher_by_idx)} records (filtered={len(teacher_raw) != len(examples)})")

    tok = AutoTokenizer.from_pretrained(args.student, cache_dir=CACHE_DIR,
                                         trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.student, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()
    L = model.config.num_hidden_layers
    D = model.config.hidden_size
    N = len(examples)
    gt_acts = torch.zeros(N, L, D, dtype=torch.float32)
    pred_acts = torch.zeros(N, L, D, dtype=torch.float32)
    gt_valid = 0; pd_valid = 0

    pbar = tqdm(range(N), desc=f"distill-acts {args.dataset}", dynamic_ncols=True)
    for i in pbar:
        ex = examples[i]
        user_msgs = main_build_messages(ex["question"], args.dataset)
        prompt_text = build_prompt_text(tok, user_msgs, bool(args.enable_thinking))
        # GT = teacher's output (skip if no teacher record for this idx, e.g. filtered out)
        t_rec = teacher_by_idx.get(i)
        t_pred = (t_rec.get("pred") or t_rec.get("answer") or "").strip() if t_rec else ""
        if t_pred:
            g = mean_hidden_all_layers(model, tok, prompt_text, t_pred, args.max_context_len)
            if g is not None:
                gt_acts[i] = g; gt_valid += 1
        # Pred = student bootstrap
        s_pred = (student_preds[i].get("pred") or student_preds[i].get("answer") or "").strip()
        if s_pred:
            p = mean_hidden_all_layers(model, tok, prompt_text, s_pred, args.max_context_len)
            if p is not None:
                pred_acts[i] = p; pd_valid += 1
        if (i + 1) % 100 == 0:
            pbar.set_postfix(gt_valid=gt_valid, pd_valid=pd_valid)

    meta = {
        "student": args.student,
        "teacher_json": os.path.abspath(args.teacher_json),
        "student_preds_json": os.path.abspath(args.student_preds_json),
        "dataset": args.dataset,
        "N": N, "num_layers": L, "hidden_size": D,
        "gt_valid": gt_valid, "pred_valid": pd_valid,
        "note": "gt_acts = student hidden on teacher pred; "
                "pred_acts = student hidden on student bootstrap pred; "
                "delta = direction in student space from student→teacher.",
    }
    print(f"\n[distill_acts] gt_valid={gt_valid}/{N}  pd_valid={pd_valid}/{N}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_with_retry({"gt_acts": gt_acts, "pred_acts": pred_acts, "meta": meta},
                    args.out, saver="torch")
    print(f"[distill_acts] saved → {args.out}")


if __name__ == "__main__":
    main()
