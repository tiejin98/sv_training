"""Answer-token-pooled hidden states for building steering vectors (paper Eq. 6).

For every training example i, with a reference completion y_ref (dataset GT
for SFT, a teacher output for distillation) and the student's own bootstrap
prediction y_pred:

    gt_acts[i, l]   = mean_t h_l(x_i, y_ref)[answer tokens]
    pred_acts[i, l] = mean_t h_l(x_i, y_pred)[answer tokens]

The steering vector at layer l is ``gt_acts[:, l] - pred_acts[:, l]``
(per sample) or its mean over samples (Mean-SV). Rows without a reference or
prediction stay zero and are ignored when averaging.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import torch
from tqdm import tqdm

from .data import apply_chat_template, build_messages, load_split
from .trainer import load_teacher


@torch.no_grad()
def mean_hidden_all_layers(model, tokenizer, prompt_text: str, answer_text: str,
                           max_length: int) -> Optional[torch.Tensor]:
    """``[num_layers, hidden]`` mean over answer tokens of ``prompt_text + answer_text``."""
    prompt_len = tokenizer(prompt_text, return_tensors="pt",
                           add_special_tokens=False)["input_ids"].shape[-1]
    full = tokenizer(prompt_text + answer_text, return_tensors="pt", add_special_tokens=False,
                     truncation=True, max_length=max_length).to(model.device)
    seq_len = full["input_ids"].shape[-1]
    if seq_len <= prompt_len:
        return None
    hidden = model(**full, output_hidden_states=True, use_cache=False).hidden_states
    # hidden_states[0] is the embedding output; block l is hidden_states[l + 1].
    return torch.stack([hidden[li][0, prompt_len:seq_len, :].mean(dim=0).float().cpu()
                        for li in range(1, model.config.num_hidden_layers + 1)])


def _pred_text(rec) -> str:
    return ((rec or {}).get("pred") or (rec or {}).get("answer") or "").strip()


def compute_acts(model_path: str, dataset: str, preds_json: str, out: str,
                 reference_json: Optional[str] = None, enable_thinking: bool = False,
                 max_context_len: int = 2048, cache_dir: Optional[str] = None,
                 limit: Optional[int] = None) -> str:
    """Build ``{"gt_acts": [N, L, D], "pred_acts": [N, L, D], "meta": {...}}``.

    ``reference_json=None`` uses dataset GT answers (SV-SFT); a teacher JSON
    gives the distillation direction (student space, student -> teacher).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    examples = load_split(dataset, "train", cache_dir=cache_dir, limit=limit)
    N = len(examples)
    preds = load_teacher(preds_json, N)
    teacher = load_teacher(reference_json, N) if reference_json else None

    tok = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=cache_dir, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    model.eval()
    L, D = model.config.num_hidden_layers, model.config.hidden_size
    gt_acts = torch.zeros(N, L, D, dtype=torch.float32)
    pred_acts = torch.zeros(N, L, D, dtype=torch.float32)
    gt_valid = pred_valid = 0

    pbar = tqdm(range(N), desc=f"acts {os.path.basename(dataset)}", dynamic_ncols=True)
    for i in pbar:
        ex = examples[i]
        prompt_text = apply_chat_template(tok, build_messages(ex["question"], dataset), True,
                                          enable_thinking)
        # Reference: raw answer string appended to the prompt (no closing chat token).
        ref = _pred_text(teacher.get(i)) if teacher is not None else ex["answer"]
        if ref:
            g = mean_hidden_all_layers(model, tok, prompt_text, ref, max_context_len)
            if g is not None:
                gt_acts[i] = g
                gt_valid += 1
        p_text = _pred_text(preds.get(i))
        if p_text:
            p = mean_hidden_all_layers(model, tok, prompt_text, p_text, max_context_len)
            if p is not None:
                pred_acts[i] = p
                pred_valid += 1
        if (i + 1) % 100 == 0:
            pbar.set_postfix(gt_valid=gt_valid, pred_valid=pred_valid)

    meta = {"model": model_path, "dataset": dataset, "N": N, "num_layers": L, "hidden_size": D,
            "reference": reference_json or "dataset GT", "preds_json": preds_json,
            "enable_thinking": enable_thinking, "max_context_len": max_context_len,
            "gt_valid": gt_valid, "pred_valid": pred_valid,
            "note": "layer index l = output of transformer block l (embedding excluded)"}
    print(f"[acts] gt_valid={gt_valid}/{N} pred_valid={pred_valid}/{N}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save({"gt_acts": gt_acts, "pred_acts": pred_acts, "meta": meta}, out)
    print(f"[acts] saved -> {out}")
    return out
