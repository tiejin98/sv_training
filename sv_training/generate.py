"""Greedy vLLM generation on a dataset split (teacher outputs or student bootstrap predictions)."""
from __future__ import annotations

import gc
import json
import os
from typing import List, Optional

from .data import apply_chat_template, build_messages, is_builtin, is_correct, load_split


def build_prompts(tokenizer, examples, dataset: str, enable_thinking: bool) -> List[str]:
    return [apply_chat_template(tokenizer, build_messages(ex["question"], dataset), True,
                                enable_thinking) for ex in examples]


def generate(model: str, dataset: str, out: str, split: str = "train",
             max_new_tokens: int = 1024, enable_thinking: bool = False,
             temperature: float = 0.0, top_p: float = 1.0, gpu_mem_util: float = 0.85,
             tensor_parallel_size: int = 1, max_model_len: Optional[int] = None,
             cache_dir: Optional[str] = None, limit: Optional[int] = None) -> str:
    """Write ``[{"idx", "question", "gt_answer", "pred", "pred_correct", "n_tokens"}, ...]``."""
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    examples = load_split(dataset, split, cache_dir=cache_dir, limit=limit)
    tok = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir, trust_remote_code=True)
    prompts = build_prompts(tok, examples, dataset, enable_thinking)

    mml = max_model_len or max(2048, max_new_tokens + 2048)
    print(f"[generate] {model} on {dataset}/{split}: {len(prompts)} prompts, max_model_len={mml}")
    llm = LLM(model=model, download_dir=cache_dir, dtype="bfloat16",
              gpu_memory_utilization=gpu_mem_util, trust_remote_code=True,
              max_model_len=mml, tensor_parallel_size=tensor_parallel_size)
    outs = llm.generate(prompts, SamplingParams(temperature=temperature, top_p=top_p,
                                                max_tokens=max_new_tokens))

    records, n_correct = [], 0
    for i, (ex, req) in enumerate(zip(examples, outs)):
        text = req.outputs[0].text if req.outputs else ""
        ok = False
        if text.strip() and is_builtin(dataset):
            try:
                ok = is_correct(text, ex, dataset)
            except Exception:
                ok = False
        n_correct += int(ok)
        records.append({"idx": i, "question": ex["question"], "gt_answer": ex["answer"],
                        "pred": text, "pred_correct": bool(ok),
                        "n_tokens": len(req.outputs[0].token_ids) if req.outputs else 0})
    print(f"[generate] correct={n_correct}/{len(records)} "
          f"({100.0 * n_correct / max(len(records), 1):.1f}%)")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"[generate] saved -> {out}")
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
