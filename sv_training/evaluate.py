"""Greedy vLLM evaluation of a (LoRA or merged) model, optionally with inference-time SV."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Optional

from .data import is_correct, load_split
from .generate import build_prompts


def merge_adapter(model_path: str, base_model: Optional[str], cache_dir: Optional[str]):
    """If ``model_path`` is a PEFT adapter dir, merge it into its base. Returns (path, tmpdir|None)."""
    cfg_path = os.path.join(model_path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        return model_path, None
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if base_model is None:
        with open(cfg_path) as f:
            base_model = json.load(f)["base_model_name_or_path"]
    print(f"[eval] merging adapter {model_path} onto {base_model}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(base_model, cache_dir=cache_dir,
                                                torch_dtype=torch.bfloat16, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, model_path).merge_and_unload()
    tmp = tempfile.mkdtemp(prefix="sv_eval_merged_")
    merged.save_pretrained(tmp)
    AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir,
                                  trust_remote_code=True).save_pretrained(tmp)
    del base, merged
    return tmp, tmp


def evaluate(model_path: str, dataset: str, out_dir: str, base_model: Optional[str] = None,
             split: str = "test", sv: Optional[str] = None, layer: Optional[int] = None,
             alpha: float = 0.0, sv_kind: str = "mean", max_new_tokens: int = 1024,
             max_model_len: int = 4096, gpu_mem_util: float = 0.85,
             tensor_parallel_size: int = 1, cache_dir: Optional[str] = None,
             limit: Optional[int] = None) -> dict:
    """Writes ``predictions.json`` and ``metrics.json`` to ``out_dir``; returns the metrics.

    With ``sv`` and ``alpha != 0`` the vector is added (``h + alpha * v``) to the
    output of block ``layer`` at every position during decoding. The paper's
    reported numbers use ``alpha = 0`` (no SV at test time).
    """
    import torch
    from transformers import AutoTokenizer

    steer = sv is not None and alpha != 0.0
    v = None
    if steer:
        from .vectors import load_vector
        if layer is None:
            raise ValueError("--layer is required with --sv")
        v = load_vector(sv, layer=layer, kind=sv_kind)
        if v.dim() != 1:
            raise ValueError("inference-time steering needs a single [D] vector")
        # vLLM V1 runs the model in a worker process; shipping the hook there pickles it.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    merged_dir, tmp = merge_adapter(model_path, base_model, cache_dir)
    try:
        examples = load_split(dataset, split, cache_dir=cache_dir, limit=limit)
        tok = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)
        prompts = build_prompts(tok, examples, dataset, enable_thinking=False)

        from vllm import LLM, SamplingParams
        llm = LLM(model=merged_dir, dtype="bfloat16", gpu_memory_utilization=gpu_mem_util,
                  tensor_parallel_size=tensor_parallel_size, trust_remote_code=True,
                  max_model_len=max_model_len, download_dir=cache_dir, enforce_eager=True)

        if steer:
            v_cpu, layer_idx, a = v.to(torch.bfloat16), int(layer), float(alpha)

            def install_hook(model):
                import torch as _torch
                v_gpu = v_cpu.to(next(model.parameters()).device).to(_torch.bfloat16)
                layers = model.model.layers if hasattr(model, "model") else model.layers

                def hook(mod, inp, out):
                    # vLLM decoder layers return (hidden, residual); their sum is the
                    # residual stream, so shifting the first term shifts the stream.
                    if isinstance(out, tuple):
                        return (out[0] + a * v_gpu,) + tuple(out[1:])
                    return out + a * v_gpu

                layers[layer_idx].register_forward_hook(hook)
                return f"hook on layer {layer_idx}"

            print(f"[eval] {llm.llm_engine.apply_model(install_hook)}, alpha={a}, "
                  f"||v||={v.norm().item():.3f}", flush=True)

        outs = llm.generate(prompts, SamplingParams(temperature=0.0, top_p=1.0,
                                                    max_tokens=max_new_tokens))
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    results, correct, total_tokens = [], 0, 0
    for ex, out in zip(examples, outs):
        text = out.outputs[0].text
        ok = is_correct(text, ex, dataset)
        correct += int(ok)
        total_tokens += len(out.outputs[0].token_ids)
        results.append({"question": ex["question"], "gt": ex.get("answer", ""), "pred": text,
                        "correct": ok, "n_tokens": len(out.outputs[0].token_ids)})
    n = len(examples)
    metrics = {"dataset": dataset, "split": split, "n": n, "correct": correct,
               "accuracy": correct / max(n, 1), "avg_tokens": total_tokens / max(n, 1),
               "model_path": model_path, "sv": sv, "layer": layer, "alpha": alpha}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] {dataset}/{split}: acc={100 * metrics['accuracy']:.2f}% ({correct}/{n}) "
          f"avg_tokens={metrics['avg_tokens']:.1f} -> {out_dir}")
    return metrics


def score_predictions(predictions_json: str, dataset: str, split: str = "test",
                      cache_dir: Optional[str] = None) -> dict:
    """Re-score a ``predictions.json`` against the dataset (CPU only, no model)."""
    with open(predictions_json, encoding="utf-8") as f:
        preds = json.load(f)
    examples = load_split(dataset, split, cache_dir=cache_dir)
    if len(preds) > len(examples):
        raise ValueError(f"{len(preds)} predictions but only {len(examples)} examples")
    correct = sum(int(is_correct(p["pred"], ex, dataset)) for p, ex in zip(preds, examples))
    n = len(preds)
    print(f"[score] {dataset}/{split}: acc={100 * correct / max(n, 1):.2f}% ({correct}/{n})")
    return {"n": n, "correct": correct, "accuracy": correct / max(n, 1)}
