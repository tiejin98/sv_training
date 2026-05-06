#!/usr/bin/env python3
"""Phase 4.7 (vLLM-based) — inference-time SV via vLLM forward hook.

Same logic as main_infer_sv.py but uses vLLM for fast generation. Patches the
target transformer layer with a forward hook that adds α*v_mean to its output.

Usage:
    python main_infer_sv_vllm.py \
        --model_path outputs/main/D-7B__Qwen3-8B--math--think--sv \
        --base_model_name Qwen/Qwen2.5-7B-Instruct \
        --acts_cache acts_cache/distill_7B__Qwen3-8B--math--think.pt \
        --dataset math --layer 15 --alpha 0.1 \
        --out_name P47vllm-7B-math-a0.1
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, tempfile
from pathlib import Path

from dataset_utils import (
    load_main_split, main_build_messages, main_is_correct,
)
from beegfs_safe_save import save_with_retry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))


def maybe_merge(model_path: str, base_model_name: str, cache_dir: str) -> str:
    if not os.path.isdir(model_path):
        return model_path
    if not os.path.exists(os.path.join(model_path, "adapter_config.json")):
        return model_path
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    print(f"Merging PEFT adapter {model_path} onto {base_model_name}...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    m = PeftModel.from_pretrained(base, model_path)
    merged = m.merge_and_unload()
    tok = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir, trust_remote_code=True)
    d = tempfile.mkdtemp(prefix="isv_v_merged_", dir="/tmp")
    merged.save_pretrained(d); tok.save_pretrained(d)
    del base, m, merged; torch.cuda.empty_cache()
    return d


def compute_v_mean(acts_cache_path: str, layer: int, dtype):
    import torch
    d = torch.load(acts_cache_path, weights_only=False, map_location="cpu")
    delta = d["gt_acts"] - d["pred_acts"]  # (N, L, H)
    v_mean = delta.mean(dim=0)[layer]  # (H,)
    norm = v_mean.norm().item()
    print(f"[infer_sv_vllm] acts={acts_cache_path} layer={layer} ||v||={norm:.4f}")
    return v_mean.to(dtype=dtype), norm


def find_decoder_layers(model):
    """Return list of transformer block modules in vLLM's model."""
    # vLLM model.model = inner Qwen2/Qwen3 model with .layers list
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError("Could not find decoder layers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--base_model_name", default=None)
    ap.add_argument("--acts_cache", required=True)
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math", "mbpp", "strategyqa"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache_dir", default=CACHE_DIR)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--out_name", required=True)
    ap.add_argument("--results_md", default=str(PROJECT_ROOT / "main_sft_and_distillation.md"))
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    base_name = args.base_model_name
    if base_name is None and os.path.isdir(args.model_path):
        cfg = os.path.join(args.model_path, "adapter_config.json")
        if os.path.exists(cfg):
            with open(cfg) as f:
                base_name = json.load(f).get("base_model_name_or_path")
    if base_name is None:
        base_name = args.model_path
    merged_dir = maybe_merge(args.model_path, base_name, args.cache_dir)

    # Load tokenizer (for prompt construction)
    tok = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)
    try:
        kw = {"enable_thinking": False}
        _ = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True, **kw)
        use_thinking_kw = True
    except TypeError:
        use_thinking_kw = False

    # Build prompts
    examples = load_main_split(args.dataset, args.split, cache_dir=args.cache_dir, limit=args.limit)
    prompts = []
    for ex in examples:
        msgs = main_build_messages(ex["question"], args.dataset)
        kw_ = {"enable_thinking": False} if use_thinking_kw else {}
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw_))
    print(f"[infer_sv_vllm] N={len(examples)} prompts ready", flush=True)

    # Load vLLM
    from vllm import LLM, SamplingParams
    print(f"[infer_sv_vllm] Loading vLLM from {merged_dir}...", flush=True)
    llm = LLM(model=merged_dir, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util, tensor_parallel_size=1,
              trust_remote_code=True, max_model_len=args.max_model_len,
              download_dir=args.cache_dir, enforce_eager=True)

    # vLLM 0.11 v1 engine: use apply_model to register hook in worker process
    v_mean_cpu, vnorm = compute_v_mean(args.acts_cache, args.layer, dtype=torch.bfloat16)
    layer_idx = int(args.layer)
    alpha = float(args.alpha)

    def install_hook(model):
        """Run inside worker process. Receives the worker's model object."""
        import torch as _torch
        v_gpu = v_mean_cpu.to(next(model.parameters()).device).to(_torch.bfloat16)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            raise RuntimeError("Could not find decoder layers")
        target = layers[layer_idx]

        def hook(mod, inp, out):
            if isinstance(out, tuple):
                return (out[0] + alpha * v_gpu,) + tuple(out[1:])
            return out + alpha * v_gpu

        target.register_forward_hook(hook)
        return f"hook on layer {layer_idx}, ||v_gpu||={v_gpu.norm().item():.3f}"

    msg = llm.llm_engine.apply_model(install_hook)
    print(f"[infer_sv_vllm] {msg}, α={alpha}, ||v_raw||={vnorm:.3f}", flush=True)

    # Run greedy generation
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
    outs = llm.generate(prompts, sp)
    # Hook lives on worker; vLLM cleans up at process exit.

    # Score
    results = []
    correct = 0
    total_tokens = 0
    for ex, out in zip(examples, outs):
        text = out.outputs[0].text
        is_c = main_is_correct(text, ex, args.dataset)
        correct += int(is_c)
        n_tok = len(out.outputs[0].token_ids)
        total_tokens += n_tok
        results.append({
            "question": ex["question"], "gt": ex.get("answer", ""),
            "pred": text, "correct": is_c, "n_tokens": n_tok,
        })

    n = len(examples)
    acc = correct / max(n, 1)
    avg_tok = total_tokens / max(1, n)

    out_dir = PROJECT_ROOT / "outputs" / "main_paper_eval" / args.out_name
    os.makedirs(out_dir, exist_ok=True)
    save_with_retry(results, str(out_dir / "predictions.json"), saver="json")
    metrics = {"exp": args.out_name, "dataset": args.dataset, "split": args.split,
               "n": n, "accuracy": acc, "avg_tokens": avg_tok,
               "model_path": args.model_path, "acts_cache": args.acts_cache,
               "layer": args.layer, "alpha": args.alpha, "v_mean_norm": vnorm}
    save_with_retry(metrics, str(out_dir / "metrics.json"), saver="json")
    print(f"\n[{args.out_name}] acc={acc*100:.2f}% ({correct}/{n}) avg_tok={avg_tok:.1f} α={args.alpha} L={args.layer}")

    line = (f"\n- **{args.out_name}** ({args.dataset}/{args.split}, α={args.alpha}, layer={args.layer}, vLLM): "
            f"acc={acc*100:.2f}%  avg_tok={avg_tok:.1f}  n={n}")
    try:
        with open(args.results_md, "a") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"WARN: could not append: {e}")

    if merged_dir.startswith("/tmp/isv_v_merged_"):
        shutil.rmtree(merged_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
