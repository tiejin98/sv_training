"""Generate teacher predictions on a training split via vLLM greedy decoding.

For each training example in load_main_split(dataset, "train"):
 - build prompt via main_build_messages + apply_chat_template
 - vLLM greedy (temperature=0, max_new_tokens configurable)
 - save list of {"idx":int, "question":str, "gt_answer":str, "pred":str,
                 "pred_correct": bool, "n_tokens": int} to JSON

Usage:
    python teacher_gen.py \\
        --model Qwen/Qwen3-8B --dataset gsm8k --split train \\
        --max_new_tokens 4096 --enable_thinking 1 \\
        --out teacher_gen/Qwen3-8B--gsm8k--think.json
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path

from dataset_utils import (  # type: ignore
    load_main_split, main_build_messages, main_is_correct, main_extract_pred_answer,
)
from beegfs_safe_save import save_with_retry  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id of teacher model")
    ap.add_argument("--dataset", required=True,
                    choices=["gsm8k", "math", "mbpp", "strategyqa"])
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--enable_thinking", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--tensor_parallel_size", type=int, default=1,
                    help="Shard model across N GPUs (use 2 for Gemma-4-26B)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"[teacher_gen] Loading dataset {args.dataset} split={args.split} ...")
    examples = load_main_split(args.dataset, args.split, cache_dir=CACHE_DIR)
    if args.limit:
        examples = examples[:args.limit]
    n = len(examples)

    print(f"[teacher_gen] Loading tokenizer for {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=CACHE_DIR,
                                         trust_remote_code=True)
    kw_thinking = {}
    try:
        _ = tok.apply_chat_template(
            [{"role": "user", "content": "x"}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=bool(args.enable_thinking),
        )
        kw_thinking = {"enable_thinking": bool(args.enable_thinking)}
    except TypeError:
        kw_thinking = {}

    prompts = []
    for ex in examples:
        msgs = main_build_messages(ex["question"], args.dataset)
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **kw_thinking
        )
        prompts.append(prompt)

    # vLLM max_model_len needs to accommodate prompt + max_new_tokens
    mml = max(2048, args.max_new_tokens + 2048)

    print(f"[teacher_gen] Loading vLLM model {args.model} (max_model_len={mml}, tp={args.tensor_parallel_size}) ...")
    llm = LLM(model=args.model, download_dir=CACHE_DIR,
              dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              trust_remote_code=True, max_model_len=mml,
              tensor_parallel_size=args.tensor_parallel_size)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_new_tokens)
    print(f"[teacher_gen] Generating {n} samples ...")
    outs = llm.generate(prompts, sp)

    records = []
    n_correct = 0
    n_empty = 0
    for i, (ex, req) in enumerate(zip(examples, outs)):
        out_text = req.outputs[0].text if req.outputs else ""
        if not out_text.strip():
            n_empty += 1
        ok = False
        if out_text.strip():
            try:
                ok = main_is_correct(out_text, ex, args.dataset)
            except Exception:
                ok = False
        if ok:
            n_correct += 1
        n_tokens = len(req.outputs[0].token_ids) if req.outputs else 0
        records.append({
            "idx": i,
            "question": ex["question"],
            "gt_answer": ex["answer"],
            "pred": out_text,
            "pred_correct": bool(ok),
            "n_tokens": int(n_tokens),
        })

    print(f"[teacher_gen] correct={n_correct}/{n}  empty={n_empty}  "
          f"keep_rate={100.0*n_correct/max(n,1):.1f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_with_retry(records, args.out, saver="json")
    print(f"[teacher_gen] saved → {args.out}")


if __name__ == "__main__":
    main()
