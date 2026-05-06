"""Compute per-instance GT & Pred acts caches for SV-SFT main-paper experiments.

For each training example i in `load_main_split(dataset, 'train', ...)`:

  gt_acts[i,ℓ,:]   = mean_t h_ℓ(x_i, y_GT_i)[answer_token_t]
  pred_acts[i,ℓ,:] = mean_t h_ℓ(x_i, ŷ_i)[answer_token_t]

where ŷ_i is the student's greedy bootstrap prediction produced with vLLM
(same chat template as main_sft.py).

Output: {"gt_acts": [N, L, D], "pred_acts": [N, L, D], "meta": {...}}
Sample positions with either truncation or empty prediction get a zero row
(per-instance delta = 0 → SV hook is a no-op for that sample).

Usage:
    python main_compute_acts.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct --dataset gsm8k \\
        --out_path acts_cache/1.5B_gsm8k_acts.pt
"""
from __future__ import annotations
import os, gc, json, argparse, tempfile, subprocess
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataset_utils import load_main_split, main_build_messages  # type: ignore
from beegfs_safe_save import save_with_retry  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))


# ---------------------------------------------------------------------------
# Stage 1 — vLLM bootstrap predictions
# ---------------------------------------------------------------------------


def generate_bootstrap_preds(model_path: str, dataset: str, max_new_tokens: int,
                              enable_thinking: bool, temperature: float,
                              top_p: float) -> List[str]:
    from vllm import LLM, SamplingParams
    # Load dataset
    examples = load_main_split(dataset, "train", cache_dir=CACHE_DIR)
    # Load tokenizer once for prompt formatting
    tok = AutoTokenizer.from_pretrained(model_path, cache_dir=CACHE_DIR, trust_remote_code=True)
    kw_thinking = {}
    try:
        _ = tok.apply_chat_template([{"role": "user", "content": "x"}],
                                    tokenize=False, add_generation_prompt=True,
                                    enable_thinking=enable_thinking)
        kw_thinking = {"enable_thinking": enable_thinking}
    except TypeError:
        kw_thinking = {}

    prompts = []
    for ex in examples:
        user_msgs = main_build_messages(ex["question"], dataset)
        prompt_text = tok.apply_chat_template(
            user_msgs, tokenize=False, add_generation_prompt=True, **kw_thinking
        )
        prompts.append(prompt_text)

    print(f"[bootstrap] loading vLLM {model_path} ...")
    llm = LLM(model=model_path, download_dir=CACHE_DIR, dtype="bfloat16",
              gpu_memory_utilization=0.85, trust_remote_code=True,
              max_model_len=4096 if enable_thinking else 2048)
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    print(f"[bootstrap] generating {len(prompts)} preds ...")
    outs = llm.generate(prompts, sp)
    # vLLM may reorder; stitch back by index from request_id if needed
    # In our usage outs preserves input order
    preds = []
    for req in outs:
        preds.append(req.outputs[0].text if req.outputs else "")
    # Release vLLM & free GPU mem
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return preds


# ---------------------------------------------------------------------------
# Stage 2 — HF teacher-forced hidden-state mean on answer tokens
# ---------------------------------------------------------------------------


@torch.no_grad()
def mean_hidden_all_layers(model, tokenizer, prompt_text: str, answer_text: str,
                            max_length: int) -> Optional[torch.Tensor]:
    """Return [num_hidden_layers, hidden_dim] mean-pooled over answer tokens, or None."""
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
    hidden = out.hidden_states  # tuple len = L+1 (0 = embed)
    num_layers = model.config.num_hidden_layers
    ans_start, ans_end = prompt_len, seq_len
    per_layer = []
    for li in range(1, num_layers + 1):
        hs = hidden[li][:, ans_start:ans_end, :]  # [1, K, D]
        per_layer.append(hs.mean(dim=1).squeeze(0).to(torch.float32).cpu())
    return torch.stack(per_layer, dim=0)  # [L, D]


def build_full_text(tok, user_msgs, answer: str, enable_thinking: bool) -> str:
    kw = {}
    try:
        _ = tok.apply_chat_template(user_msgs, tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=enable_thinking)
        kw = {"enable_thinking": enable_thinking}
    except TypeError:
        pass
    full_msgs = user_msgs + [{"role": "assistant", "content": answer}]
    return tok.apply_chat_template(full_msgs, tokenize=False,
                                    add_generation_prompt=False, **kw)


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


def compute_acts_for_split(model_path: str, dataset: str, preds: List[str],
                            out_path: str, enable_thinking: bool,
                            max_context_len: int):
    examples = load_main_split(dataset, "train", cache_dir=CACHE_DIR)
    assert len(examples) == len(preds), f"{len(examples)} vs {len(preds)}"

    print(f"[acts] loading HF {model_path} ...")
    tok = AutoTokenizer.from_pretrained(model_path, cache_dir=CACHE_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()
    L = model.config.num_hidden_layers
    D = model.config.hidden_size

    N = len(examples)
    gt_acts = torch.zeros(N, L, D, dtype=torch.float32)
    pred_acts = torch.zeros(N, L, D, dtype=torch.float32)
    gt_valid = 0
    pred_valid = 0
    skipped_pred_empty = 0

    pbar = tqdm(range(N), desc=f"acts {dataset}", dynamic_ncols=True)
    for i in pbar:
        ex = examples[i]
        user_msgs = main_build_messages(ex["question"], dataset)
        prompt_text = build_prompt_text(tok, user_msgs, enable_thinking)

        # v2 semantics (analysis_layer_navigator.collect_per_sample_acts):
        # full_text = prompt_text + raw answer string (no chat_template wrap,
        # no trailing <|im_end|>). Same for pred.

        # GT
        g = mean_hidden_all_layers(model, tok, prompt_text, ex["answer"], max_context_len)
        if g is not None:
            gt_acts[i] = g
            gt_valid += 1

        # Pred
        pred_text = (preds[i] or "").strip()
        if not pred_text:
            skipped_pred_empty += 1
        else:
            p = mean_hidden_all_layers(model, tok, prompt_text, pred_text, max_context_len)
            if p is not None:
                pred_acts[i] = p
                pred_valid += 1

        if (i + 1) % 100 == 0:
            pbar.set_postfix(gt_valid=gt_valid, pred_valid=pred_valid,
                             pred_empty=skipped_pred_empty)

    meta = {
        "model_path": model_path,
        "dataset": dataset,
        "N": N,
        "num_layers": L,
        "hidden_size": D,
        "enable_thinking": enable_thinking,
        "max_context_len": max_context_len,
        "gt_valid": gt_valid,
        "pred_valid": pred_valid,
        "skipped_pred_empty": skipped_pred_empty,
        "note": "gt_acts / pred_acts are mean-pooled over answer tokens; "
                "layers index 0..L-1 correspond to transformer blocks 1..L "
                "(embedding layer excluded).",
    }

    print(f"\n[acts] gt_valid={gt_valid}/{N}  pred_valid={pred_valid}/{N}  "
          f"pred_empty={skipped_pred_empty}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_with_retry({"gt_acts": gt_acts, "pred_acts": pred_acts, "meta": meta},
                    out_path, saver="torch")
    print(f"[acts] saved -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF path (e.g. Qwen/Qwen2.5-1.5B-Instruct) or local dir")
    ap.add_argument("--dataset", required=True,
                    choices=["gsm8k", "math", "mbpp", "strategyqa"])
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--enable_thinking", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=512,
                    help="bootstrap gen budget")
    ap.add_argument("--max_context_len", type=int, default=2048,
                    help="teacher-forced context truncation (match main_sft.py)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--preds_json", default=None,
                    help="Optional pre-generated bootstrap preds JSON "
                         "[{\"idx\":int,\"pred\":str}, ...]")
    args = ap.parse_args()

    # --- Stage 1: bootstrap preds via vLLM (unless pre-loaded) ---
    if args.preds_json and os.path.isfile(args.preds_json):
        print(f"[bootstrap] loading preds from {args.preds_json}")
        with open(args.preds_json) as f:
            raw = json.load(f)
        preds_map = {r["idx"]: r["pred"] for r in raw}
        n_total = len(load_main_split(args.dataset, "train", cache_dir=CACHE_DIR))
        preds = [preds_map.get(i, "") for i in range(n_total)]
    else:
        preds = generate_bootstrap_preds(
            args.model, args.dataset, args.max_new_tokens,
            bool(args.enable_thinking), args.temperature, args.top_p,
        )
        # Dump preds alongside the acts cache for reuse / debugging
        preds_path = args.out_path.replace(".pt", "_bootpreds.json")
        if preds_path == args.out_path:
            preds_path = args.out_path + ".bootpreds.json"
        os.makedirs(os.path.dirname(preds_path) or ".", exist_ok=True)
        save_with_retry([{"idx": i, "pred": p} for i, p in enumerate(preds)],
                        preds_path, saver="json")
        print(f"[bootstrap] saved preds -> {preds_path}")

    # --- Stage 2: HF hidden-state extraction ---
    compute_acts_for_split(
        args.model, args.dataset, preds, args.out_path,
        enable_thinking=bool(args.enable_thinking),
        max_context_len=args.max_context_len,
    )


if __name__ == "__main__":
    main()
