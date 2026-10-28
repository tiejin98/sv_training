"""Command-line interface: ``sv-train <command> ...`` (see ``sv-train <command> --help``)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__

DEFAULT_CACHE = os.environ.get("HF_HOME") or None
METHODS = ("plain", "custom", "mean_sv_add", "slerp", "per_inst_add", "sv_first_k",
           "sv_multi_layer", "random_gauss")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cache_dir", default=DEFAULT_CACHE, help="HF cache dir (default: $HF_HOME)")
    p.add_argument("--limit", type=int, default=None, help="use only the first N examples")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

TRAIN_EPILOG = """\
steering vectors
  Any vector file can be injected at any layer. Accepted formats: .pt/.npy/
  .safetensors holding a [D] tensor (one vector for all samples) or an [N, D]
  tensor (row i is used for training example i), a {"direction": v} dict, a
  {"vectors": {layer: v}} dict, or an acts cache from `sv-train acts`.

  single injection:
    sv-train train ... --sv my_vec.pt --layer 15 --alpha 0.5 --op add
  several layers / vectors (repeat --inject; unspecified keys use the flags):
    sv-train train ... --inject path=a.pt,layer=10,alpha=0.1 \\
                       --inject path=b.pt,layer=15,alpha=0.5,op=slerp

paper presets (--method, vectors built from --acts_cache):
  mean_sv_add     dataset-mean gt-pred, additive            (alpha 0.5)
  slerp           per-sample unit gt-pred, Slerp            (alpha 0.1)
  per_inst_add    per-sample unit gt-pred, additive
  sv_first_k      per-sample Slerp on the first --k answer tokens
  sv_multi_layer  per-sample Slerp at every layer in --multi_layers
  random_gauss    per-sample random unit directions (control)
"""


def _train_parser(sub) -> None:
    p = sub.add_parser("train", help="LoRA SFT / distillation with steering vectors",
                       epilog=TRAIN_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", "--model_path", dest="model", required=True,
                   help="HF model id or local dir")
    p.add_argument("--dataset", required=True,
                   help="gsm8k | math | mbpp | strategyqa, or a .json/.jsonl with question/answer")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--method", default=None, choices=METHODS,
                   help="default: custom if a vector is given, else plain")

    g = p.add_argument_group("steering")
    g.add_argument("--sv", "--acts_cache", "--mean_dir_override", dest="sv", default=None,
                   help="steering vector file (or acts cache for the paper presets)")
    g.add_argument("--layer", type=int, default=None,
                   help="0-based block index (default L/2+1, e.g. 15 for 28 layers)")
    g.add_argument("--alpha", type=float, default=None,
                   help="add: h + alpha*v; slerp: rotation fraction (default 0.5 add / 0.1 slerp)")
    g.add_argument("--op", choices=["add", "slerp"], default=None)
    g.add_argument("--sv_kind", choices=["auto", "mean", "per_sample"], default=None,
                   help="for acts caches / [N, D] files: reduce to the mean or keep per-sample")
    g.add_argument("--normalize", action="store_true", help="unit-normalise vectors")
    g.add_argument("--first_k", "--k", dest="first_k", type=int, default=None,
                   help="steer only the first K answer tokens")
    g.add_argument("--skip_first_k", "--sv_skip_first_k_answer", dest="skip_first_k", type=int,
                   default=0, help="do not steer the first K answer tokens")
    g.add_argument("--all_tokens", action="store_true",
                   help="steer prompt tokens too (default: answer tokens only)")
    g.add_argument("--multi_layers", default="10,15,20", help="layers for sv_multi_layer")
    g.add_argument("--inject", action="append", default=[], metavar="SPEC",
                   help="extra injection 'path=...,layer=...,alpha=...,op=...,kind=...,"
                        "normalize=1,first_k=...'; repeatable")

    d = p.add_argument_group("data / distillation")
    d.add_argument("--teacher_json", "--distill_teacher_json", dest="teacher_json", default=None,
                   help="train on teacher outputs (from `sv-train generate`) instead of GT")
    d.add_argument("--keep_only_correct", "--distill_keep_only_correct", dest="keep_only_correct",
                   type=int, default=1, help="drop teacher records with pred_correct=False")
    d.add_argument("--append_eos", "--append_student_eos", dest="append_eos", type=int, default=0)
    d.add_argument("--enable_thinking", type=int, default=0)
    d.add_argument("--max_context_len", type=int, default=2048)

    o = p.add_argument_group("optimisation (paper defaults)")
    o.add_argument("--lr", type=float, default=5e-5)
    o.add_argument("--epochs", "--num_train_epochs", dest="epochs", type=float, default=1.0)
    o.add_argument("--max_steps", type=int, default=-1)
    o.add_argument("--batch_size", type=int, default=1)
    o.add_argument("--grad_accum", type=int, default=8)
    o.add_argument("--lora_r", type=int, default=8)
    o.add_argument("--lora_alpha", type=int, default=16)
    o.add_argument("--lora_dropout", type=float, default=0.05)
    o.add_argument("--full_finetune", action="store_true", help="train all weights (no LoRA)")
    o.add_argument("--seed", type=int, default=42)
    o.add_argument("--gradient_checkpointing", type=int, default=0)
    o.add_argument("--logging_steps", type=int, default=20)
    o.add_argument("--report_to", default="none", help="e.g. wandb")
    o.add_argument("--run_name", default=None)
    _add_common(p)
    p.set_defaults(func=run_train)


def build_injections(args) -> List["InjectSpec"]:
    from .train import PRESETS, InjectSpec

    method = args.method or ("custom" if (args.sv or args.inject) else "plain")
    if method == "plain":
        if args.sv or args.inject:
            raise SystemExit("--method plain does not take --sv/--inject")
        return []
    if method == "custom":
        op = args.op or "add"
        base = InjectSpec(path=args.sv, layer=args.layer,
                          alpha=args.alpha if args.alpha is not None else (0.5 if op == "add" else 0.1),
                          op=op, kind=args.sv_kind or "auto", normalize=args.normalize,
                          first_k=args.first_k, skip_first_k=args.skip_first_k,
                          answer_only=not args.all_tokens)
        specs = [base] if args.sv else []
        specs += [InjectSpec.parse(s, base) for s in args.inject]
        if not specs:
            raise SystemExit("--method custom needs --sv or --inject")
        for s in specs:
            if not s.path:
                raise SystemExit(f"injection without a vector path: {s}")
        return specs
    if method not in PRESETS:
        raise SystemExit(f"unknown --method {method}")

    pre = PRESETS[method]
    if pre["kind"] != "random" and not args.sv:
        raise SystemExit(f"--method {method} needs --acts_cache (or --sv)")
    kind = args.sv_kind or pre["kind"]
    spec = dict(path=args.sv, layer=args.layer,
                alpha=args.alpha if args.alpha is not None else pre["alpha"],
                op=args.op or pre["op"], kind=kind, normalize=pre["normalize"] or args.normalize,
                first_k=args.first_k if method == "sv_first_k" else None,
                skip_first_k=args.skip_first_k, answer_only=not args.all_tokens)
    if method == "sv_first_k" and spec["first_k"] is None:
        spec["first_k"] = 32
    if method == "sv_multi_layer":
        return [InjectSpec(**{**spec, "layer": int(li)}) for li in args.multi_layers.split(",")]
    return [InjectSpec(**spec)]


def run_train(args) -> None:
    from .train import TrainConfig, train

    cfg = TrainConfig(
        model=args.model, dataset=args.dataset, output_dir=args.output_dir,
        injections=build_injections(args), teacher_json=args.teacher_json,
        keep_only_correct=bool(args.keep_only_correct), append_eos=bool(args.append_eos),
        enable_thinking=bool(args.enable_thinking), cache_dir=args.cache_dir, limit=args.limit,
        lr=args.lr, epochs=args.epochs, max_steps=args.max_steps, batch_size=args.batch_size,
        grad_accum=args.grad_accum, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, full_finetune=args.full_finetune,
        max_context_len=args.max_context_len, seed=args.seed,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        logging_steps=args.logging_steps, report_to=args.report_to, run_name=args.run_name)
    train(cfg)


# ---------------------------------------------------------------------------
# other commands
# ---------------------------------------------------------------------------


def run_generate(args) -> None:
    from .generate import generate
    generate(args.model, args.dataset, args.out, split=args.split,
             max_new_tokens=args.max_new_tokens, enable_thinking=bool(args.enable_thinking),
             temperature=args.temperature, top_p=args.top_p, gpu_mem_util=args.gpu_mem_util,
             tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len,
             cache_dir=args.cache_dir, limit=args.limit)


def run_acts(args) -> None:
    from .acts import compute_acts
    compute_acts(args.model, args.dataset, args.preds_json, args.out,
                 reference_json=args.reference_json, enable_thinking=bool(args.enable_thinking),
                 max_context_len=args.max_context_len, cache_dir=args.cache_dir, limit=args.limit)


def run_vector(args) -> None:
    import torch

    from .vectors import load_vector, save_vector
    v = load_vector(args.acts, layer=args.layer, kind=args.kind)
    if args.normalize:
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if args.half:
        v = v.to(torch.float16)
    save_vector(v, args.out, meta={"source": os.path.basename(args.acts), "layer": args.layer,
                                   "kind": args.kind, "normalized": args.normalize})
    print(f"[vector] {tuple(v.shape)} -> {args.out}")


def run_eval(args) -> None:
    from .evaluate import evaluate
    out_dir = args.out_dir or os.path.join(
        "outputs", "eval", f"{os.path.basename(os.path.normpath(args.model))}-{args.dataset}"
                           f"-a{args.alpha:g}")
    evaluate(args.model, args.dataset, out_dir, base_model=args.base_model, split=args.split,
             sv=args.sv, layer=args.layer, alpha=args.alpha, sv_kind=args.sv_kind,
             max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len,
             gpu_mem_util=args.gpu_mem_util, tensor_parallel_size=args.tensor_parallel_size,
             cache_dir=args.cache_dir, limit=args.limit)


def run_score(args) -> None:
    from .evaluate import score_predictions
    score_predictions(args.predictions, args.dataset, split=args.split, cache_dir=args.cache_dir)


def run_inspect(args) -> None:
    """Summarise a vector / acts cache / prediction JSON (CPU only)."""
    import torch
    for path in args.files:
        print(f"== {path}")
        if path.endswith(".json") or path.endswith(".jsonl"):
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(x) for x in f] if path.endswith(".jsonl") else json.load(f)
            print(f"  {len(rows)} records, fields: {sorted(rows[0].keys()) if rows else []}")
            if rows and "pred_correct" in rows[0]:
                k = sum(bool(r["pred_correct"]) for r in rows)
                print(f"  pred_correct: {k}/{len(rows)} ({100.0 * k / len(rows):.1f}%)")
            if rows and "pred" in rows[0]:
                empty = sum(not (r.get("pred") or "").strip() for r in rows)
                avg = sum(len(r.get("pred") or "") for r in rows) / len(rows)
                print(f"  empty preds: {empty}, mean pred length: {avg:.0f} chars")
            continue
        from .vectors import _load_raw, load_vector
        obj = _load_raw(path)
        if isinstance(obj, dict):
            for k, val in obj.items():
                desc = (f"tensor{tuple(val.shape)} {val.dtype}" if torch.is_tensor(val)
                        else (val if not isinstance(val, dict) else f"dict({len(val)})"))
                print(f"  {k}: {desc}")
            if "gt_acts" in obj and args.layer is not None:
                v = load_vector(path, layer=args.layer, kind="mean")
                print(f"  mean SV @ layer {args.layer}: ||v||={v.norm():.4f}")
            if "meta" in obj and isinstance(obj["meta"], dict):
                for k, val in obj["meta"].items():
                    print(f"    meta.{k}: {val}")
        else:
            print(f"  tensor{tuple(obj.shape)} {obj.dtype}")
        try:
            v = load_vector(path, layer=args.layer)
            norms = v.norm(dim=-1)
            print(f"  as steering vector: shape={tuple(v.shape)} ||v|| mean={norms.mean():.4f}")
        except Exception as e:  # noqa: BLE001 - inspection is best effort
            print(f"  (not loadable as a vector without more info: {e})")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sv-train",
                                 description="Steering vectors as a training signal (NeurIPS 2026).")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="greedy vLLM outputs on a split (teacher data or "
                                        "student bootstrap predictions)")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--enable_thinking", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    _add_common(p)
    p.set_defaults(func=run_generate)

    p = sub.add_parser("acts", help="answer-token hidden states on reference vs. own predictions")
    p.add_argument("--model", required=True, help="the student model")
    p.add_argument("--dataset", required=True)
    p.add_argument("--preds_json", required=True,
                   help="student bootstrap predictions (from `sv-train generate`)")
    p.add_argument("--reference_json", "--teacher_json", dest="reference_json", default=None,
                   help="teacher outputs for distillation (default: dataset GT answers)")
    p.add_argument("--out", required=True)
    p.add_argument("--enable_thinking", type=int, default=0)
    p.add_argument("--max_context_len", type=int, default=2048)
    _add_common(p)
    p.set_defaults(func=run_acts)

    p = sub.add_parser("vector", help="extract a steering vector at one layer from an acts cache")
    p.add_argument("--acts", required=True, help="acts cache (or any vector file)")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--kind", choices=["mean", "per_sample"], default="mean")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--half", action="store_true", help="store as float16 (smaller files)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=run_vector)

    _train_parser(sub)

    p = sub.add_parser("eval", help="greedy vLLM evaluation, optionally with inference-time SV")
    p.add_argument("--model", "--model_path", dest="model", required=True,
                   help="adapter dir from `sv-train train`, merged model dir, or HF id")
    p.add_argument("--base_model", "--base_model_name", dest="base_model", default=None)
    p.add_argument("--dataset", required=True,
                   help="gsm8k | math | mbpp | strategyqa | aime24 | aime25")
    p.add_argument("--split", default="test")
    p.add_argument("--sv", "--acts_cache", dest="sv", default=None,
                   help="vector for inference-time steering (only used when --alpha != 0)")
    p.add_argument("--sv_kind", choices=["auto", "mean"], default="mean")
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.0, help="0 = no SV (paper setting)")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--out_dir", default=None, help="default outputs/eval/<model>-<dataset>-a<alpha>")
    _add_common(p)
    p.set_defaults(func=run_eval)

    p = sub.add_parser("score", help="re-score a predictions.json (CPU only)")
    p.add_argument("--predictions", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--cache_dir", default=DEFAULT_CACHE)
    p.set_defaults(func=run_score)

    p = sub.add_parser("inspect", help="summarise vector / acts / prediction files (CPU only)")
    p.add_argument("files", nargs="+")
    p.add_argument("--layer", type=int, default=None)
    p.set_defaults(func=run_inspect)
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
