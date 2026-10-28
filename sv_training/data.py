"""Dataset adapters: loading, prompt construction, answer extraction, scoring.

Built-in datasets: gsm8k, math, mbpp, strategyqa (train + test) and
aime24, aime25 (eval only). Any other value of ``dataset`` is treated as a
path to a local ``.json`` / ``.jsonl`` file with ``question`` and ``answer``
fields (see :func:`load_split`).

Every example is normalised into a dict with at least
``{"question": str, "answer": str, "dataset": str}``.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

BUILTIN_DATASETS = ("gsm8k", "math", "mbpp", "strategyqa", "aime24", "aime25")
TRAIN_DATASETS = ("gsm8k", "math", "mbpp", "strategyqa")

# ---------------------------------------------------------------------------
# Instructions / prompts (kept verbatim from the paper experiments)
# ---------------------------------------------------------------------------

_GSM8K_INSTRUCTION = (
    "You are a helpful math assistant.\n"
    "Solve the problem step by step.\n"
    "You MUST end your response with a final line in the EXACT format:\n"
    "#### {final answer}\n"
    "The final line must be the last line. Do not add anything after it.\n"
)

_MATH_INSTRUCTION = (
    "You are an expert mathematician.\n"
    "Solve the following problem step by step, showing your work.\n"
    "You MUST put your final answer inside \\boxed{} at the very end of your response.\n"
    "Example last line: Therefore the answer is $\\boxed{42}$.\n"
)

_MBPP_INSTRUCTION = (
    "You are an expert Python programmer.\n"
    "Write a Python function that solves the following task. "
    "Your response MUST be a single Python function (no imports, no tests, no comments) "
    "that matches the required signature. "
    "Wrap ONLY the function code inside a ```python ... ``` fenced block with no additional text before or after.\n"
)

_STRATEGYQA_INSTRUCTION = (
    "You are a careful reasoner.\n"
    "Answer the following yes/no question by first reasoning step by step, "
    "then ending your response with a final line in the EXACT format:\n"
    "#### yes\n"
    "or\n"
    "#### no\n"
)

_AIME_INSTRUCTION = (
    "You are a precise mathematician. Solve the AIME problem step by step. "
    "The answer is an integer from 0 to 999. "
    "End your response with the answer enclosed in \\boxed{}, e.g. \\boxed{123}."
)


def is_builtin(dataset: str) -> bool:
    return dataset in BUILTIN_DATASETS


def build_messages(question: str, dataset: str) -> List[Dict[str, str]]:
    """Single-turn chat messages for a question."""
    if dataset == "gsm8k":
        return [{"role": "user", "content": _GSM8K_INSTRUCTION + "\nProblem:\n" + question}]
    if dataset == "math":
        return [{"role": "user", "content": _MATH_INSTRUCTION + "\nProblem:\n" + question}]
    if dataset == "mbpp":
        return [{"role": "user", "content": _MBPP_INSTRUCTION + "\nProblem:\n" + question}]
    if dataset == "strategyqa":
        return [{"role": "user", "content": _STRATEGYQA_INSTRUCTION + "\nQuestion:\n" + question}]
    if dataset in ("aime24", "aime25"):
        return [{"role": "user", "content": _AIME_INSTRUCTION + "\n\nProblem:\n" + question}]
    # Custom file: the question is used verbatim as the user turn.
    return [{"role": "user", "content": question}]


def apply_chat_template(tokenizer, messages, add_generation_prompt: bool,
                        enable_thinking: Optional[bool] = None) -> str:
    """`apply_chat_template` that passes `enable_thinking` only when supported."""
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_gsm8k(split: str, cache_dir: Optional[str]) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", cache_dir=cache_dir)
    return [{"question": ex["question"], "answer": ex["answer"], "dataset": "gsm8k",
             "has_solution": True} for ex in ds[split]]


def _load_math(split: str, cache_dir: Optional[str]) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    # The canonical HF id has changed over time; try known mirrors in order.
    # nlile/hendrycks-MATH-benchmark: train=12000, test=500 (MATH-500).
    candidates = [
        ("nlile/hendrycks-MATH-benchmark", None),
        ("hendrycks/competition_math", None),
        ("lighteval/MATH", "all"),
        ("EleutherAI/hendrycks_math", "all"),
    ]
    ds = None
    for name, cfg in candidates:
        try:
            ds = load_dataset(name, cfg, cache_dir=cache_dir) if cfg else \
                load_dataset(name, cache_dir=cache_dir)
            print(f"[data] Loaded MATH from '{name}'")
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError("Could not load MATH. Tried: " + ", ".join(n for n, _ in candidates))
    examples = []
    for ex in ds[split]:
        examples.append({
            "question": ex.get("problem") or ex.get("question") or "",
            # Full LaTeX solution (ends with \boxed{}) is preferred over the bare answer.
            "answer": ex.get("solution") or ex.get("answer") or "",
            "level": ex.get("level", ""),
            "type": ex.get("subject") or ex.get("type") or "",
            "dataset": "math",
            "has_solution": True,
        })
    return examples


def _load_mbpp(split: str, cache_dir: Optional[str]) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    # google-research-datasets/mbpp (full): 374 train, 500 test.
    ds = load_dataset("google-research-datasets/mbpp", "full", cache_dir=cache_dir)
    if split not in ("train", "test"):
        raise ValueError(f"MBPP: unsupported split {split!r}")
    examples = []
    for ex in ds[split]:
        tests = ex.get("test_list", []) or []
        problem_text = ex.get("text") or ex.get("prompt", "")
        if tests:
            problem_text = problem_text.rstrip() + "\nYour code should pass these tests:\n" + "\n".join(tests)
        examples.append({
            "question": problem_text,
            "answer": ex.get("code", ""),
            "task_id": ex["task_id"],
            "test_list": tests,
            "test_setup_code": ex.get("test_setup_code", ""),
            "dataset": "mbpp",
            "has_solution": True,
        })
    return examples


def _load_strategyqa(split: str, cache_dir: Optional[str]) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = None
    for name in ("ChilleD/StrategyQA", "voidful/StrategyQA", "wics/strategy-qa"):
        try:
            ds = load_dataset(name, cache_dir=cache_dir)
            print(f"[data] Loaded StrategyQA from '{name}'")
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError("Could not load StrategyQA")
    if split == "train":
        data = ds.get("train") or list(ds.values())[0]
    elif split == "test":
        data = ds.get("test") or ds.get("validation") or list(ds.values())[-1]
    else:
        raise ValueError(f"StrategyQA: unsupported split {split!r}")

    examples = []
    for ex in data:
        ans = ex.get("answer", None)
        if isinstance(ans, bool):
            ans_str = "yes" if ans else "no"
        else:
            ans_str = str(ans).strip().lower()
            if ans_str in ("true", "1"):
                ans_str = "yes"
            elif ans_str in ("false", "0"):
                ans_str = "no"
        # ChilleD/StrategyQA stores `facts` as one prose string; others use a list.
        facts_raw = ex.get("facts")
        reasoning = ""
        if isinstance(facts_raw, str):
            reasoning = facts_raw.strip()
        elif isinstance(facts_raw, list) and facts_raw:
            reasoning = "\n".join(f"- {f}" for f in facts_raw if f)
        examples.append({
            "question": ex.get("question", ""),
            "answer": f"{reasoning}\n#### {ans_str}" if reasoning else f"#### {ans_str}",
            "bool_answer": ans_str,
            "dataset": "strategyqa",
            "has_solution": bool(reasoning),
        })
    return examples


def _load_aime24(cache_dir: Optional[str]) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/AIME_2024", cache_dir=cache_dir)
    return [{"question": ex["Problem"], "answer": str(int(ex["Answer"])),
             "int_answer": int(ex["Answer"]), "id": ex.get("ID", ""),
             "dataset": "aime24", "has_solution": True} for ex in ds["train"]]


def _load_aime25(cache_dir: Optional[str]) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    examples = []
    for cfg in ("AIME2025-I", "AIME2025-II"):
        ds = load_dataset("opencompass/AIME2025", cfg, cache_dir=cache_dir)
        for ex in ds[list(ds.keys())[0]]:
            m = re.match(r"-?\d+", str(ex["answer"]).strip())
            if m is None:
                print(f"[data] aime25: skipping non-integer answer {ex['answer']!r}")
                continue
            n = int(m.group(0))
            examples.append({"question": ex["question"], "answer": str(n), "int_answer": n,
                             "id": cfg, "dataset": "aime25", "has_solution": True})
    return examples


def _load_file(path: str, question_field: str = "question",
               answer_field: str = "answer") -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            rows = [json.loads(line) for line in f if line.strip()]
        else:
            rows = json.load(f)
    examples = []
    for i, r in enumerate(rows):
        if question_field not in r:
            raise KeyError(f"{path} row {i} has no '{question_field}' field")
        examples.append({**r, "question": r[question_field],
                         "answer": r.get(answer_field, ""), "dataset": "custom"})
    return examples


def load_split(dataset: str, split: str = "train", cache_dir: Optional[str] = None,
               limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load a split as a list of example dicts.

    ``dataset`` is a built-in name or a path to a ``.json``/``.jsonl`` file
    (list of ``{"question": ..., "answer": ...}``; ``split`` is ignored for files).
    Example order defines the sample index used by per-sample steering vectors.
    """
    if dataset == "gsm8k":
        ex = _load_gsm8k(split, cache_dir)
    elif dataset == "math":
        ex = _load_math(split, cache_dir)
    elif dataset == "mbpp":
        ex = _load_mbpp(split, cache_dir)
    elif dataset == "strategyqa":
        ex = _load_strategyqa(split, cache_dir)
    elif dataset == "aime24":
        ex = _load_aime24(cache_dir)
    elif dataset == "aime25":
        ex = _load_aime25(cache_dir)
    elif os.path.isfile(dataset):
        ex = _load_file(dataset)
    else:
        raise ValueError(f"Unknown dataset {dataset!r}: use one of {BUILTIN_DATASETS} "
                         "or a path to a .json/.jsonl file")
    return ex[:limit] if limit else ex


# ---------------------------------------------------------------------------
# Answer extraction & correctness
# ---------------------------------------------------------------------------

_HASH_LINE_RE = re.compile(r"^####\s*(.+?)\s*$", re.MULTILINE)
_GT_HASH_RE = re.compile(r"####\s*(.*?)(?:\n|$)", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_YESNO_HASH_RE = re.compile(r"####\s*(yes|no)\b", re.IGNORECASE)
_YESNO_FALLBACK_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_LAST_NUMBER_RE = re.compile(r"(-?\$?\d[\d,]*(?:\.\d+)?)")


def _extract_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...}, handling nested braces."""
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return None
    try:
        brace_start = text.index("{", idx)
    except ValueError:
        return None
    depth = 0
    for i, ch in enumerate(text[brace_start:], brace_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1: i].strip()
    return None


def extract_gt_answer(text: str, dataset: str) -> Optional[str]:
    if dataset == "gsm8k":
        m = _GT_HASH_RE.search(text)
        return m.group(1).strip() if m else None
    if dataset == "math":
        return _extract_boxed(text)
    if dataset == "mbpp":
        return text
    if dataset == "strategyqa":
        m = _YESNO_HASH_RE.search(text)
        return m.group(1).lower() if m else None
    raise ValueError(f"No GT extractor for dataset {dataset!r}")


def _aime_extract_int(text: str) -> Optional[int]:
    """AIME prediction -> integer in [0, 999]: \\boxed{N}, else last number near the end."""
    if not text:
        return None
    candidates = list(_BOXED_RE.findall(text)) or _LAST_NUMBER_RE.findall(text[-200:])
    for raw in reversed(candidates):
        s = raw.strip().replace(",", "").replace("$", "").replace("\\,", "")
        m = re.search(r"-?\d+", s)
        if not m:
            continue
        n = int(m.group(0))
        if 0 <= n <= 999:
            return n
    return None


def extract_pred_answer(text: str, dataset: str) -> Optional[str]:
    if dataset in ("aime24", "aime25"):
        n = _aime_extract_int(text)
        return str(n) if n is not None else None
    if dataset == "mbpp":
        m = _CODE_FENCE_RE.search(text)
        if m:
            return m.group(1).strip()
        return text.strip() if "def " in text else None
    if dataset == "strategyqa":
        m = _YESNO_HASH_RE.search(text)
        if m:
            return m.group(1).lower()
        m = _YESNO_FALLBACK_RE.findall(text[-200:])
        return m[-1].lower() if m else None
    if dataset == "gsm8k":
        # `#### N` first; then \boxed{} (thinking teachers often use LaTeX); then last number.
        matches = _HASH_LINE_RE.findall(text)
        if matches:
            return matches[-1].strip()
        matches = _BOXED_RE.findall(text)
        if matches:
            return matches[-1].strip()
        nums = _LAST_NUMBER_RE.findall(text[-300:])
        if nums:
            return nums[-1].replace(",", "").replace("$", "").strip()
        return None
    if dataset == "math":
        return _extract_boxed(text)
    raise ValueError(f"No prediction extractor for dataset {dataset!r}")


def normalize_answer(s: Optional[str], dataset: str) -> Optional[str]:
    if s is None:
        return None
    s = re.sub(r"\s+", " ", s.strip())
    s = (s.replace("，", ",").replace("。", ".").replace("（", "(")
          .replace("）", ")").replace("：", ":"))
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # 1,234 -> 1234
    s = re.sub(r"\$\s+", "$", s)
    return s


def _mbpp_run_tests(code: str, test_list: List[str], test_setup_code: str = "",
                    timeout: int = 8) -> bool:
    """Run predicted code + asserts in a subprocess; True iff all pass."""
    if not code:
        return False
    full = (test_setup_code + "\n" if test_setup_code else "") + code + "\n"
    full += "".join(t + "\n" for t in test_list)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _aime_math_verify(pred_text: str, gt_int: int) -> Optional[bool]:
    """LaTeX-aware equivalence via `math-verify` if installed; None if unavailable/unparseable."""
    try:
        from math_verify import parse, verify  # type: ignore
        pred_parsed = parse(pred_text)
        gold_parsed = parse(f"${gt_int}$")
        if not pred_parsed or not gold_parsed:
            return None
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        return None


def is_correct(pred_text: Optional[str], example: Dict[str, Any], dataset: str) -> bool:
    """Unified correctness check; `example` is the loaded split entry."""
    if pred_text is None:
        return False
    if dataset in ("aime24", "aime25"):
        gn = example.get("int_answer")
        if gn is None:
            return False
        mv = _aime_math_verify(pred_text, int(gn))
        if mv is not None:
            return mv
        pn = _aime_extract_int(pred_text)
        return pn is not None and pn == int(gn)
    if dataset == "mbpp":
        code = extract_pred_answer(pred_text, "mbpp")
        return bool(code) and _mbpp_run_tests(code, example.get("test_list", []),
                                              example.get("test_setup_code", ""))
    if dataset == "strategyqa":
        pa = extract_pred_answer(pred_text, "strategyqa")
        ga = example.get("bool_answer")
        return pa is not None and ga is not None and pa == ga
    if dataset in ("gsm8k", "math"):
        ga = normalize_answer(extract_gt_answer(example["answer"], dataset), dataset)
        pa = normalize_answer(extract_pred_answer(pred_text, dataset), dataset)
        return ga is not None and pa is not None and ga == pa
    raise ValueError(f"No verifier for dataset {dataset!r}")
