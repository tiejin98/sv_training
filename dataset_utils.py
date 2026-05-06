"""Dataset adapters for the main paper experiments.

Extends exploration_code/dataset_utils.py to support MBPP and StrategyQA in
addition to GSM8K and MATH. Also standardises:

- load_main_split(name, split, cache_dir)  → list[dict]
- main_get_instruction(name)               → system instruction string
- main_build_messages(question, name)      → chat messages list
- main_extract_gt_answer(text, name)       → canonical GT final answer
- main_extract_pred_answer(text, name)     → canonical model-output answer
- main_is_correct(pred_text, gt_record, name) → bool
"""
from __future__ import annotations
import os, re, sys, subprocess, tempfile, signal, json
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "exploration_code"))
from _dataset_utils_base import (  # type: ignore
    load_split as _old_load_split,
    get_instruction as _old_get_instruction,
    build_messages as _old_build_messages,
    extract_gt_final_answer as _old_extract_gt,
    extract_pred_final_answer as _old_extract_pred,
    normalize_answer as _old_normalize,
    answers_match as _old_answers_match,
)

from datasets import load_dataset

SUPPORTED = ("gsm8k", "math", "mbpp", "strategyqa", "aime24", "aime25")

# ---------------------------------------------------------------------------
# Instructions / prompts
# ---------------------------------------------------------------------------

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


def main_get_instruction(dataset_name: str) -> str:
    if dataset_name == "mbpp":
        return _MBPP_INSTRUCTION
    if dataset_name == "strategyqa":
        return _STRATEGYQA_INSTRUCTION
    if dataset_name in ("aime24", "aime25"):
        return _AIME_INSTRUCTION
    return _old_get_instruction(dataset_name)


def main_build_messages(question: str, dataset_name: str) -> List[Dict[str, str]]:
    if dataset_name == "mbpp":
        # MBPP problem text typically ends with test cases; include them in the prompt
        instr = _MBPP_INSTRUCTION
        return [{"role": "user", "content": instr + "\nProblem:\n" + question}]
    if dataset_name == "strategyqa":
        instr = _STRATEGYQA_INSTRUCTION
        return [{"role": "user", "content": instr + "\nQuestion:\n" + question}]
    if dataset_name in ("aime24", "aime25"):
        return [{"role": "user", "content": _AIME_INSTRUCTION + "\n\nProblem:\n" + question}]
    return _old_build_messages(question, dataset_name)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_mbpp(split: str, cache_dir: str) -> List[Dict[str, Any]]:
    # google-research-datasets/mbpp (full): 374 train, 500 test, 90 validation, 10 prompt
    # MBPP fields: task_id, text, code, test_list, test_setup_code
    ds = load_dataset("google-research-datasets/mbpp", "full", cache_dir=cache_dir)
    # Their split naming: train / test / prompt / validation
    # For paper: use train(374) and test(500) — "prompt" and "validation" are optional
    if split == "train":
        data = ds["train"]
    elif split == "test":
        data = ds["test"]
    else:
        raise ValueError(f"MBPP: unsupported split {split!r}")

    examples = []
    for ex in data:
        # Problem text: "text" field (full config). Prepend to test cases.
        tests = ex.get("test_list", []) or []
        problem_text = ex.get("text") or ex.get("prompt", "")
        if tests:
            problem_text = problem_text.rstrip() + "\nYour code should pass these tests:\n" + "\n".join(tests)
        examples.append({
            "question": problem_text,
            "answer":   ex.get("code", ""),                   # GT code
            "task_id":  ex["task_id"],
            "test_list": tests,
            "test_setup_code": ex.get("test_setup_code", ""),
            "dataset":  "mbpp",
            "has_solution": True,
        })
    return examples


def _load_strategyqa(split: str, cache_dir: str) -> List[Dict[str, Any]]:
    # ChilleD/StrategyQA (or voidful/StrategyQA): train ~2290, test ~490
    # Fields: question, answer (bool), facts (list[str]), decomposition
    _candidates = [
        ("ChilleD/StrategyQA", None),
        ("voidful/StrategyQA", None),
        ("wics/strategy-qa",   None),
    ]
    ds = None
    loaded = None
    for name, cfg in _candidates:
        try:
            ds = load_dataset(name, cache_dir=cache_dir) if cfg is None else load_dataset(name, cfg, cache_dir=cache_dir)
            loaded = name
            print(f"[dataset] Loaded StrategyQA from '{name}'")
            break
        except Exception as e:
            continue
    if ds is None:
        raise RuntimeError("Could not load StrategyQA from any of: " + ", ".join(n for n, _ in _candidates))

    # Resolve split name — the three datasets use different names
    if split == "train":
        data = ds.get("train") or list(ds.values())[0]
    elif split == "test":
        # Most StrategyQA distributions use "test" or "validation"
        data = ds.get("test") or ds.get("validation") or list(ds.values())[-1]
    else:
        raise ValueError(f"StrategyQA: unsupported split {split!r}")

    examples = []
    for ex in data:
        q = ex.get("question", "")
        ans = ex.get("answer", None)
        # Normalise bool answer to "yes"/"no" string
        if isinstance(ans, bool):
            ans_str = "yes" if ans else "no"
        else:
            ans_str = str(ans).strip().lower()
            if ans_str in ("true", "1"):
                ans_str = "yes"
            elif ans_str in ("false", "0"):
                ans_str = "no"
        # Reasoning chain if available — ChilleD/StrategyQA stores `facts`
        # as a single prose string; some other variants use a list.
        facts_raw = ex.get("facts")
        reasoning = ""
        if isinstance(facts_raw, str):
            reasoning = facts_raw.strip()
        elif isinstance(facts_raw, list) and facts_raw:
            reasoning = "\n".join(f"- {f}" for f in facts_raw if f)
        if reasoning:
            gt_full = f"{reasoning}\n#### {ans_str}"
        else:
            gt_full = f"#### {ans_str}"
        examples.append({
            "question": q,
            "answer":   gt_full,
            "bool_answer": ans_str,
            "dataset":  "strategyqa",
            "has_solution": bool(reasoning),
        })
    return examples


def _load_aime24(cache_dir: str) -> List[Dict[str, Any]]:
    ds = load_dataset("Maxwell-Jia/AIME_2024", cache_dir=cache_dir)
    data = ds["train"]
    examples = []
    for ex in data:
        examples.append({
            "question": ex["Problem"],
            "answer": str(int(ex["Answer"])),
            "int_answer": int(ex["Answer"]),
            "id": ex.get("ID", ""),
            "dataset": "aime24",
            "has_solution": True,
        })
    print(f"[dataset] Loaded AIME-2024: {len(examples)} problems")
    return examples


def _load_aime25(cache_dir: str) -> List[Dict[str, Any]]:
    examples = []
    for cfg in ("AIME2025-I", "AIME2025-II"):
        ds = load_dataset("opencompass/AIME2025", cfg, cache_dir=cache_dir)
        s = list(ds.keys())[0]
        for ex in ds[s]:
            ans_raw = str(ex["answer"]).strip()
            m = re.match(r"-?\d+", ans_raw)
            if m is None:
                print(f"[aime25] WARN: skip non-int answer {ans_raw!r}")
                continue
            n = int(m.group(0))
            examples.append({
                "question": ex["question"],
                "answer": str(n),
                "int_answer": n,
                "id": cfg,
                "dataset": "aime25",
                "has_solution": True,
            })
    print(f"[dataset] Loaded AIME-2025 (I+II): {len(examples)} problems")
    return examples


def load_main_split(dataset_name: str, split: str, cache_dir: str,
                    limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if dataset_name == "mbpp":
        ex = _load_mbpp(split, cache_dir)
    elif dataset_name == "strategyqa":
        ex = _load_strategyqa(split, cache_dir)
    elif dataset_name == "aime24":
        ex = _load_aime24(cache_dir)
    elif dataset_name == "aime25":
        ex = _load_aime25(cache_dir)
    else:
        return _old_load_split(dataset_name, split, cache_dir, limit)
    if limit:
        ex = ex[:limit]
    return ex


# ---------------------------------------------------------------------------
# Answer extraction & correctness
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_YESNO_HASH_RE = re.compile(r"####\s*(yes|no)\b", re.IGNORECASE)
_YESNO_FALLBACK_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def main_extract_gt_answer(text: str, dataset_name: str) -> Optional[str]:
    if dataset_name == "mbpp":
        # For MBPP the "GT answer" is the code itself; we don't compare strings.
        # Return the code (used during training supervision only).
        return text
    if dataset_name == "strategyqa":
        m = _YESNO_HASH_RE.search(text)
        return m.group(1).lower() if m else None
    return _old_extract_gt(text, dataset_name)


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_LAST_NUMBER_RE = re.compile(r"(-?\$?\d[\d,]*(?:\.\d+)?)")


def _aime_extract_int(text: str) -> Optional[int]:
    """AIME pred → integer 0-999. Try \\boxed{N}, then last number near tail."""
    if not text:
        return None
    matches = _BOXED_RE.findall(text)
    candidates = list(matches)
    if not candidates:
        # Fallback: look at the very end of the text for a final number
        tail = text[-200:]
        candidates = _LAST_NUMBER_RE.findall(tail)
    for raw in reversed(candidates):
        s = raw.strip().replace(",", "").replace("$", "").replace("\\,", "")
        # Strip prefixes like "0", leading zeros are fine for AIME (000-999)
        m = re.search(r"-?\d+", s)
        if not m:
            continue
        try:
            n = int(m.group(0))
            if 0 <= n <= 999:
                return n
        except ValueError:
            continue
    return None


def main_extract_pred_answer(text: str, dataset_name: str) -> Optional[str]:
    if dataset_name in ("aime24", "aime25"):
        n = _aime_extract_int(text)
        return str(n) if n is not None else None
    if dataset_name == "mbpp":
        # Extract the first fenced python block
        m = _CODE_FENCE_RE.search(text)
        if m:
            return m.group(1).strip()
        # Fallback: take everything after last '```python' or the whole text
        if "def " in text:
            return text.strip()
        return None
    if dataset_name == "strategyqa":
        m = _YESNO_HASH_RE.search(text)
        if m:
            return m.group(1).lower()
        # Fallback: last yes/no mention in the last 80 chars
        tail = text[-200:]
        m = _YESNO_FALLBACK_RE.findall(tail)
        return m[-1].lower() if m else None
    if dataset_name == "gsm8k":
        # Try #### first (standard GSM8K), fall back to \boxed{} (teacher models
        # like Math-7B and Qwen3-8B often use LaTeX), fall back to last number.
        out = _old_extract_pred(text, dataset_name)
        if out is not None:
            return out
        matches = _BOXED_RE.findall(text)
        if matches:
            return matches[-1].strip()
        # Last-resort: final number in text (useful when teacher just says "the answer is N")
        tail = text[-300:]
        nums = _LAST_NUMBER_RE.findall(tail)
        if nums:
            return nums[-1].replace(",", "").replace("$", "").strip()
        return None
    return _old_extract_pred(text, dataset_name)


def _mbpp_run_tests(pred_code: str, test_list: List[str], test_setup_code: str = "",
                     timeout: int = 8) -> bool:
    """Run pred code + tests in a subprocess, return True if all pass."""
    if not pred_code:
        return False
    full = (test_setup_code + "\n" if test_setup_code else "") + pred_code + "\n"
    for test in test_list:
        full += test + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full)
        path = f.name
    try:
        r = subprocess.run(
            [sys.executable, path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try: os.remove(path)
        except OSError: pass


def _aime_is_correct_via_math_verify(pred_text: str, gt_int: int) -> Optional[bool]:
    """Use math-verify (same as rl_experiments/math_reward.py) for LaTeX equivalence.
    Returns True/False if math-verify could parse, None on parse failure."""
    try:
        from math_verify import parse, verify  # type: ignore
        pred_parsed = parse(pred_text)
        gold_parsed = parse(f"${gt_int}$")
        if not pred_parsed or not gold_parsed:
            return None
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        return None


def main_is_correct(pred_text: str, gt_record: Dict[str, Any], dataset_name: str) -> bool:
    """Unified correctness check. `gt_record` is the loaded split entry."""
    if pred_text is None:
        return False
    if dataset_name in ("aime24", "aime25"):
        gn = gt_record.get("int_answer")
        if gn is None:
            return False
        # Primary: math-verify on the full pred text (handles \boxed{}, equations, units)
        mv = _aime_is_correct_via_math_verify(pred_text, int(gn))
        if mv is not None:
            return mv
        # Fallback: regex extract last integer in [0,999] from \boxed{} or text tail
        pn = _aime_extract_int(pred_text)
        return pn is not None and pn == int(gn)
    if dataset_name == "mbpp":
        code = main_extract_pred_answer(pred_text, "mbpp")
        if not code:
            return False
        return _mbpp_run_tests(code, gt_record.get("test_list", []),
                                gt_record.get("test_setup_code", ""))
    if dataset_name == "strategyqa":
        pa = main_extract_pred_answer(pred_text, "strategyqa")
        ga = gt_record.get("bool_answer")
        return pa is not None and ga is not None and pa == ga
    # GSM8K / MATH — use main_extract_pred_answer (handles both #### and \\boxed{}).
    ga = _old_normalize(_old_extract_gt(gt_record["answer"], dataset_name), dataset_name)
    pa = _old_normalize(main_extract_pred_answer(pred_text, dataset_name), dataset_name)
    return ga is not None and pa is not None and ga == pa


# Re-export normalise + GT/pred helpers so callers only import from this module
normalize_answer = _old_normalize
