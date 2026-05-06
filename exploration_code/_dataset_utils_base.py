"""
Unified dataset utilities for GSM8K, MATH (lighteval/MATH), and AIME 2024.

Supported dataset names:
  "gsm8k"    - openai/gsm8k (train + test, full solutions)
  "math"     - lighteval/MATH (train + test, full solutions)
  "aime2024" - AI-MO/aimo-validation-aime (eval only, integer answers)
  "aime_hist"- Maxwell-Jia/AIME_1983_2024 (train + eval, has solutions)

Every example is normalised into:
  {"question": str, "answer": str, "dataset": str, ...extra metadata...}

For training / steering-vector computation you need full solution text in
"answer".  For evaluation-only tasks (aime2024) the "answer" field contains
just the final integer string.  Scripts that need a solution body (compute_sv,
train) should check for `ex.get("has_solution", True)`.
"""

import re
from typing import Optional, List, Dict, Any

from datasets import load_dataset


# ──────────────────────────────────────────────────────────────
# 1.  Per-dataset system instructions
# ──────────────────────────────────────────────────────────────

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

_AIME_INSTRUCTION = (
    "You are an expert mathematician solving an AIME competition problem.\n"
    "The answer is an integer between 0 and 999 (inclusive).\n"
    "Solve the problem step by step.\n"
    "You MUST end your response with a final line in the EXACT format:\n"
    "#### {integer answer}\n"
    "The final line must be the last line. Do not add anything after it.\n"
)


def get_instruction(dataset_name: str) -> str:
    if dataset_name == "gsm8k":
        return _GSM8K_INSTRUCTION
    elif dataset_name == "math":
        return _MATH_INSTRUCTION
    elif dataset_name in ("aime2024", "aime_hist"):
        return _AIME_INSTRUCTION
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ──────────────────────────────────────────────────────────────
# 2.  Message builders (inference & training)
# ──────────────────────────────────────────────────────────────

def build_messages(question: str, dataset_name: str) -> List[Dict[str, str]]:
    """Single-turn inference message list."""
    instruction = get_instruction(dataset_name)
    return [{"role": "user", "content": instruction + "\nProblem:\n" + question}]


def build_train_messages(
    question: str,
    answer: str,
    dataset_name: str,
    pred_prefix: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Training message pair.

    If pred_prefix is given (pred-as-prefix mode to address input-prefix gap),
    the user turn contains the original question and the assistant turn contains
    pred_prefix + a separator + the GT answer.  The caller is responsible for
    masking pred_prefix tokens with -100 in the labels.
    """
    instruction = get_instruction(dataset_name)
    user_content = instruction + "\nProblem:\n" + question

    if pred_prefix:
        # assistant sees its own previous wrong attempt, then GT
        # The training script should label pred_prefix tokens as -100
        assistant_content = pred_prefix + "\n\n[Correction]\n\n" + answer
    else:
        assistant_content = answer

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


# ──────────────────────────────────────────────────────────────
# 3.  Answer extraction
# ──────────────────────────────────────────────────────────────

_HASH_LINE_RE = re.compile(r"^####\s*(.+?)\s*$", re.MULTILINE)
_GT_HASH_RE   = re.compile(r"####\s*(.*?)(?:\n|$)", re.DOTALL)


def _extract_boxed(text: str) -> Optional[str]:
    """Return content of the last \\boxed{...}, handling nested braces."""
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
                return text[brace_start + 1 : i].strip()
    return None


def extract_gt_final_answer(answer_text: str, dataset_name: str) -> Optional[str]:
    """Extract the final answer token from a ground-truth solution string."""
    if dataset_name == "gsm8k":
        m = _GT_HASH_RE.search(answer_text)
        return m.group(1).strip() if m else None
    elif dataset_name == "math":
        return _extract_boxed(answer_text)
    elif dataset_name in ("aime2024", "aime_hist"):
        text = answer_text.strip()
        # Raw integer string (as stored in dataset)
        if re.match(r"^\d+$", text):
            return text
        # Generated format
        m = _HASH_LINE_RE.search(text)
        if m:
            return m.group(1).strip()
        return text
    raise ValueError(f"Unknown dataset: {dataset_name}")


def extract_pred_final_answer(gen_text: str, dataset_name: str) -> Optional[str]:
    """Extract the model's predicted final answer from its generated text."""
    if dataset_name == "gsm8k":
        matches = _HASH_LINE_RE.findall(gen_text)
        return matches[-1].strip() if matches else None
    elif dataset_name == "math":
        return _extract_boxed(gen_text)
    elif dataset_name in ("aime2024", "aime_hist"):
        matches = _HASH_LINE_RE.findall(gen_text)
        return matches[-1].strip() if matches else None
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ──────────────────────────────────────────────────────────────
# 4.  Normalisation & correctness
# ──────────────────────────────────────────────────────────────

def normalize_answer(s: Optional[str], dataset_name: str) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = (
        s.replace("，", ",")
         .replace("。", ".")
         .replace("（", "(")
         .replace("）", ")")
         .replace("：", ":")
    )
    # Remove thousand-separator commas: 1,234 -> 1234
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"\$\s+", "$", s)
    if dataset_name in ("aime2024", "aime_hist"):
        # Strip leading zeros; "007" → "7"
        s = s.lstrip("0") or "0"
    return s


def answers_match(gt_text: str, pred_text: Optional[str], dataset_name: str) -> bool:
    """Return True if extracted+normalised answers match."""
    gt_ans  = normalize_answer(extract_gt_final_answer(gt_text, dataset_name), dataset_name)
    pred_ans = normalize_answer(pred_text, dataset_name)
    if gt_ans is None or pred_ans is None:
        return False
    return gt_ans == pred_ans


# ──────────────────────────────────────────────────────────────
# 5.  Dataset loading
# ──────────────────────────────────────────────────────────────

def load_split(
    dataset_name: str,
    split: str,
    cache_dir: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load a dataset split and return a list of unified example dicts:

      {
        "question"     : str,   # problem / question text
        "answer"       : str,   # full solution or final-answer string
        "dataset"      : str,   # dataset_name tag
        "has_solution" : bool,  # True when "answer" contains a full solution
        ...                     # optional metadata (level, type, year, …)
      }
    """
    examples: List[Dict[str, Any]] = []

    if dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", cache_dir=cache_dir)
        data = ds[split]
        for ex in data:
            examples.append({
                "question": ex["question"],
                "answer":   ex["answer"],
                "dataset":  "gsm8k",
                "has_solution": True,
            })

    elif dataset_name == "math":
        # Try multiple known HuggingFace IDs — the canonical name has changed over time
        # nlile/hendrycks-MATH-benchmark: train=12000, test=500; fields: problem, solution, answer, subject, level
        # hendrycks/competition_math: train=7500, test=5000; fields: problem, solution, level, type
        _math_candidates = [
            ("nlile/hendrycks-MATH-benchmark", None),
            ("hendrycks/competition_math",     None),
            ("lighteval/MATH",                 "all"),
            ("EleutherAI/hendrycks_math",      "all"),
        ]
        ds = None
        _loaded_hf_name = None
        for _hf_name, _cfg in _math_candidates:
            try:
                ds = load_dataset(_hf_name, _cfg, cache_dir=cache_dir) if _cfg else \
                     load_dataset(_hf_name, cache_dir=cache_dir)
                _loaded_hf_name = _hf_name
                print(f"[dataset_utils] Loaded MATH from '{_hf_name}'")
                break
            except Exception:
                continue
        if ds is None:
            raise RuntimeError(
                "Could not load MATH dataset. Tried: "
                + ", ".join(n for n, _ in _math_candidates)
            )
        data = ds[split]
        for ex in data:
            question = ex.get("problem") or ex.get("question") or ""
            # solution field contains full LaTeX solution (with \boxed{}); prefer it over bare answer
            answer   = ex.get("solution") or ex.get("answer") or ""
            # subject field used by nlile/hendrycks-MATH-benchmark; type by hendrycks/competition_math
            subject  = ex.get("subject") or ex.get("type") or ""
            examples.append({
                "question": question,
                "answer":   answer,
                "level":    ex.get("level", ""),
                "type":     subject,
                "dataset":  "math",
                "has_solution": True,
            })

    elif dataset_name == "aime2024":
        # AI-MO/aimo-validation-aime – eval-only, no solution text.
        # split argument is ignored (only one canonical split exists).
        ds = load_dataset("AI-MO/aimo-validation-aime", cache_dir=cache_dir)
        key = list(ds.keys())[0]
        data = ds[key]
        for ex in data:
            problem = ex.get("problem") or ex.get("Problem") or ""
            answer  = str(ex.get("answer") or ex.get("Answer") or "")
            examples.append({
                "question":     problem,
                "answer":       answer,
                "dataset":      "aime2024",
                "has_solution": False,   # only final integer, no solution body
            })

    elif dataset_name == "aime_hist":
        # Maxwell-Jia/AIME_1983_2024 – has Solution field; usable for training / SV.
        ds = load_dataset("Maxwell-Jia/AIME_1983_2024", cache_dir=cache_dir)
        # This dataset typically has only a "train" split containing all problems.
        key = list(ds.keys())[0]
        data = ds[key]
        for ex in data:
            problem  = ex.get("Problem") or ex.get("problem") or ""
            solution = ex.get("Solution") or ex.get("solution") or ""
            answer   = str(ex.get("Answer") or ex.get("answer") or "")
            # Use solution as the "answer" body if available
            body = solution if solution.strip() else f"#### {answer}"
            examples.append({
                "question":     problem,
                "answer":       body,
                "final_answer": answer,
                "year":         ex.get("Year", ""),
                "dataset":      "aime_hist",
                "has_solution": bool(solution.strip()),
            })

    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            "Choose from: gsm8k, math, aime2024, aime_hist"
        )

    if limit is not None:
        examples = examples[:limit]
    return examples


# ──────────────────────────────────────────────────────────────
# 6.  Tokenisation helper (shared across training scripts)
# ──────────────────────────────────────────────────────────────

def tokenize_example(
    tokenizer,
    question: str,
    answer: str,
    dataset_name: str,
    max_length: int,
    pred_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tokenise a (question, answer) pair with prompt-token masking.

    Labels:  prompt tokens → -100  |  answer tokens → token_id
    If pred_prefix is provided (pred-as-prefix mode):
      The pred_prefix tokens inside the assistant turn are also set to -100.
      Only the GT answer tokens after "[Correction]" are supervised.
    """
    full_messages = build_train_messages(question, answer, dataset_name, pred_prefix)
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )

    # Determine the prompt-only length (everything before assistant reply)
    prompt_messages = full_messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    prompt_len = len(prompt_ids)

    input_ids = full["input_ids"]
    labels = input_ids.copy()

    # Mask prompt tokens
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    # Mask pred_prefix tokens inside the assistant turn
    if pred_prefix:
        sep = "\n\n[Correction]\n\n"
        sep_tokens = tokenizer(sep, add_special_tokens=False)["input_ids"]
        # Find sep_tokens in input_ids after prompt_len
        search_region = input_ids[prompt_len:]
        sep_start_in_region = _find_subseq(search_region, sep_tokens)
        if sep_start_in_region is not None:
            correction_start = prompt_len + sep_start_in_region + len(sep_tokens)
            # Mask everything before the correction start
            for i in range(prompt_len, min(correction_start, len(labels))):
                labels[i] = -100

    full["labels"] = labels
    return full


def _find_subseq(seq: list, subseq: list) -> Optional[int]:
    """Return index of first occurrence of subseq in seq, or None."""
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i : i + m] == subseq:
            return i
    return None
