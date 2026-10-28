from sv_training.data import build_messages, extract_pred_answer, is_correct

GSM_GT = {"answer": "Natalia sold 48/2 = 24 clips in May.\n#### 72"}


def test_gsm8k_scoring():
    assert is_correct("so the total is 72.\n#### 72", GSM_GT, "gsm8k")
    assert is_correct("<think>...</think> The answer is \\boxed{72}", GSM_GT, "gsm8k")
    assert is_correct("altogether she sold 72", GSM_GT, "gsm8k")  # last-number fallback
    assert not is_correct("#### 71", GSM_GT, "gsm8k")
    assert extract_pred_answer("#### 1,234", "gsm8k") == "1,234"
    assert is_correct("#### 1,234", {"answer": "#### 1234"}, "gsm8k")


def test_math_scoring():
    ex = {"answer": "so we get $\\boxed{\\frac{1}{2}}$."}
    assert is_correct("Therefore the answer is $\\boxed{\\frac{1}{2}}$.", ex, "math")
    assert not is_correct("Therefore the answer is $\\boxed{2}$.", ex, "math")


def test_strategyqa_and_aime():
    assert is_correct("reasoning...\n#### yes", {"bool_answer": "yes"}, "strategyqa")
    assert not is_correct("#### no", {"bool_answer": "yes"}, "strategyqa")
    assert extract_pred_answer("so \\boxed{073}", "aime24") == "73"


def test_mbpp_runs_tests():
    ex = {"test_list": ["assert add(2, 3) == 5"], "test_setup_code": ""}
    assert is_correct("```python\ndef add(a, b):\n    return a + b\n```", ex, "mbpp")
    assert not is_correct("```python\ndef add(a, b):\n    return a - b\n```", ex, "mbpp")


def test_prompts_are_unchanged():
    msg = build_messages("What is 1+1?", "gsm8k")[0]["content"]
    assert msg.startswith("You are a helpful math assistant.\n")
    assert msg.endswith("\nProblem:\nWhat is 1+1?")
    assert build_messages("Hi", "custom.jsonl") == [{"role": "user", "content": "Hi"}]
