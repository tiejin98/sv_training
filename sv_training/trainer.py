"""SFT records, collator and a HF Trainer that feeds labels / sample indices to steering hooks."""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import Trainer

from .data import apply_chat_template, build_messages, load_split
from .steering import SteeringContext


def load_teacher(path: str, n_examples: int) -> Dict[int, dict]:
    """Teacher / prediction JSON -> {dataset_idx: record}.

    Either aligned (one record per example, in order) or filtered (records
    carry an ``idx`` field; missing indices are skipped).
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if len(raw) == n_examples and all("idx" not in r or int(r["idx"]) == i for i, r in enumerate(raw)):
        return {i: r for i, r in enumerate(raw)}
    return {int(r["idx"]): r for r in raw}


def build_sft_records(tokenizer, dataset: str, split: str = "train",
                      cache_dir: Optional[str] = None, max_context_len: int = 2048,
                      enable_thinking: bool = False,
                      teacher_json: Optional[str] = None,
                      keep_only_correct: bool = True,
                      append_eos: bool = False,
                      limit: Optional[int] = None) -> List[Dict]:
    """Records ``{"input_ids", "labels", "sample_idx"}``: prompt tokens are masked to -100.

    With ``teacher_json`` the assistant turn is the teacher's ``pred`` (hard
    distillation); by default only records with ``pred_correct`` are kept.
    ``sample_idx`` is the dataset index, used to look up per-sample vectors.
    """
    examples = load_split(dataset, split, cache_dir=cache_dir, limit=limit)
    teacher = load_teacher(teacher_json, len(examples)) if teacher_json else None
    if teacher is not None:
        print(f"[sft] teacher: {len(teacher)} records from {teacher_json}")
    eos = tokenizer.eos_token if (append_eos and tokenizer.eos_token) else ""

    records, n_long, n_wrong, n_missing = [], 0, 0, 0
    for i, ex in enumerate(examples):
        if teacher is not None:
            t = teacher.get(i)
            if t is None:
                n_missing += 1
                continue
            if keep_only_correct and not t.get("pred_correct", False):
                n_wrong += 1
                continue
            answer = (t.get("pred") or t.get("answer") or "").strip()
            if not answer:
                continue
        else:
            answer = ex["answer"]
        if eos:
            answer = answer.rstrip() + eos
        msgs = build_messages(ex["question"], dataset)
        prompt_text = apply_chat_template(tokenizer, msgs, True, enable_thinking)
        full_text = apply_chat_template(
            tokenizer, msgs + [{"role": "assistant", "content": answer}], False, enable_thinking)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        if len(full_ids) <= len(prompt_ids):
            continue
        if len(full_ids) > max_context_len:
            n_long += 1
            continue
        records.append({
            "input_ids": full_ids,
            "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
            "sample_idx": i,
        })
    if n_long:
        print(f"[sft] skipped {n_long} records longer than max_context_len={max_context_len}")
    if teacher is not None:
        print(f"[sft] kept {len(records)} records (skipped {n_wrong} wrong-teacher, "
              f"{n_missing} without teacher record)")
    return records


class ListDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


class SFTCollator:
    """Right-pads a batch; padded positions get label -100 and attention 0."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        T = max(len(r["input_ids"]) for r in batch)
        ids = torch.full((len(batch), T), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), T), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), T), dtype=torch.long)
        for b, r in enumerate(batch):
            n = len(r["input_ids"])
            ids[b, :n] = torch.tensor(r["input_ids"])
            labels[b, :n] = torch.tensor(r["labels"])
            attn[b, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": attn,
                "sample_idx": torch.tensor([r["sample_idx"] for r in batch], dtype=torch.long)}


class SteeringTrainer(Trainer):
    """HF Trainer that exposes each batch's labels / sample indices to steering hooks."""

    def __init__(self, *args, steering_context: Optional[SteeringContext] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.steering_context = steering_context or SteeringContext()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_idx = inputs.pop("sample_idx", None)
        with self.steering_context.batch(labels=inputs.get("labels"), sample_idx=sample_idx):
            outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss
