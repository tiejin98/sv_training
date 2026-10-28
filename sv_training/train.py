"""LoRA SFT / distillation with steering vectors injected during training."""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .steering import Injection, SteeringContext, attach, detach
from .vectors import load_vector, random_vectors

# Paper presets (Section 4): how each --method builds its vectors from an acts cache.
#   kind:      "mean" -> one dataset-level vector; "per_sample" -> one row per example
#   normalize: unit-normalise before injecting; op: injection primitive
PRESETS = {
    "mean_sv_add": dict(kind="mean", normalize=False, op="add", alpha=0.5),
    "slerp": dict(kind="per_sample", normalize=True, op="slerp", alpha=0.1),
    "per_inst_add": dict(kind="per_sample", normalize=True, op="add", alpha=0.1),
    "sv_first_k": dict(kind="per_sample", normalize=True, op="slerp", alpha=0.1),
    "sv_multi_layer": dict(kind="per_sample", normalize=True, op="slerp", alpha=0.1),
    "random_gauss": dict(kind="random", normalize=True, op="slerp", alpha=0.1),
}


@dataclass
class InjectSpec:
    """A user-facing injection description (resolved to :class:`Injection` once the model is known)."""
    path: Optional[str]
    layer: Optional[int] = None
    alpha: float = 0.5
    op: str = "add"
    kind: str = "auto"
    normalize: bool = False
    first_k: Optional[int] = None
    skip_first_k: int = 0
    answer_only: bool = True

    @classmethod
    def parse(cls, text: str, defaults: "InjectSpec") -> "InjectSpec":
        """Parse ``path=v.pt,layer=15,alpha=0.5,op=add,kind=mean,normalize=1,first_k=32``.

        A bare path is also accepted. Missing keys come from ``defaults``.
        """
        spec = InjectSpec(**{k: getattr(defaults, k) for k in defaults.__dataclass_fields__})
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                spec.path = part
                continue
            k, v = (s.strip() for s in part.split("=", 1))
            if k in ("layer", "first_k", "skip_first_k"):
                setattr(spec, k, int(v))
            elif k == "alpha":
                spec.alpha = float(v)
            elif k in ("normalize", "answer_only"):
                setattr(spec, k, v.lower() in ("1", "true", "yes"))
            elif k in ("path", "op", "kind"):
                setattr(spec, k, v)
            else:
                raise ValueError(f"unknown key {k!r} in --inject {text!r}")
        return spec

    def resolve(self, n_samples: int, hidden_size: int, seed: int = 0) -> Injection:
        if self.layer is None:
            raise ValueError(f"no layer given for {self.path}")
        if self.kind == "random":
            vec = random_vectors(n_samples, hidden_size, seed=seed)
            name = f"random(seed={seed})"
        else:
            vec = load_vector(self.path, layer=self.layer, kind=self.kind)
            name = os.path.basename(self.path)
        return Injection(vector=vec, layer=self.layer, alpha=self.alpha, op=self.op,
                         normalize=self.normalize, first_k=self.first_k,
                         skip_first_k=self.skip_first_k, answer_only=self.answer_only, name=name)


@dataclass
class TrainConfig:
    model: str
    dataset: str
    output_dir: str
    injections: List[InjectSpec] = field(default_factory=list)
    teacher_json: Optional[str] = None
    keep_only_correct: bool = True
    append_eos: bool = False
    enable_thinking: bool = False
    cache_dir: Optional[str] = None
    limit: Optional[int] = None
    # optimisation (paper defaults)
    lr: float = 5e-5
    epochs: float = 1.0
    batch_size: int = 1
    grad_accum: int = 8
    max_steps: int = -1
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    full_finetune: bool = False
    max_context_len: int = 2048
    seed: int = 42
    gradient_checkpointing: bool = False
    bf16: bool = True
    logging_steps: int = 20
    report_to: str = "none"
    run_name: Optional[str] = None


def default_layer(num_layers: int) -> int:
    """Paper placement rule l = L/2 + 1 (15 for 28-layer Qwen2.5)."""
    return num_layers // 2 + 1


def train(cfg: TrainConfig) -> str:
    """Run (SV-)SFT and save the adapter (or full model) to ``cfg.output_dir``."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    from .trainer import ListDataset, SFTCollator, SteeringTrainer, build_sft_records

    print(f"[train] loading {cfg.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.model, cache_dir=cfg.cache_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_bf16 = cfg.bf16 and torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, cache_dir=cfg.cache_dir, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    num_layers, hidden = model.config.num_hidden_layers, model.config.hidden_size
    print(f"[train] num_hidden_layers={num_layers} hidden_size={hidden}")

    if not cfg.full_finetune:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=cfg.lora_target.split(",")))
        model.print_trainable_parameters()

    records = build_sft_records(
        tok, cfg.dataset, "train", cache_dir=cfg.cache_dir, max_context_len=cfg.max_context_len,
        enable_thinking=cfg.enable_thinking, teacher_json=cfg.teacher_json,
        keep_only_correct=cfg.keep_only_correct, append_eos=cfg.append_eos, limit=cfg.limit)
    print(f"[train] {len(records)} training records")
    if not records:
        raise RuntimeError("no training records")

    ctx = SteeringContext()
    n_samples = max(r["sample_idx"] for r in records) + 1
    injections = []
    for spec in cfg.injections:
        if spec.layer is None:
            spec.layer = default_layer(num_layers)
        inj = spec.resolve(n_samples, hidden, seed=cfg.seed)
        if inj.per_sample and inj._table.shape[0] < n_samples:
            raise RuntimeError(f"{inj.name} has {inj._table.shape[0]} rows but dataset "
                               f"indices go up to {n_samples - 1}")
        injections.append(inj)
        print(f"[train] inject {inj.describe()}")
    handles = attach(model, injections, ctx)

    if cfg.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    work_dir = tempfile.mkdtemp(prefix="sv_train_")
    args = TrainingArguments(
        output_dir=work_dir,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        logging_steps=cfg.logging_steps,
        save_strategy="no",
        bf16=use_bf16,
        report_to=[] if cfg.report_to in ("", "none") else cfg.report_to.split(","),
        run_name=cfg.run_name or os.path.basename(os.path.normpath(cfg.output_dir)),
        seed=cfg.seed,
        remove_unused_columns=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = SteeringTrainer(
        model=model, args=args, train_dataset=ListDataset(records),
        data_collator=SFTCollator(tok.pad_token_id), processing_class=tok,
        steering_context=ctx)
    trainer.train()

    detach(handles)
    os.makedirs(cfg.output_dir, exist_ok=True)
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"[train] saved to {cfg.output_dir}", flush=True)
    return cfg.output_dir
