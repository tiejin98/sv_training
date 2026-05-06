"""Minimal steering helpers extracted from the project's training infrastructure.

Provides:
- slerp_rotate: spherical-linear interpolation toward a unit direction
- unwrap_ddp / unwrap_to_base_model / find_transformer_blocks: model unwrapping
- SteerAnswerOnlyTrainer: HF Trainer subclass that surfaces `labels` to forward hooks
  so per-sample steering can mask non-answer tokens
"""
from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn
from transformers import Trainer


# ─────────────────────────────────────────────────────────────────
# Slerp on the unit sphere — preserves ||h||
# ─────────────────────────────────────────────────────────────────
def slerp_rotate(h: torch.Tensor, v_dir: torch.Tensor, t) -> torch.Tensor:
    """Rotate `h` toward unit vector `v_dir` by fraction `t` along the great circle.

    Args:
        h:     [..., D] hidden state(s).
        v_dir: [D] unit vector (or [..., D] broadcast-compatible).
        t:     scalar or [..., 1] interpolation fraction in [0, 1].

    Returns:
        Tensor of the same dtype/shape as `h`, with norms preserved.
    """
    orig_dtype = h.dtype
    h = h.float()
    v_dir = v_dir.float()
    orig_norms = h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    h_hat = h / orig_norms

    if v_dir.dim() == 1:
        v_dir = v_dir.unsqueeze(0).expand_as(h_hat)

    cos_theta = (h_hat * v_dir).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta).clamp_min(1e-8)

    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device=h.device, dtype=h.dtype)
    if t.dim() == 0:
        t = t.view(1, 1).expand(h.shape[0], 1)

    theta_new = (1 - t) * theta
    coeff_h = torch.sin(theta_new) / sin_theta
    coeff_v = torch.sin(t * theta) / sin_theta
    rotated = coeff_h * h_hat + coeff_v * v_dir
    return (rotated * orig_norms).to(orig_dtype)


# ─────────────────────────────────────────────────────────────────
# Model unwrapping (DDP + PEFT + nested)
# ─────────────────────────────────────────────────────────────────
def unwrap_ddp(m: nn.Module) -> nn.Module:
    while hasattr(m, "module"):
        m = m.module
    return m


def unwrap_to_base_model(m: nn.Module) -> nn.Module:
    m = unwrap_ddp(m)
    if hasattr(m, "get_base_model"):
        try:
            return m.get_base_model()
        except Exception:
            pass
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        return m.base_model.model
    return m


def find_transformer_blocks(base: nn.Module, num_layers: int) -> nn.ModuleList:
    """Walk common HF model paths to find the transformer block list.

    Works for Qwen2/Qwen3, Llama, Gemma, etc.
    """
    candidate_paths = [
        "model.layers", "model.model.layers",
        "transformer.h", "model.transformer.h",
        "decoder.layers", "model.decoder.layers",
        "layers", "h", "blocks",
    ]
    for path in candidate_paths:
        cur = base
        ok = True
        for attr in path.split("."):
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok and isinstance(cur, nn.ModuleList) and len(cur) == num_layers:
            return cur
    for _, mod in base.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == num_layers:
            return mod
    raise RuntimeError("Could not locate transformer blocks (ModuleList).")


# ─────────────────────────────────────────────────────────────────
# Trainer subclass that exposes `labels` to forward hooks
# ─────────────────────────────────────────────────────────────────
class SteerAnswerOnlyTrainer(Trainer):
    """HF Trainer that stashes `labels` on the model so per-sample
    steering hooks can mask non-answer (label == -100) positions."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels", None)
        wrapper = model
        base_ref = unwrap_ddp(model)

        if torch.is_tensor(labels):
            wrapper._steer_labels = labels
            base_ref._steer_labels = labels
        else:
            wrapper._steer_labels = None
            base_ref._steer_labels = None

        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        wrapper._steer_labels = None
        base_ref._steer_labels = None

        return (loss, outputs) if return_outputs else loss
