"""Steering-vector injection into the residual stream of HF transformer blocks.

An :class:`Injection` says *what* to inject (a global ``[D]`` vector or a
per-sample ``[N, D]`` matrix), *where* (block index, answer tokens only,
optionally only the first K of them) and *how* (``add``: ``h + alpha*v`` or
``slerp``: rotate ``h`` toward ``v`` by fraction ``alpha``, preserving norm).

Usage with any HF model (no Trainer required)::

    ctx = SteeringContext()
    handles = attach(model, [Injection(vector=v, layer=15, alpha=0.5, op="add")], ctx)
    with ctx.batch(labels=labels, sample_idx=idx):
        loss = model(input_ids=..., labels=labels).loss
    detach(handles)

Layer indices are 0-based and exclude the embedding: layer ``l`` is the
output of transformer block ``l`` (``hidden_states[l + 1]`` in HF outputs).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

OPS = ("add", "slerp")


def slerp_rotate(h: torch.Tensor, v_dir: torch.Tensor, t) -> torch.Tensor:
    """Rotate ``h`` toward unit vector ``v_dir`` by fraction ``t`` along the great circle.

    ``h`` is ``[K, D]``; ``v_dir`` is ``[D]`` or ``[K, D]``; ``t`` is a scalar.
    Norms of ``h`` are preserved; output has the dtype of ``h``.
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

    coeff_h = torch.sin((1 - t) * theta) / sin_theta
    coeff_v = torch.sin(t * theta) / sin_theta
    rotated = coeff_h * h_hat + coeff_v * v_dir
    return (rotated * orig_norms).to(orig_dtype)


# ---------------------------------------------------------------------------
# Locating transformer blocks
# ---------------------------------------------------------------------------


def unwrap_model(m: nn.Module) -> nn.Module:
    """Strip DDP/accelerate (`.module`) and PEFT wrappers."""
    while hasattr(m, "module") and isinstance(m.module, nn.Module):
        m = m.module
    if hasattr(m, "get_base_model"):
        try:
            return m.get_base_model()
        except Exception:
            pass
    return m


def find_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    """Return the ModuleList of decoder blocks (Qwen2/3, Llama, Gemma, Mistral, GPT-2, ...)."""
    base = unwrap_model(model)
    num_layers = getattr(getattr(base, "config", None), "num_hidden_layers", None)
    for path in ("model.layers", "model.model.layers", "model.language_model.layers",
                 "transformer.h", "model.transformer.h", "gpt_neox.layers",
                 "model.decoder.layers", "decoder.layers", "layers", "h", "blocks"):
        cur = base
        for attr in path.split("."):
            cur = getattr(cur, attr, None)
            if cur is None:
                break
        if isinstance(cur, nn.ModuleList) and (num_layers is None or len(cur) == num_layers):
            return cur
    for _, mod in base.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == num_layers:
            return mod
    raise RuntimeError("Could not locate the transformer block list of this model.")


# ---------------------------------------------------------------------------
# Injection spec + per-batch context
# ---------------------------------------------------------------------------


@dataclass
class Injection:
    """One steering intervention at one layer.

    vector:      ``[D]`` (same vector for every sample) or ``[N, D]`` (row ``i`` is
                 used for the training example with dataset index ``i``).
    layer:       0-based transformer block index whose output is modified.
    alpha:       additive scale (``add``) or Slerp fraction in [0, 1] (``slerp``).
    op:          ``"add"`` or ``"slerp"``. Slerp needs unit vectors: ``normalize``
                 is forced on for it.
    normalize:   unit-normalise each vector before injecting.
    first_k:     if set, steer only the first K answer tokens.
    skip_first_k: leave the first K answer tokens unsteered (still in the loss).
    answer_only: steer only positions whose label != -100 (paper setting). If
                 False, every position is steered and labels are not required.
    """

    vector: torch.Tensor
    layer: int
    alpha: float = 0.5
    op: str = "add"
    normalize: bool = False
    first_k: Optional[int] = None
    skip_first_k: int = 0
    answer_only: bool = True
    name: str = ""
    _table: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        if self.op not in OPS:
            raise ValueError(f"op must be one of {OPS}, got {self.op!r}")
        v = torch.as_tensor(self.vector).detach().to(torch.float32).cpu()
        if v.dim() not in (1, 2):
            raise ValueError(f"steering vector must be [D] or [N, D], got shape {tuple(v.shape)}")
        if self.op == "slerp" or self.normalize:
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self._table = v

    @property
    def per_sample(self) -> bool:
        return self._table.dim() == 2

    @property
    def hidden_size(self) -> int:
        return self._table.shape[-1]

    def describe(self) -> str:
        kind = f"per-sample[{self._table.shape[0]}]" if self.per_sample else "global"
        scope = "answer tokens" if self.answer_only else "all tokens"
        if self.first_k:
            scope += f", first {self.first_k}"
        if self.skip_first_k:
            scope += f", skip first {self.skip_first_k}"
        norm = self._table.norm(dim=-1).mean().item()
        return (f"{self.name or 'sv'}: layer={self.layer} op={self.op} alpha={self.alpha} "
                f"{kind} ||v||={norm:.4f} ({scope})")


class SteeringContext:
    """Holds the labels / sample indices of the batch currently being forwarded."""

    def __init__(self):
        self.labels: Optional[torch.Tensor] = None
        self.sample_idx: Optional[torch.Tensor] = None
        self.enabled = True

    @contextlib.contextmanager
    def batch(self, labels=None, sample_idx=None):
        self.labels, self.sample_idx = labels, sample_idx
        try:
            yield self
        finally:
            self.labels, self.sample_idx = None, None


def _positions(mask_row: torch.Tensor, inj: Injection) -> torch.Tensor:
    pos = mask_row.nonzero(as_tuple=True)[0]
    sk = inj.skip_first_k
    if sk > 0 and pos.numel() > sk:
        pos = pos[sk:]
    if inj.first_k is not None:
        pos = pos[: inj.first_k]
    return pos


def make_hook(inj: Injection, ctx: SteeringContext, training_only: bool = True):
    """Forward hook implementing ``inj`` on a decoder block's output."""

    def hook(module, inputs, output):
        if not ctx.enabled or (training_only and not module.training):
            return output
        h = output[0] if isinstance(output, tuple) else output  # [B, T, D]
        B, T, _ = h.shape
        if inj.answer_only:
            if ctx.labels is None:
                return output
            mask = ctx.labels != -100
        else:
            mask = torch.ones(B, T, dtype=torch.bool, device=h.device)

        h_new = None
        for b in range(B):
            if inj.per_sample:
                if ctx.sample_idx is None:
                    return output
                idx = int(ctx.sample_idx[b])
                if idx < 0 or idx >= inj._table.shape[0]:
                    continue
                v = inj._table[idx]
            else:
                v = inj._table
            v = v.to(device=h.device, dtype=h.dtype)
            if v.norm() < 1e-6:  # dropped / zero rows are a no-op
                continue
            pos = _positions(mask[b], inj)
            if pos.numel() == 0:
                continue
            if h_new is None:
                h_new = h.clone()
            if inj.op == "add":
                h_new[b, pos, :] = h_new[b, pos, :] + inj.alpha * v
            else:
                h_new[b, pos, :] = slerp_rotate(h_new[b, pos, :], v, float(inj.alpha))

        if h_new is None:
            return output
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        return h_new

    return hook


def attach(model: nn.Module, injections: Sequence[Injection], ctx: SteeringContext,
           training_only: bool = True) -> List[torch.utils.hooks.RemovableHandle]:
    """Register one forward hook per injection; returns the hook handles."""
    blocks = find_transformer_blocks(model)
    hidden = getattr(unwrap_model(model).config, "hidden_size", None)
    handles = []
    for inj in injections:
        if not 0 <= inj.layer < len(blocks):
            raise ValueError(f"layer {inj.layer} out of range: model has {len(blocks)} blocks")
        if hidden is not None and inj.hidden_size != hidden:
            raise ValueError(f"vector dim {inj.hidden_size} != model hidden_size {hidden} "
                             f"({inj.describe()})")
        handles.append(blocks[inj.layer].register_forward_hook(make_hook(inj, ctx, training_only)))
    return handles


def detach(handles: Iterable[torch.utils.hooks.RemovableHandle]) -> None:
    for h in handles:
        h.remove()
