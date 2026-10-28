"""Steering vectors as a training signal in LLM post-training (NeurIPS 2026).

Core API::

    from sv_training import Injection, SteeringContext, attach, load_vector

    v = load_vector("my_vector.pt")                       # [D] or [N, D]
    ctx = SteeringContext()
    handles = attach(model, [Injection(v, layer=15, alpha=0.5, op="add")], ctx)
    with ctx.batch(labels=labels, sample_idx=sample_idx):
        loss = model(input_ids=input_ids, labels=labels).loss

or train end-to-end with :func:`sv_training.train.train` / ``sv-train train``.
"""
from .steering import (Injection, SteeringContext, attach, detach, find_transformer_blocks,
                       slerp_rotate)
from .vectors import load_vector, mean_vector_from_acts, save_vector

__version__ = "0.1.0"

__all__ = [
    "Injection", "SteeringContext", "attach", "detach", "find_transformer_blocks",
    "slerp_rotate", "load_vector", "mean_vector_from_acts", "save_vector", "__version__",
]
