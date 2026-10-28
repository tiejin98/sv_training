"""Loading, building and saving steering vectors.

Accepted files for :func:`load_vector`:

* ``.pt`` / ``.pth`` / ``.bin`` (``torch.save``), ``.npy``, ``.safetensors``
* a bare tensor ``[D]`` (global) or ``[N, D]`` (per-sample, row = dataset index)
* a dict with ``"direction"`` or ``"vector"`` -> that tensor
* a dict with ``"vectors"`` / ``"raw_mean_vectors"`` keyed by layer
  (``15`` or ``"block_15"``) -> the entry for the requested layer
* an acts cache ``{"gt_acts": [N, L, D], "pred_acts": [N, L, D]}``
  (from ``sv-train acts``) -> mean or per-sample ``gt - pred`` at the layer
* a dict with ``"delta"`` ``[N, D]`` or ``[N, L, D]``
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch


def _load_raw(path: str) -> Any:
    if path.endswith(".npy"):
        import numpy as np
        return torch.from_numpy(np.load(path))
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def acts_delta(cache: Dict[str, torch.Tensor], layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample ``gt - pred`` at ``layer`` and a validity mask (both acts non-zero)."""
    if "delta" in cache:
        d = cache["delta"]
        d = d[:, layer, :] if d.dim() == 3 else d
        d = d.to(torch.float32)
        return d, d.norm(dim=-1) > 1e-6
    gt, pd = cache["gt_acts"], cache["pred_acts"]
    if gt.dim() == 3:
        gt, pd = gt[:, layer, :], pd[:, layer, :]
    valid = (gt.norm(dim=-1) > 1e-6) & (pd.norm(dim=-1) > 1e-6)
    return (gt - pd).to(torch.float32), valid


def mean_vector_from_acts(cache: Dict[str, torch.Tensor], layer: int) -> torch.Tensor:
    """Dataset-level SV: mean of ``gt - pred`` over rows where both acts exist."""
    d, valid = acts_delta(cache, layer)
    n_valid = int(valid.sum())
    if 0 < n_valid < d.shape[0]:
        print(f"[vectors] mean SV: dropping {d.shape[0] - n_valid} rows with missing acts "
              f"({n_valid}/{d.shape[0]} used)")
        return d[valid].mean(dim=0)
    return d.mean(dim=0)


def _pick_layer(table: Dict[Any, torch.Tensor], layer: Optional[int], path: str) -> torch.Tensor:
    if layer is None:
        if len(table) == 1:
            return next(iter(table.values()))
        raise ValueError(f"{path} holds vectors for several layers {list(table)}; pass a layer")
    for key in (layer, str(layer), f"block_{layer}", f"layer_{layer}"):
        if key in table:
            return table[key]
    raise KeyError(f"{path} has no vector for layer {layer}; available: {list(table)}")


def load_vector(path: str, layer: Optional[int] = None, kind: str = "auto") -> torch.Tensor:
    """Load a steering vector as float32 ``[D]`` or ``[N, D]``.

    ``kind`` only matters for acts caches / delta matrices: ``"mean"`` reduces
    to the dataset-level vector, ``"per_sample"`` keeps one row per example.
    ``"auto"`` means ``"mean"`` for acts caches and "as stored" otherwise.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    obj = _load_raw(path)
    if isinstance(obj, dict):
        if "gt_acts" in obj or "delta" in obj:
            if layer is None:
                raise ValueError(f"{path} is an acts cache: a layer is required")
            if kind == "per_sample":
                v, _ = acts_delta(obj, layer)
            else:
                v = mean_vector_from_acts(obj, layer)
        elif "direction" in obj or "vector" in obj:
            v = obj.get("direction", obj.get("vector"))
        elif "vectors" in obj or "raw_mean_vectors" in obj:
            v = _pick_layer(obj.get("vectors", obj.get("raw_mean_vectors")), layer, path)
        elif all(torch.is_tensor(x) for x in obj.values()):  # e.g. safetensors file
            v = _pick_layer(obj, layer, path)
        else:
            raise ValueError(f"Unrecognised vector file {path}: keys {list(obj)}")
    else:
        v = obj
    v = torch.as_tensor(v).to(torch.float32)
    if v.dim() == 2 and kind == "mean":
        v = v.mean(dim=0)
    if v.dim() not in (1, 2):
        raise ValueError(f"{path}: expected [D] or [N, D], got shape {tuple(v.shape)}")
    return v


def save_vector(v: torch.Tensor, path: str, meta: Optional[Dict[str, Any]] = None) -> None:
    """Save as ``{"direction": v, "meta": {...}}`` (readable by :func:`load_vector`)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    meta = dict(meta or {})
    meta.setdefault("shape", list(v.shape))
    if v.dim() == 1:
        meta.setdefault("v_norm", float(v.norm()))
    torch.save({"direction": v.contiguous(), "meta": meta}, path)


def random_vectors(n: int, dim: int, seed: int = 0) -> torch.Tensor:
    """Per-sample Gaussian directions (the random control of Table 2)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g)
