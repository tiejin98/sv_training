"""Verbatim copies of the hook implementations used for the paper runs (original
sft.py / _steering_helpers.py). Only used to check that the packaged
`sv_training.steering` hooks are numerically identical."""
import torch


def unwrap_ddp(m):
    while hasattr(m, "module"):
        m = m.module
    return m


class SVState:
    def __init__(self):
        self.sample_idx = -1
        self.sv_skip_first_k_answer = 0
        self.current_labels = None


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


def make_per_inst_slerp_hook(deltas: torch.Tensor, state: SVState, alpha: float):
    """Per-instance Slerp on answer tokens. deltas: [N, D] unit-normalised."""
    def hook(module, inputs, output):
        if not module.training:
            return output
        h = output[0] if isinstance(output, tuple) else output  # [B, T, D]
        assert h.shape[0] == 1
        idx = state.sample_idx
        if idx < 0 or idx >= deltas.shape[0]:
            return output
        # labels mask: set on the wrapped model by PerSampleTrainer before forward
        base_ref = unwrap_ddp(module) if hasattr(module, 'module') else None
        labels = getattr(module, "_sv_labels", None)
        if labels is None:
            # look up via shared closure attribute set by Trainer
            labels = state.current_labels  # set by trainer
        if labels is None:
            return output
        mask = (labels[0] != -100)  # [T]
        if not mask.any():
            return output
        v = deltas[idx].to(h.dtype).to(h.device)  # [D]
        # Skip slerp for zero-norm deltas (e.g. correct-only filter zeros out
        # rows where bootstrap pred was already correct). slerp_rotate with
        # v=0 gives cos_θ=0 → scales h by sin((1-α)π/2) ≈ 0.988 instead of
        # leaving it unchanged; bypass entirely.
        if v.norm() < 1e-6:
            return output
        T = h.shape[1]
        idx_positions = mask.nonzero(as_tuple=True)[0]
        # Phase 7: optionally skip first-K answer tokens (code-aware mask: drop def line)
        sk = getattr(state, "sv_skip_first_k_answer", 0)
        if sk > 0 and idx_positions.numel() > sk:
            idx_positions = idx_positions[sk:]  # answer-token positions
        if idx_positions.numel() == 0:
            return output
        h_new = h.clone()
        h_slice = h_new[0, idx_positions, :]          # [K, D]
        h_slice = slerp_rotate(h_slice, v, float(alpha))
        h_new[0, idx_positions, :] = h_slice
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        return h_new
    return hook


def make_mean_additive_hook(mean_dir: torch.Tensor, state: SVState, alpha: float):
    """Mean-SV additive on answer tokens: h ← h + α·v_mean.

    mean_dir: [D] — IMPORTANT: passed AS-IS (no re-normalisation inside hook).
    v2 calibration used RAW (unnormalised) mean vector with α=0.5; caller must
    pass the raw delta-mean (shape [D]) to match v2 algorithmically. Applying
    to answer tokens only via labels != -100 mask. Not per-sample — single
    global direction for all training samples.
    """
    v_mean = mean_dir.to(torch.float32)

    def hook(module, inputs, output):
        if not module.training:
            return output
        h = output[0] if isinstance(output, tuple) else output  # [B, T, D]
        labels = state.current_labels
        if labels is None:
            return output
        mask = (labels[0] != -100)
        if not mask.any():
            return output
        v = v_mean.to(h.dtype).to(h.device)  # [D] — raw, not unit-normalised
        idx_positions = mask.nonzero(as_tuple=True)[0]
        # Phase 7: optionally skip first-K answer tokens (code-aware mask: drop def line)
        sk = getattr(state, "sv_skip_first_k_answer", 0)
        if sk > 0 and idx_positions.numel() > sk:
            idx_positions = idx_positions[sk:]
        if idx_positions.numel() == 0:
            return output
        h_new = h.clone()
        h_new[0, idx_positions, :] = h_new[0, idx_positions, :] + alpha * v
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        return h_new
    return hook


def make_per_inst_slerp_first_k_hook(deltas: torch.Tensor, state: SVState,
                                      alpha: float, k: int):
    """Per-instance Slerp on FIRST K answer tokens (front-loaded scope)."""
    def hook(module, inputs, output):
        if not module.training:
            return output
        h = output[0] if isinstance(output, tuple) else output
        assert h.shape[0] == 1
        idx = state.sample_idx
        if idx < 0 or idx >= deltas.shape[0]:
            return output
        labels = state.current_labels
        if labels is None:
            return output
        mask = (labels[0] != -100)
        if not mask.any():
            return output
        v = deltas[idx].to(h.dtype).to(h.device)
        if v.norm() < 1e-6:
            return output
        idx_positions = mask.nonzero(as_tuple=True)[0]
        # Phase 7: optionally skip first-K answer tokens (code-aware mask: drop def line)
        sk = getattr(state, "sv_skip_first_k_answer", 0)
        if sk > 0 and idx_positions.numel() > sk:
            idx_positions = idx_positions[sk:]   # all answer positions
        if idx_positions.numel() == 0:
            return output
        # Keep only the FIRST k of those
        idx_positions_k = idx_positions[:k]
        h_new = h.clone()
        h_slice = h_new[0, idx_positions_k, :]
        h_slice = slerp_rotate(h_slice, v, float(alpha))
        h_new[0, idx_positions_k, :] = h_slice
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        return h_new
    return hook


def make_per_inst_additive_hook(deltas: torch.Tensor, state: SVState, alpha: float):
    """Per-instance additive: h ← h + α·v_i  where v_i = delta_i / ‖delta_i‖."""
    def hook(module, inputs, output):
        if not module.training:
            return output
        h = output[0] if isinstance(output, tuple) else output
        assert h.shape[0] == 1
        idx = state.sample_idx
        if idx < 0 or idx >= deltas.shape[0]:
            return output
        labels = state.current_labels
        if labels is None:
            return output
        mask = (labels[0] != -100)
        if not mask.any():
            return output
        v = deltas[idx].to(h.dtype).to(h.device)
        if v.norm() < 1e-6:
            return output
        idx_positions = mask.nonzero(as_tuple=True)[0]
        # Phase 7: optionally skip first-K answer tokens (code-aware mask: drop def line)
        sk = getattr(state, "sv_skip_first_k_answer", 0)
        if sk > 0 and idx_positions.numel() > sk:
            idx_positions = idx_positions[sk:]
        if idx_positions.numel() == 0:
            return output
        h_new = h.clone()
        h_new[0, idx_positions, :] = h_new[0, idx_positions, :] + alpha * v
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        return h_new
    return hook


def make_random_gaussian_per_sample_hook(n_samples: int, hidden_dim: int,
                                          state: SVState, alpha: float,
                                          seed: int = 0):
    """Control: per-sample random Gaussian direction (unit-norm). Drawn once at init,
    kept fixed per sample_idx. Use to verify SV gain isn't just "any perturbation helps"."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n_samples, hidden_dim, generator=g)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def hook(module, inputs, output):
        if not module.training:
            return output
        h = output[0] if isinstance(output, tuple) else output
        assert h.shape[0] == 1
        idx = state.sample_idx
        if idx < 0 or idx >= dirs.shape[0]:
            return output
        labels = state.current_labels
        if labels is None:
            return output
        mask = (labels[0] != -100)
        if not mask.any():
            return output
        v = dirs[idx].to(h.dtype).to(h.device)
        idx_positions = mask.nonzero(as_tuple=True)[0]
        # Phase 7: optionally skip first-K answer tokens (code-aware mask: drop def line)
        sk = getattr(state, "sv_skip_first_k_answer", 0)
        if sk > 0 and idx_positions.numel() > sk:
            idx_positions = idx_positions[sk:]
        if idx_positions.numel() == 0:
            return output
        h_new = h.clone()
        h_slice = h_new[0, idx_positions, :]
        h_slice = slerp_rotate(h_slice, v, float(alpha))
        h_new[0, idx_positions, :] = h_slice
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        return h_new
    return hook


