import _reference as ref
import pytest
import torch
from conftest import HIDDEN, LAYERS, make_tiny_model

from sv_training.steering import (Injection, SteeringContext, attach, detach,
                                  find_transformer_blocks, slerp_rotate)
from sv_training.vectors import random_vectors

N = 5
T = 12
PROMPT = 5  # first PROMPT positions are prompt tokens (label -100)


def _batch(sample_idx=2):
    torch.manual_seed(1)
    ids = torch.randint(0, 128, (1, T))
    labels = ids.clone()
    labels[0, :PROMPT] = -100
    return ids, labels, torch.tensor([sample_idx])


def _run_reference(model, layer, hook_fn, state, labels, sample_idx, ids):
    state.sample_idx = int(sample_idx[0])
    state.current_labels = labels
    h = model.model.layers[layer].register_forward_hook(hook_fn)
    out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    grad = model.model.embed_tokens.weight.grad.clone()
    model.zero_grad()
    h.remove()
    return out.logits.detach(), grad


def _run_new(model, injections, labels, sample_idx, ids):
    ctx = SteeringContext()
    handles = attach(model, injections, ctx)
    with ctx.batch(labels=labels, sample_idx=sample_idx):
        out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    grad = model.model.embed_tokens.weight.grad.clone()
    model.zero_grad()
    detach(handles)
    return out.logits.detach(), grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("method", ["mean_sv_add", "slerp", "per_inst_add", "sv_first_k",
                                    "random_gauss", "skip_first_k"])
def test_hooks_match_paper_implementation(method, dtype):
    model = make_tiny_model(dtype).train()
    ids, labels, idx = _batch()
    torch.manual_seed(3)
    raw = torch.randn(N, HIDDEN) * 4
    unit = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    layer = 2
    state = ref.SVState()

    if method == "mean_sv_add":
        mean = raw.mean(0)
        hook = ref.make_mean_additive_hook(mean, state, 0.5)
        inj = Injection(mean, layer, alpha=0.5, op="add")
    elif method == "slerp":
        hook = ref.make_per_inst_slerp_hook(unit, state, 0.1)
        inj = Injection(raw, layer, alpha=0.1, op="slerp", normalize=True)
    elif method == "per_inst_add":
        hook = ref.make_per_inst_additive_hook(unit, state, 0.1)
        inj = Injection(raw, layer, alpha=0.1, op="add", normalize=True)
    elif method == "sv_first_k":
        hook = ref.make_per_inst_slerp_first_k_hook(unit, state, 0.1, 3)
        inj = Injection(raw, layer, alpha=0.1, op="slerp", normalize=True, first_k=3)
    elif method == "skip_first_k":
        state.sv_skip_first_k_answer = 2
        hook = ref.make_per_inst_slerp_hook(unit, state, 0.1)
        inj = Injection(raw, layer, alpha=0.1, op="slerp", normalize=True, skip_first_k=2)
    else:
        hook = ref.make_random_gaussian_per_sample_hook(N, HIDDEN, state, 0.1, seed=42)
        inj = Injection(random_vectors(N, HIDDEN, seed=42), layer, alpha=0.1, op="slerp",
                        normalize=True)

    logits_ref, grad_ref = _run_reference(model, layer, hook, state, labels, idx, ids)
    logits_new, grad_new = _run_new(model, [inj], labels, idx, ids)
    assert torch.equal(logits_ref, logits_new)
    assert torch.equal(grad_ref, grad_new)

    # and steering actually changed something
    logits_plain = model(input_ids=ids).logits.detach()
    assert not torch.equal(logits_plain, logits_new)


def test_multi_layer_matches_reference():
    model = make_tiny_model().train()
    ids, labels, idx = _batch()
    raw = torch.randn(N, LAYERS, HIDDEN)
    state = ref.SVState()
    state.sample_idx, state.current_labels = int(idx[0]), labels
    handles = []
    for li in (1, 2, 3):
        d = raw[:, li] / raw[:, li].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        handles.append(model.model.layers[li].register_forward_hook(
            ref.make_per_inst_slerp_hook(d, state, 0.1)))
    logits_ref = model(input_ids=ids).logits.detach()
    for h in handles:
        h.remove()
    injs = [Injection(raw[:, li], li, alpha=0.1, op="slerp") for li in (1, 2, 3)]
    logits_new, _ = _run_new(model, injs, labels, idx, ids)
    assert torch.equal(logits_ref, logits_new)


def test_only_answer_tokens_are_steered():
    model = make_tiny_model().train()
    ids, labels, idx = _batch()
    captured = {}
    blocks = find_transformer_blocks(model)
    h = blocks[2].register_forward_hook(lambda m, i, o: captured.setdefault("plain", o))
    model(input_ids=ids)
    h.remove()

    v = torch.randn(HIDDEN)
    ctx = SteeringContext()
    handles = attach(model, [Injection(v, 2, alpha=0.5, op="add")], ctx)
    # hooks run in registration order: our steering hook first, then the capture hook
    h = blocks[2].register_forward_hook(lambda m, i, o: captured.setdefault("steered", o))
    with ctx.batch(labels=labels, sample_idx=idx):
        model(input_ids=ids)
    h.remove()
    detach(handles)

    plain = captured["plain"][0] if isinstance(captured["plain"], tuple) else captured["plain"]
    steered = captured["steered"][0] if isinstance(captured["steered"], tuple) else captured["steered"]
    assert torch.equal(plain[0, :PROMPT], steered[0, :PROMPT])
    assert torch.allclose(steered[0, PROMPT:] - plain[0, PROMPT:], 0.5 * v.expand(T - PROMPT, -1),
                          atol=1e-5)


def test_eval_mode_and_missing_labels_are_noops():
    model = make_tiny_model()
    ids, labels, idx = _batch()
    plain = model.eval()(input_ids=ids).logits
    ctx = SteeringContext()
    handles = attach(model, [Injection(torch.randn(HIDDEN), 1, alpha=1.0)], ctx)
    with ctx.batch(labels=labels, sample_idx=idx):
        assert torch.equal(model.eval()(input_ids=ids).logits, plain)  # eval mode
    assert torch.equal(model.train()(input_ids=ids).logits, plain)  # no labels in context
    detach(handles)


def test_all_tokens_and_batched_per_sample():
    model = make_tiny_model().train()
    ids = torch.randint(0, 128, (2, T))
    table = torch.randn(N, HIDDEN)
    single = []
    for b, sidx in enumerate((1, 4)):
        ctx = SteeringContext()
        hs = attach(model, [Injection(table, 2, alpha=0.3, answer_only=False)], ctx)
        with ctx.batch(sample_idx=torch.tensor([sidx])):
            single.append(model(input_ids=ids[b:b + 1]).logits.detach())
        detach(hs)
    ctx = SteeringContext()
    hs = attach(model, [Injection(table, 2, alpha=0.3, answer_only=False)], ctx)
    with ctx.batch(sample_idx=torch.tensor([1, 4])):
        batched = model(input_ids=ids).logits.detach()
    detach(hs)
    assert torch.allclose(batched, torch.cat(single), atol=1e-5)


def test_slerp_properties():
    h = torch.randn(7, 16)
    v = torch.randn(16)
    v = v / v.norm()
    assert torch.allclose(slerp_rotate(h, v, 0.0), h, atol=1e-5)
    out = slerp_rotate(h, v, 0.3)
    assert torch.allclose(out.norm(dim=-1), h.norm(dim=-1), atol=1e-4)
    end = slerp_rotate(h, v, 1.0)
    assert torch.allclose(end / end.norm(dim=-1, keepdim=True), v.expand(7, -1), atol=1e-3)


def test_bad_specs_raise(tiny_model):
    with pytest.raises(ValueError):
        Injection(torch.randn(HIDDEN), 1, op="mul")
    with pytest.raises(ValueError):
        attach(tiny_model, [Injection(torch.randn(HIDDEN + 1), 1)], SteeringContext())
    with pytest.raises(ValueError):
        attach(tiny_model, [Injection(torch.randn(HIDDEN), LAYERS)], SteeringContext())
