"""End-to-end CPU smoke test: `sv-train train` on a tiny offline model and a custom JSONL file."""
import json
import os

import torch
from conftest import HIDDEN, make_tiny_model, make_tiny_tokenizer

from sv_training import steering
from sv_training.cli import main


def test_train_with_custom_vector(tmp_path, monkeypatch):
    rows = [{"question": f"what is {i} plus {i}", "answer": f"it is {2 * i}"} for i in range(6)]
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    tok = make_tiny_tokenizer([r["question"] + " " + r["answer"] for r in rows])
    model = make_tiny_model()
    model.resize_token_embeddings(len(tok))
    model_dir = tmp_path / "model"
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)

    per_sample = torch.randn(len(rows), HIDDEN)
    torch.save(per_sample, tmp_path / "per_sample.pt")
    torch.save({"direction": torch.randn(HIDDEN)}, tmp_path / "global.pt")

    calls = {"n": 0}
    real_make_hook = steering.make_hook

    def counting_make_hook(inj, ctx, training_only=True):
        hook = real_make_hook(inj, ctx, training_only)

        def wrapped(module, inputs, output):
            out = hook(module, inputs, output)
            calls["n"] += int(out is not output)
            return out
        return wrapped

    monkeypatch.setattr(steering, "make_hook", counting_make_hook)

    out_dir = tmp_path / "adapter"
    main(["train", "--model", str(model_dir), "--dataset", str(data), "--output_dir", str(out_dir),
          "--sv", str(tmp_path / "global.pt"), "--layer", "1", "--alpha", "0.5",
          "--inject", f"path={tmp_path / 'per_sample.pt'},layer=2,op=slerp,alpha=0.1",
          "--max_steps", "2", "--grad_accum", "2", "--logging_steps", "1"])

    assert os.path.isfile(out_dir / "adapter_config.json")
    assert calls["n"] >= 2 * 4  # two injections x four forward passes
