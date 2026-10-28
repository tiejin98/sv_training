import os

import numpy as np
import pytest
import torch

from sv_training.vectors import load_vector, mean_vector_from_acts, save_vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLED_SV = os.path.join(ROOT, "quick_start", "data", "sv_Qwen2.5-1.5B_gsm8k_L15_mean.pt")


def test_formats(tmp_path):
    v = torch.randn(8)
    m = torch.randn(5, 8)
    torch.save(v, tmp_path / "bare.pt")
    torch.save(m, tmp_path / "matrix.pt")
    torch.save({"direction": v, "meta": {}}, tmp_path / "dir.pt")
    torch.save({"raw_mean_vectors": {"block_15": v}}, tmp_path / "blocks.pt")
    torch.save({"vectors": {3: v, 7: -v}}, tmp_path / "multi.pt")
    np.save(tmp_path / "v.npy", v.numpy())
    from safetensors.torch import save_file
    save_file({"layer_15": v}, str(tmp_path / "v.safetensors"))

    assert torch.equal(load_vector(str(tmp_path / "bare.pt")), v)
    assert torch.equal(load_vector(str(tmp_path / "matrix.pt")), m)
    assert torch.allclose(load_vector(str(tmp_path / "matrix.pt"), kind="mean"), m.mean(0))
    assert torch.equal(load_vector(str(tmp_path / "dir.pt")), v)
    assert torch.equal(load_vector(str(tmp_path / "blocks.pt"), layer=15), v)
    assert torch.equal(load_vector(str(tmp_path / "blocks.pt")), v)  # single entry
    assert torch.equal(load_vector(str(tmp_path / "multi.pt"), layer=7), -v)
    with pytest.raises(ValueError):
        load_vector(str(tmp_path / "multi.pt"))
    with pytest.raises(KeyError):
        load_vector(str(tmp_path / "multi.pt"), layer=4)
    assert torch.allclose(load_vector(str(tmp_path / "v.npy")), v)
    assert torch.equal(load_vector(str(tmp_path / "v.safetensors"), layer=15), v)

    save_vector(m, str(tmp_path / "saved.pt"))
    assert torch.equal(load_vector(str(tmp_path / "saved.pt")), m)


def test_acts_cache(tmp_path):
    gt = torch.randn(6, 3, 4)
    pd = torch.randn(6, 3, 4)
    gt[0] = 0  # no reference for sample 0 -> excluded from the mean
    pd[5] = 0  # no prediction for sample 5 -> excluded from the mean
    path = str(tmp_path / "acts.pt")
    torch.save({"gt_acts": gt, "pred_acts": pd}, path)

    mean = load_vector(path, layer=1)
    assert torch.allclose(mean, (gt[1:5, 1] - pd[1:5, 1]).mean(0))
    assert torch.allclose(mean, mean_vector_from_acts({"gt_acts": gt, "pred_acts": pd}, 1))
    per = load_vector(path, layer=1, kind="per_sample")
    assert per.shape == (6, 4)
    assert torch.allclose(per, gt[:, 1] - pd[:, 1])
    with pytest.raises(ValueError):
        load_vector(path)  # layer required


@pytest.mark.skipif(not os.path.isfile(BUNDLED_SV), reason="bundled quick-start data not present")
def test_bundled_vector():
    v = load_vector(BUNDLED_SV, layer=15)
    assert v.shape == (1536,)
    assert abs(v.norm().item() - 11.4509) < 1e-3
