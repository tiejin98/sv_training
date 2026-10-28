import pytest

from sv_training.cli import build_injections, build_parser


def _specs(*argv):
    args = build_parser().parse_args(["train", "--model", "m", "--dataset", "gsm8k",
                                      "--output_dir", "o", *argv])
    return build_injections(args)


def test_plain():
    assert _specs() == []
    assert _specs("--method", "plain") == []


def test_presets_match_paper_settings():
    (s,) = _specs("--method", "mean_sv_add", "--acts_cache", "a.pt", "--layer", "15")
    assert (s.path, s.layer, s.alpha, s.op, s.kind, s.normalize) == ("a.pt", 15, 0.5, "add", "mean", False)
    (s,) = _specs("--method", "slerp", "--acts_cache", "a.pt")
    assert (s.alpha, s.op, s.kind, s.normalize, s.layer) == (0.1, "slerp", "per_sample", True, None)
    (s,) = _specs("--method", "sv_first_k", "--acts_cache", "a.pt", "--k", "16")
    assert s.first_k == 16
    specs = _specs("--method", "sv_multi_layer", "--acts_cache", "a.pt", "--multi_layers", "10,15,20")
    assert [s.layer for s in specs] == [10, 15, 20]
    (s,) = _specs("--method", "random_gauss")
    assert s.kind == "random" and s.path is None
    # legacy flag name for the bundled direction file
    (s,) = _specs("--method", "mean_sv_add", "--mean_dir_override", "d.pt")
    assert s.path == "d.pt"


def test_custom_vectors():
    (s,) = _specs("--sv", "v.pt", "--layer", "12", "--alpha", "2", "--op", "add")
    assert (s.path, s.layer, s.alpha, s.op) == ("v.pt", 12, 2.0, "add")
    (s,) = _specs("--sv", "v.pt", "--op", "slerp")
    assert s.alpha == 0.1
    specs = _specs("--layer", "15", "--inject", "path=a.pt,layer=10,alpha=0.1",
                   "--inject", "b.pt", "--inject", "path=c.pt,op=slerp,kind=per_sample,first_k=8")
    assert [(s.path, s.layer, s.alpha, s.op) for s in specs] == [
        ("a.pt", 10, 0.1, "add"), ("b.pt", 15, 0.5, "add"), ("c.pt", 15, 0.5, "slerp")]
    assert specs[2].kind == "per_sample" and specs[2].first_k == 8
    with pytest.raises(ValueError):
        _specs("--inject", "path=a.pt,colour=red")


def test_errors():
    with pytest.raises(SystemExit):
        _specs("--method", "slerp")  # needs an acts cache
    with pytest.raises(SystemExit):
        _specs("--method", "plain", "--sv", "v.pt")


def test_legacy_flag_aliases():
    args = build_parser().parse_args([
        "train", "--model_path", "m", "--dataset", "gsm8k", "--output_dir", "o",
        "--num_train_epochs", "2", "--distill_teacher_json", "t.json",
        "--sv_skip_first_k_answer", "3"])
    assert (args.model, args.epochs, args.teacher_json, args.skip_first_k) == ("m", 2.0, "t.json", 3)
