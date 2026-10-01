import json
import os

import pytest
from conftest import SMALL_MODEL

from visu_predict import __version__
from visu_predict.cli import main


def _train(root, out, seed, *extra):
    args = ["train", "--dataset", "SYN", "--data", root, "--out", out, "--device", "cpu",
            "--seed", str(seed), "--run-name", f"syn_s{seed}", "--batch-size", "8", "--epochs", "2",
            "--max-train-batches", "3", "--save-predictions", *SMALL_MODEL, *extra]
    assert main(args) == 0
    return os.path.join(out, f"syn_s{seed}")


def test_version_and_help(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and __version__ in capsys.readouterr().out
    assert main([]) == 1
    assert "train" in capsys.readouterr().out


def test_baselines_command(syn, tmp_path):
    root, _ = syn
    assert main(["baselines", "--dataset", "SYN", "--data", root, "--out", str(tmp_path), "--device", "cpu"]) == 0
    with open(tmp_path / "SYN_baselines" / "results.json") as f:
        res = json.load(f)
    assert set(res) == {"persistence", "historical_average"}


@pytest.mark.slow
def test_train_evaluate_aggregate_ensemble(syn, tmp_path, capsys):
    root, _ = syn
    out = str(tmp_path / "runs")
    r42 = _train(root, out, 42)
    r43 = _train(root, out, 43)
    with open(os.path.join(r42, "results.json")) as f:
        recorded = json.load(f)
    assert recorded["args"]["dataset"] == "SYN" and "func" not in recorded["args"]

    # evaluate reproduces the recorded test metrics from best.pt alone
    metrics_file = str(tmp_path / "eval.json")
    assert main(["evaluate", r42, "--data", root, "--device", "cpu", "--json", metrics_file,
                 "--save-predictions", str(tmp_path / "preds.npz")]) == 0
    with open(metrics_file) as f:
        evaluated = json.load(f)
    assert evaluated["all"]["mae"] == pytest.approx(recorded["test"]["all"]["mae"], abs=1e-5)
    assert os.path.exists(tmp_path / "preds.npz")

    # the two seeds form one configuration
    capsys.readouterr()
    stats_file = str(tmp_path / "stats.json")
    assert main(["aggregate", out, "--json", stats_file]) == 0
    with open(stats_file) as f:
        groups = json.load(f)
    assert len(groups) == 1 and groups[0]["label"] == "syn" and len(groups[0]["runs"]) == 2
    assert "± " in capsys.readouterr().out

    ens_file = str(tmp_path / "ens.json")
    assert main(["ensemble", r42, r43, "--json", ens_file]) == 0
    with open(ens_file) as f:
        ens = json.load(f)
    assert ens["storage_deviation"] < 0.01          # fp16 prediction files re-score faithfully
    members = [m["recomputed"]["all"]["mae"] for m in ens["members"].values()]
    assert ens["ensemble"]["all"]["mae"] <= max(members) + 1e-9


@pytest.mark.slow
def test_evaluate_rebuilds_runs_without_a_stored_config(syn, tmp_path):
    """Checkpoints trained before configs were stored are rebuilt from results.json."""
    import torch

    root, _ = syn
    run = _train(root, str(tmp_path), 7, "--graph-bias")
    ck = torch.load(os.path.join(run, "best.pt"), weights_only=False)
    del ck["model_class"], ck["model_config"]
    torch.save(ck, os.path.join(run, "best.pt"))
    assert main(["evaluate", run, "--data", root, "--device", "cpu", "--json", str(tmp_path / "m.json")]) == 0
    with open(tmp_path / "m.json") as f, open(os.path.join(run, "results.json")) as g:
        assert json.load(f)["all"]["mae"] == pytest.approx(json.load(g)["test"]["all"]["mae"], abs=1e-5)


def test_weather_command_needs_a_location_for_custom_datasets(syn):
    root, _ = syn
    with pytest.raises(ValueError, match="--lat, --lon and --tz"):
        main(["weather", "--data", root, "--datasets", "SYN"])
    assert main(["weather", "--data", root, "--datasets", "SYN", "METR-LA", "--lat", "1", "--lon", "2"]) == 2
