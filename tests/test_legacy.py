import os

import pytest
from conftest import SMALL_MODEL

from visu_predict.cli import main
from visu_predict.data import load_st_benchmark

pytest.importorskip("seaborn", reason="needs the [legacy] extra")


def test_legacy_adapter_runs_in_the_benchmark_harness(syn):
    from visu_predict.legacy.adapter import LegacyTransformerAdapter

    root, _ = syn
    data = load_st_benchmark(root, "SYN", batch_size=8)
    b = next(iter(data.train))
    model = LegacyTransformerAdapter(num_nodes=6, d_model=32, nhead=4, num_layers=1, dim_feedforward=64)
    assert model(b["x"], b["tod"], b["dow"]).shape == (8, 12, 6)


@pytest.mark.slow
def test_legacy_model_trains_and_evaluates_through_the_cli(syn, tmp_path):
    root, _ = syn
    out = str(tmp_path)
    args = ["train", "--dataset", "SYN", "--data", root, "--out", out, "--device", "cpu", "--model", "legacy",
            "--legacy-d-model", "32", "--legacy-layers", "1", "--legacy-heads", "4", "--epochs", "1",
            "--max-train-batches", "2", "--batch-size", "8", "--run-name", "legacy", *SMALL_MODEL]
    assert main(args) == 0
    run = os.path.join(out, "legacy")
    assert main(["evaluate", run, "--data", root, "--device", "cpu"]) == 0
