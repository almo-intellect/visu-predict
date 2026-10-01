import json
import os
import sys

import pytest
from conftest import SMALL_MODEL

from visu_predict.job_queue import build_command, load_queue, run_queue


def test_queue_file_parsing(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"name": "a", "args": "--dataset METR-LA"}, {"name": "b"}]))
    assert load_queue(str(path)) == [("a", "--dataset METR-LA"), ("b", "")]
    path.write_text("[{broken")
    assert load_queue(str(path)) is None           # a half-synced file keeps the previous queue


def test_command_line_of_a_queued_run():
    cmd = build_command("r1", "--dataset PEMS-BAY --history-lags 288 2016", "--precision bf16",
                        "/data", "/runs", resume=True)
    assert cmd[:6] == [sys.executable, "-W", "ignore", "-m", "visu_predict", "train"]
    assert cmd[6:] == ["--precision", "bf16", "--dataset", "PEMS-BAY", "--history-lags", "288", "2016",
                       "--resume", "--data", "/data", "--out", "/runs", "--run-name", "r1"]


@pytest.mark.slow
def test_queue_runs_jobs_to_completion(syn, tmp_path):
    root, _ = syn
    args = " ".join(["--dataset SYN --device cpu --epochs 1 --max-train-batches 2 --batch-size 8", *SMALL_MODEL])
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps([{"name": "job1", "args": args}, {"name": "job2", "args": args + " --seed 1"}]))
    out = tmp_path / "runs"
    outcome = run_queue(str(queue), root, str(out), max_concurrent=2, common="", poll=0.5, stagger=0)
    assert outcome == {"job1": True, "job2": True}
    assert all(os.path.exists(out / name / "results.json") for name in ("job1", "job2"))
    # a second pass finds nothing left to do
    assert run_queue(str(queue), root, str(out), common="", poll=0.5, stagger=0) == {}
