"""
Run training jobs from a JSON queue file, several at a time on one GPU.

    visu-predict queue --queue configs/paper_runs.json --data data --out runs --max-concurrent 2

The queue file is a list of ``{"name": ..., "args": ...}`` entries, where
``args`` are ``visu-predict train`` options (they override ``--common``), e.g.::

    [{"name": "PEMS-BAY_st_hist",
      "args": "--dataset PEMS-BAY --history-lags 288 2016"}]

The file is re-read on every poll, so experiments can be appended while the
runner is active (e.g. by editing the file on Google Drive). A run is skipped
when its ``results.json`` exists, resumed from ``last.pt`` when present, and
left alone when another process is evidently training it (log modified in the
last ``--busy-minutes``). Failed runs are retried once with ``--resume``. The
runner exits when every queued run is finished or has failed twice.
"""

import json
import os
import shlex
import subprocess
import sys
import time


def load_queue(path: str) -> list[tuple[str, str]] | None:
    """Return ``[(name, args), ...]``, or None if the file cannot be read right now."""
    try:
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        return [(d["name"], d.get("args", "")) for d in items]
    except Exception as e:  # a half-synced file must not kill the runner
        print(f"[warn] could not read queue ({e}); keeping previous queue", flush=True)
        return None


def last_epoch_line(run_dir: str) -> str:
    p = os.path.join(run_dir, "train_log.txt")
    if not os.path.exists(p):
        return "(starting)"
    with open(p, encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip() for line in f if line.strip()]
    ep = [line for line in lines if line.startswith("Epoch")]
    return (ep[-1] if ep else (lines[-1] if lines else "(empty)"))[:150]


def build_command(name: str, args: str, common: str, data_dir: str, out_dir: str, resume: bool) -> list[str]:
    # common options first, so a run's own args can override them
    cmd = [sys.executable, "-W", "ignore", "-m", "visu_predict", "train",
           *shlex.split(common), *shlex.split(args)]
    if resume:
        cmd.append("--resume")
    return [*cmd, "--data", data_dir, "--out", out_dir, "--run-name", name]


def run_queue(queue_path: str, data_dir: str, out_dir: str, max_concurrent: int = 2,
              common: str = "--save-predictions", poll: float = 120.0,
              busy_minutes: float = 15.0, stagger: float = 15.0) -> dict[str, bool]:
    """Process the queue until every run is finished or has failed twice.

    Returns ``{run name: succeeded}`` for the runs launched by this call.
    """
    queue_path, data_dir, out_dir = (os.path.abspath(p) for p in (queue_path, data_dir, out_dir))
    running: dict[str, tuple[subprocess.Popen, object]] = {}
    attempts: dict[str, int] = {}
    outcome: dict[str, bool] = {}
    queue: list[tuple[str, str]] = []
    t0 = time.time()
    while True:
        q = load_queue(queue_path)
        if q is not None:
            queue = q
        # reap finished processes
        for name in list(running):
            proc, logf = running[name]
            if proc.poll() is not None:
                logf.close()
                ok = proc.returncode == 0
                outcome[name] = ok
                print(f"[finish] {name}: {'OK' if ok else f'FAILED (exit {proc.returncode})'}", flush=True)
                del running[name]
        # launch pending runs
        pending = []
        for name, args in queue:
            run_dir = os.path.join(out_dir, name)
            if name in running or os.path.exists(os.path.join(run_dir, "results.json")):
                continue
            if attempts.get(name, 0) >= 2:
                continue
            log = os.path.join(run_dir, "train_log.txt")
            if (name not in attempts and os.path.exists(log)
                    and time.time() - os.path.getmtime(log) < busy_minutes * 60):
                continue  # being trained by another process
            pending.append((name, args))
        for name, args in pending:
            if len(running) >= max_concurrent:
                break
            run_dir = os.path.join(out_dir, name)
            os.makedirs(run_dir, exist_ok=True)
            resume = os.path.exists(os.path.join(run_dir, "last.pt"))
            cmd = build_command(name, args, common, data_dir, out_dir, resume)
            logf = open(os.path.join(run_dir, "stdout.txt"), "a")  # closed when the run is reaped
            running[name] = (subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT), logf)
            attempts[name] = attempts.get(name, 0) + 1
            print(f"[launch] {name}{' (resume)' if resume else ''}", flush=True)
            time.sleep(stagger)
        waiting = [n for n, _ in pending if n not in running]
        elapsed = (time.time() - t0) / 60
        print(f"--- {time.strftime('%H:%M:%S')} | {elapsed:.0f} min | running {len(running)}, "
              f"waiting {len(waiting)}", flush=True)
        for name in running:
            print(f"   {name:28s} {last_epoch_line(os.path.join(out_dir, name))}", flush=True)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "queue_runner_status.json"), "w") as f:
            json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "running": list(running),
                       "waiting": waiting, "elapsed_min": round(elapsed, 1)}, f)
        if not running and not waiting:
            print("Queue empty - all runs finished.", flush=True)
            return outcome
        time.sleep(poll)
