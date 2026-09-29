#!/usr/bin/env python
"""
Run benchmark experiments from a JSON queue file, several at a time on one GPU.

    python queue_runner.py --queue <queue.json> --out <results/benchmark> \
        --data /content/data --max-concurrent 4

queue.json is a list of {"name": ..., "args": ...} entries, e.g.
    [{"name": "PEMS-BAY_st_hist",
      "args": "--dataset PEMS-BAY --model st_transformer --history-lags 288 2016"}]

The file is re-read on every poll, so experiments can be appended while the
runner is active (e.g. by editing the file on Google Drive). A run is skipped
when its results.json exists, resumed from last.pt when present, and left
alone when another process is evidently training it (log modified in the last
``--busy-minutes``). Failed runs are retried once with --resume. The runner
exits when every queued run is finished or has failed twice.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def load_queue(path):
    try:
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        return [(d["name"], d.get("args", "")) for d in items]
    except Exception as e:  # a half-synced file must not kill the runner
        print(f"[warn] could not read queue ({e}); keeping previous queue", flush=True)
        return None


def last_epoch_line(run_dir):
    p = os.path.join(run_dir, "train_log.txt")
    if not os.path.exists(p):
        return "(starting)"
    with open(p, encoding="utf-8", errors="ignore") as f:
        lines = [l.rstrip() for l in f if l.strip()]
    ep = [l for l in lines if l.startswith("Epoch")]
    return (ep[-1] if ep else (lines[-1] if lines else "(empty)"))[:150]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-concurrent", type=int, default=4)
    ap.add_argument("--common", default="--precision tf32 --save-predictions")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--busy-minutes", type=float, default=15.0)
    a = ap.parse_args()
    a.queue, a.out, a.data = (os.path.abspath(p) for p in (a.queue, a.out, a.data))

    running = {}          # name -> (Popen, logfile)
    attempts = {}         # name -> launches so far
    queue = []
    t0 = time.time()
    while True:
        q = load_queue(a.queue)
        if q is not None:
            queue = q
        # reap finished processes
        for name in list(running):
            proc, logf = running[name]
            if proc.poll() is not None:
                logf.close()
                ok = proc.returncode == 0
                print(f"[finish] {name}: {'OK' if ok else f'FAILED (exit {proc.returncode})'}", flush=True)
                del running[name]
        # launch pending runs
        pending = []
        for name, args in queue:
            run_dir = os.path.join(a.out, name)
            if name in running or os.path.exists(os.path.join(run_dir, "results.json")):
                continue
            if attempts.get(name, 0) >= 2:
                continue
            log = os.path.join(run_dir, "train_log.txt")
            if (name not in attempts and os.path.exists(log)
                    and time.time() - os.path.getmtime(log) < a.busy_minutes * 60):
                continue  # being trained by another process
            pending.append((name, args))
        for name, args in pending:
            if len(running) >= a.max_concurrent:
                break
            run_dir = os.path.join(a.out, name)
            os.makedirs(run_dir, exist_ok=True)
            resume = "--resume" if os.path.exists(os.path.join(run_dir, "last.pt")) else ""
            cmd = (f'cd "{HERE}" && "{sys.executable}" -W ignore run_benchmark.py {args} {a.common} '
                   f'{resume} --input-dir "{a.data}" --output-dir "{a.out}" --run-name {name}')
            logf = open(os.path.join(run_dir, "stdout.txt"), "a")
            running[name] = (subprocess.Popen(cmd, shell=True, stdout=logf, stderr=subprocess.STDOUT), logf)
            attempts[name] = attempts.get(name, 0) + 1
            print(f"[launch] {name} {resume}".rstrip(), flush=True)
            time.sleep(15)
        waiting = [n for n, _ in pending if n not in running]
        elapsed = (time.time() - t0) / 60
        print(f"--- {time.strftime('%H:%M:%S')} | {elapsed:.0f} min | running {len(running)}, "
              f"waiting {len(waiting)}", flush=True)
        for name in running:
            print(f"   {name:28s} {last_epoch_line(os.path.join(a.out, name))}", flush=True)
        with open(os.path.join(a.out, "queue_runner_status.json"), "w") as f:
            json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "running": list(running),
                       "waiting": waiting, "elapsed_min": round(elapsed, 1)}, f)
        if not running and not waiting:
            print("Queue empty - all runs finished.", flush=True)
            break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
