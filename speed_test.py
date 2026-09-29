#!/usr/bin/env python
"""Measure STTransformer training throughput for several precision / compile
settings (single process). Usage: python speed_test.py <data_dir> [dataset]"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from traffic_transformer.metrics import masked_mae_loss  # noqa: E402
from traffic_transformer.st_data import load_st_benchmark  # noqa: E402
from traffic_transformer.st_model import STTransformer  # noqa: E402
from traffic_transformer.st_training import _autocast  # noqa: E402


def bench(data, precision, compile_, graph_bias=False, steps=60, warmup=15):
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = STTransformer(num_nodes=data.num_nodes, steps_per_day=data.steps_per_day,
                          adj=data.adj, graph_bias=graph_bias).cuda()
    fwd = torch.compile(model) if compile_ else model
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    it = iter(data.train)
    torch.cuda.reset_peak_memory_stats()
    t_start = None
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize()
            t_start = time.time()
        batch = next(it)
        with _autocast("cuda", precision):
            out = fwd(batch["x"], batch["tod"], batch["dow"])
        loss = masked_mae_loss(data.scaler.inverse_transform(out.float()), batch["y"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    ms = (time.time() - t_start) / steps * 1000
    mem = torch.cuda.max_memory_allocated() / 2**30
    n_steps = len(data.train)
    print(f"{precision:5s} compile={str(compile_):5s} graph_bias={str(graph_bias):5s} | "
          f"{ms:6.1f} ms/step | epoch ~{ms * n_steps / 60000:5.1f} min (train only) | "
          f"peak {mem:4.1f} GB", flush=True)


if __name__ == "__main__":
    data_dir = sys.argv[1]
    dataset = sys.argv[2] if len(sys.argv) > 2 else "PEMS-BAY"
    data = load_st_benchmark(data_dir, dataset, batch_size=16, device="cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} | {dataset} | {len(data.train)} steps/epoch")
    bench(data, "tf32", False)
    bench(data, "bf16", False)
    bench(data, "bf16", False, graph_bias=True)
    try:
        bench(data, "bf16", True)
        bench(data, "tf32", True)
    except Exception as e:  # compile problems must not hide the other numbers
        print(f"compile failed: {type(e).__name__}: {e}")
