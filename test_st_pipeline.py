"""CPU tests for the V19 benchmark pipeline (runs in well under a minute).

    python test_st_pipeline.py [path/to/Transformers_Input]

The optional data directory enables real-data checks (split sizes that match
the literature, weather timezone correction).
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traffic_transformer.metrics import horizon_metrics, masked_mae_loss, masked_metrics  # noqa: E402
from traffic_transformer.st_data import (  # noqa: E402
    dcrnn_window_splits, load_st_benchmark, load_weather_features,
)
from traffic_transformer.st_model import (  # noqa: E402
    GraphDistanceBias, LegacyTransformerAdapter, STTransformer, hop_distance_matrix,
)
from traffic_transformer.st_training import naive_baselines  # noqa: E402

RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{(' - ' + extra) if extra else ''}")


def synthetic_dataset(root, T=2000, N=6):
    idx = pd.date_range("2024-01-01", periods=T, freq="5min")
    rng = np.random.default_rng(0)
    base = 50 + 10 * np.sin(2 * np.pi * np.arange(T) / 288)[:, None]
    vals = base + rng.normal(0, 1, (T, N))
    vals[rng.integers(0, T, 50), rng.integers(0, N, 50)] = 0.0
    pd.DataFrame(vals, index=idx, columns=[f"s{i}" for i in range(N)]).to_csv(
        os.path.join(root, "SYN.csv"))
    adj = np.eye(N, dtype=np.float32)
    for i in range(N - 1):
        adj[i, i + 1] = 0.8
    import pickle
    with open(os.path.join(root, "adj_SYN.pkl"), "wb") as f:
        pickle.dump([[f"s{i}" for i in range(N)], {f"s{i}": i for i in range(N)}, adj], f)
    return vals


def main(data_dir=None):
    torch.manual_seed(0)

    # --- splits follow DCRNN rounding --------------------------------------
    sp = dcrnn_window_splits(52116, 12, 12)
    check("PEMS-BAY split sizes = 36465/5209/10419",
          (len(sp["train"]), len(sp["val"]), len(sp["test"])) == (36465, 5209, 10419))
    sp = dcrnn_window_splits(34272, 12, 12)
    check("METR-LA split sizes = 23974/3425/6850",
          (len(sp["train"]), len(sp["val"]), len(sp["test"])) == (23974, 3425, 6850))

    # --- masked metrics against a hand computation --------------------------
    y = np.array([[[10.0, 0.0], [20.0, 40.0]]])      # (1, 2, 2), one masked zero
    p = np.array([[[12.0, 5.0], [18.0, 44.0]]])
    m = masked_metrics(p, y)
    exp_mae = (2 + 2 + 4) / 3
    exp_rmse = np.sqrt((4 + 4 + 16) / 3)
    exp_mape = (0.2 + 0.1 + 0.1) / 3 * 100
    check("masked MAE/RMSE/MAPE exclude zero labels",
          np.isclose(m["mae"], exp_mae) and np.isclose(m["rmse"], exp_rmse)
          and np.isclose(m["mape"], exp_mape), f"{m}")
    hm = horizon_metrics(p, y, horizons=(1, 2))
    check("horizon metrics indexed 1-based", np.isclose(hm["h1"]["mae"], 2.0)
          and np.isclose(hm["h2"]["mae"], 3.0))
    lt = masked_mae_loss(torch.tensor(p), torch.tensor(y))
    check("torch masked MAE matches numpy", np.isclose(lt.item(), exp_mae))

    # --- data bundle on synthetic data --------------------------------------
    root = tempfile.mkdtemp(prefix="st_test_")
    vals = synthetic_dataset(root)
    data = load_st_benchmark(root, "SYN", batch_size=8, device="cpu")
    batch = next(iter(data.train))
    check("batch shapes", batch["x"].shape == (8, 12, 6, 1) and batch["y"].shape == (8, 12, 6)
          and batch["tod"].shape == (8, 12), str({k: tuple(v.shape) for k, v in batch.items()}))
    check("targets are raw (zeros kept)", float((data.test.t["y"] == 0).sum()) > 0)
    s = int(data.splits["test"][0])
    first = next(iter(data.test))
    check("test window alignment",
          np.allclose(first["y"][0].numpy(), vals[s + 12:s + 24], atol=1e-4)
          and np.allclose(data.scaler.inverse_transform(first["x"][0, :, :, 0]).numpy(),
                          vals[s:s + 12], atol=1e-3))
    tod0 = int(first["tod"][0, 0])
    ts0 = data.timestamps[s]
    check("time-of-day slot at 5-min resolution", tod0 == (ts0.hour * 60 + ts0.minute) // 5)

    # --- history lags ----------------------------------------------------------
    hist = load_st_benchmark(root, "SYN", batch_size=8, device="cpu", history_lags=(288, 576))
    hb = next(iter(hist.test))
    s = int(hist.splits["test"][0])
    lag_day = hist.scaler.inverse_transform(hb["x"][0, :, :, 1]).numpy()
    lag_2d = hist.scaler.inverse_transform(hb["x"][0, :, :, 2]).numpy()
    check("history lag channels align with target times",
          hb["x"].shape[-1] == 3 and hist.input_dim == 3
          and np.allclose(lag_day, vals[s + 12 - 288:s + 24 - 288], atol=1e-3)
          and np.allclose(lag_2d, vals[s + 12 - 576:s + 24 - 576], atol=1e-3))
    check("history lags drop only early train windows",
          hist.test.num_samples == data.test.num_samples and hist.val.num_samples == data.val.num_samples
          and hist.train.num_samples == data.train.num_samples - (576 - 12),
          f"train {data.train.num_samples} -> {hist.train.num_samples}")
    m_hist = STTransformer(num_nodes=6, input_dim=3, num_temporal_layers=1, num_spatial_layers=1)
    check("STTransformer accepts lag channels",
          m_hist(hb["x"], hb["tod"], hb["dow"]).shape == (hb["x"].shape[0], 12, 6))

    # --- models --------------------------------------------------------------
    model = STTransformer(num_nodes=6, adj=data.adj, graph_bias=True, exo_dim=3,
                          exo_embedding_dim=8, num_temporal_layers=1, num_spatial_layers=1)
    exo = torch.randn(8, 12, 3)
    out = model(batch["x"], batch["tod"], batch["dow"], exo)
    check("STTransformer output shape", out.shape == (8, 12, 6))
    gb = model.graph_bias()
    check("graph bias zero at init and shaped (H, N, N)", gb.shape == (4, 6, 6) and gb.abs().sum() == 0)
    hops = hop_distance_matrix(data.adj, max_hops=6)
    check("hop distances on a chain", hops[0, 1] == 1 and hops[0, 3] == 3 and hops[3, 0] == 7)
    out.mean().backward()
    check("graph bias receives gradients", model.graph_bias.fwd.weight.grad is not None
          and model.graph_bias.fwd.weight.grad.abs().sum() > 0)
    base = STTransformer(num_nodes=6)
    n_base = sum(p.numel() for p in base.parameters())
    check("base model_dim = 152 (STAEformer config)", base.model_dim == 152, f"params {n_base:,}")
    # transfer to a different network: only sensor-specific parameters change
    with torch.no_grad():
        model.graph_bias.fwd.weight.fill_(0.5)
    w_before = model.temporal_layers[0].attn.qkv.weight.clone()
    adj9 = np.eye(9, dtype=np.float32) + np.eye(9, k=1, dtype=np.float32)
    model.adapt_to_graph(9, adj=adj9, freeze_shared=True)
    xb9 = torch.randn(2, 12, 9, 1)
    out9 = model(xb9, batch["tod"][:2], batch["dow"][:2], torch.randn(2, 12, 3))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    check("adapt_to_graph: new node count, shared weights kept, bias tables carried over",
          out9.shape == (2, 12, 9) and torch.equal(model.temporal_layers[0].attn.qkv.weight, w_before)
          and float(model.graph_bias.fwd.weight.mean()) == 0.5
          and trainable == {"adaptive_embedding", "graph_bias.fwd.weight", "graph_bias.bwd.weight"},
          f"trainable: {sorted(trainable)}")
    legacy = LegacyTransformerAdapter(num_nodes=6, d_model=32, nhead=4, num_layers=1, dim_feedforward=64)
    check("legacy adapter output shape", legacy(batch["x"], batch["tod"], batch["dow"]).shape == (8, 12, 6))

    # --- baselines -------------------------------------------------------
    res = naive_baselines(data)
    check("naive baselines finite", all(np.isfinite(r["all"]["mae"]) for r in res.values()),
          f"persistence {res['persistence']['all']['mae']:.3f}, HA {res['historical_average']['all']['mae']:.3f}")

    # --- real data checks ---------------------------------------------------
    if data_dir and os.path.exists(os.path.join(data_dir, "clean_weather_data_pems_bay.csv")):
        w = pd.read_csv(os.path.join(data_dir, "clean_weather_data_pems_bay.csv"), parse_dates=["datetime"])
        ts = pd.date_range("2017-01-02", "2017-04-30", freq="5min")
        arr, names = load_weather_features(os.path.join(data_dir, "clean_weather_data_pems_bay.csv"),
                                           ts, fit_end=len(ts))
        temp = pd.Series(arr[:, names.index("temperature")], index=ts)
        peak = int(temp.groupby(temp.index.hour).mean().idxmax())
        raw_peak = int(w.set_index("datetime")["temperature"].groupby(lambda t: t.hour).mean().idxmax())
        check("weather timezone fix moves temperature peak to the afternoon",
              12 <= peak <= 17, f"raw file peak {raw_peak}:00 -> aligned peak {peak}:00 local")

    if data_dir and os.path.exists(os.path.join(data_dir, "weather_PEMS-BAY_era5_local.csv")):
        ts2 = pd.date_range("2017-01-01", "2017-06-30 23:55", freq="5min")
        arr2, names2 = load_weather_features(
            os.path.join(data_dir, "weather_PEMS-BAY_era5_local.csv"), ts2, fit_end=len(ts2),
            traffic_tz="US/Pacific", weather_tz="US/Pacific")
        temp2 = pd.Series(arr2[:, names2.index("temperature")], index=ts2)
        peak2 = int(temp2.groupby(temp2.index.hour).mean().idxmax())
        cover2 = float(arr2[:, names2.index("weather_available")].mean())
        check("rebuilt weather is local time with full coverage",
              13 <= peak2 <= 17 and cover2 > 0.999,
              f"warmest hour {peak2}:00, coverage {cover2:.1%}, {len(names2)} features")

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    return all(RESULTS)


if __name__ == "__main__":
    ok = main(sys.argv[1] if len(sys.argv) > 1 else None)
    sys.exit(0 if ok else 1)
