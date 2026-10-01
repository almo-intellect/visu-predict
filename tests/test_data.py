import os
import warnings

import numpy as np
import pandas as pd
import pytest

from visu_predict.data import (
    dcrnn_window_splits,
    load_adjacency,
    load_st_benchmark,
    load_weather_features,
)


@pytest.mark.parametrize(("timesteps", "sizes"), [
    (52116, (36465, 5209, 10419)),     # PEMS-BAY
    (34272, (23974, 3425, 6850)),      # METR-LA
])
def test_splits_match_the_literature(timesteps, sizes):
    sp = dcrnn_window_splits(timesteps, 12, 12)
    assert (len(sp["train"]), len(sp["val"]), len(sp["test"])) == sizes


def test_batches_have_the_documented_shapes(syn):
    root, _ = syn
    data = load_st_benchmark(root, "SYN", batch_size=8)
    batch = next(iter(data.train))
    assert batch["x"].shape == (8, 12, 6, 1)
    assert batch["y"].shape == (8, 12, 6)
    assert batch["tod"].shape == (8, 12) and batch["dow"].shape == (8, 12)
    assert data.adj is not None and data.adj.shape == (6, 6)
    assert data.sensor_ids == [f"s{i}" for i in range(6)]


def test_targets_stay_raw_and_windows_align(syn):
    root, vals = syn
    data = load_st_benchmark(root, "SYN", batch_size=8)
    assert float((data.test.t["y"] == 0).sum()) > 0          # zeros kept (masked later)
    s = int(data.splits["test"][0])
    first = next(iter(data.test))
    assert np.allclose(first["y"][0].numpy(), vals[s + 12:s + 24], atol=1e-4)
    assert np.allclose(data.scaler.inverse_transform(first["x"][0, :, :, 0]).numpy(), vals[s:s + 12], atol=1e-3)
    ts0 = data.timestamps[s]
    assert int(first["tod"][0, 0]) == (ts0.hour * 60 + ts0.minute) // 5


def test_scaler_uses_training_period_only(syn):
    root, vals = syn
    data = load_st_benchmark(root, "SYN")
    fit_end = int(data.splits["train"][-1]) + 12
    assert np.isclose(data.scaler.mean, vals[:fit_end].astype(np.float32).mean(), atol=1e-3)


def test_history_lags_align_with_target_times(syn):
    root, vals = syn
    base = load_st_benchmark(root, "SYN", batch_size=8)
    hist = load_st_benchmark(root, "SYN", batch_size=8, history_lags=(288, 576))
    hb = next(iter(hist.test))
    s = int(hist.splits["test"][0])
    lag_day = hist.scaler.inverse_transform(hb["x"][0, :, :, 1]).numpy()
    lag_2d = hist.scaler.inverse_transform(hb["x"][0, :, :, 2]).numpy()
    assert hb["x"].shape[-1] == 3 and hist.input_dim == 3
    assert np.allclose(lag_day, vals[s + 12 - 288:s + 24 - 288], atol=1e-3)
    assert np.allclose(lag_2d, vals[s + 12 - 576:s + 24 - 576], atol=1e-3)
    # only early training windows are dropped; val/test are unchanged
    assert hist.test.num_samples == base.test.num_samples
    assert hist.val.num_samples == base.val.num_samples
    assert hist.train.num_samples == base.train.num_samples - (576 - 12)


def test_history_lags_must_not_leak(syn):
    root, _ = syn
    with pytest.raises(ValueError):
        next(iter(load_st_benchmark(root, "SYN", history_lags=(20,)).train))


def test_missing_dataset_error_explains_how_to_get_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="visu-predict download"):
        load_st_benchmark(str(tmp_path), "METR-LA")


def test_adjacency_formats(tmp_path):
    adj = np.arange(9, dtype=np.float32).reshape(3, 3)
    np.save(tmp_path / "a.npy", adj)
    a, ids = load_adjacency(str(tmp_path / "a.npy"))
    assert np.array_equal(a, adj) and ids == ["0", "1", "2"]
    pd.to_pickle(adj, tmp_path / "b.pkl")
    a, _ = load_adjacency(str(tmp_path / "b.pkl"))
    assert np.array_equal(a, adj)
    pd.to_pickle([["x", "y", "z"], {"x": 0, "y": 1, "z": 2}, adj], tmp_path / "c.pkl")
    a, ids = load_adjacency(str(tmp_path / "c.pkl"))
    assert np.array_equal(a, adj) and ids == ["x", "y", "z"]


def _hourly_weather(start, periods, peak_hour):
    t = pd.date_range(start, periods=periods, freq="h")
    temp = 15 + 8 * np.cos(2 * np.pi * (t.hour - peak_hour) / 24)
    return pd.DataFrame({"datetime": t, "temperature": temp, "hourly_precipitation": 0.0,
                         "weather_condition": "Fair"})


def test_weather_is_converted_to_traffic_time(tmp_path):
    # a file stamped in UTC whose afternoon peak shows up at 22:00 UTC
    path = str(tmp_path / "weather_utc.csv")
    _hourly_weather("2024-01-01", 24 * 20, peak_hour=22).to_csv(path, index=False)
    ts = pd.date_range("2024-01-02", "2024-01-15", freq="5min")
    arr, names = load_weather_features(path, ts, fit_end=len(ts), traffic_tz="US/Pacific", weather_tz="UTC")
    temp = pd.Series(arr[:, names.index("temperature")], index=ts)
    assert int(temp.groupby(temp.index.hour).mean().idxmax()) == 14    # 22:00 UTC = 14:00 PST
    assert names[-1] == "weather_available" and arr[:, -1].min() == 1.0
    assert {"is_rain", "is_fog", "is_cloudy"} <= set(names)


def test_custom_dataset_weather_needs_no_timezone(make_dataset):
    root, _ = make_dataset()
    _hourly_weather("2024-01-01", 24 * 8, peak_hour=15).to_csv(
        os.path.join(root, "weather_SYN_era5_local.csv"), index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)      # no fallback / missing-file warnings
        data = load_st_benchmark(root, "SYN", use_weather=True)
    assert data.exo_dim == len(data.exo_names) and "temperature" in data.exo_names
    assert data.exo_dim > 0 and "exo" in next(iter(data.train))


def test_real_data_matches_the_protocol(real_data_dir):
    for name, nodes, test_windows in (("METR-LA", 207, 6850), ("PEMS-BAY", 325, 10419)):
        if not os.path.exists(os.path.join(real_data_dir, f"{name}.csv")):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # adjacency must load and match the CSV sensor order
            data = load_st_benchmark(real_data_dir, name)
        assert data.num_nodes == nodes and data.test.num_samples == test_windows
        assert data.adj is not None and data.adj.shape == (nodes, nodes)
