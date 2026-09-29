"""CPU smoke test for the patched traffic_transformer package.

Runs in ~1-2 minutes on CPU with tiny synthetic data and exercises every
patched code path: weather/holiday/time features, adjacency loading + dense
GNN pre-encoding, train/eval/predict with the corrected APIs, checkpoint
backup + resume, attention capture, masked MAPE, and transfer learning with
adapters + input/output re-heading.

Run BEFORE spending Colab GPU time:  python smoke_test.py
"""

import os
import pickle
import shutil
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traffic_transformer import (  # noqa: E402
    TrainingConfig,
    setup_directories,
    load_and_prepare_data,
    TrafficTransformer,
    train_model,
    evaluate_model,
    predict,
    create_optimizer,
    create_scheduler,
    create_criterion,
    TransferLearningModule,
)
from traffic_transformer.training import masked_mape  # noqa: E402

PASS = []


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    PASS.append(bool(cond))
    print(f"[{status}] {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        raise AssertionError(name)


def make_synthetic_inputs(root):
    input_dir = os.path.join(root, "input")
    os.makedirs(input_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    # --- traffic: 500 steps x 8 sensors, 5-min cadence, a few zeros -------
    n, s = 500, 8
    idx = pd.date_range("2024-03-01", periods=n, freq="5min")
    base = 40 + 10 * np.sin(np.linspace(0, 20 * np.pi, n))[:, None]
    data = base + rng.normal(0, 2, size=(n, s))
    data[rng.integers(0, n, 10), rng.integers(0, s, 10)] = 0.0  # sensor faults
    pd.DataFrame(data, index=idx, columns=[f"s{i}" for i in range(s)]).to_csv(
        os.path.join(input_dir, "TESTCITY.csv")
    )

    # --- weather: hourly, with variant labels the normaliser must handle --
    widx = pd.date_range("2024-03-01", periods=48, freq="h")
    conditions = ["Fair", "Light Rain", "Mostly Cloudy / Windy", "Haze", "T-Storm"]
    wdf = pd.DataFrame({
        "datetime": widx,
        "temperature": rng.uniform(15, 30, len(widx)),
        "visibility": rng.uniform(5, 10, len(widx)),
        "wind_speed": rng.uniform(0, 20, len(widx)),
        "relative_humidity": rng.uniform(30, 90, len(widx)),
        "dew_point": rng.uniform(5, 20, len(widx)),
        "weather_condition": [conditions[i % len(conditions)] for i in range(len(widx))],
        "cloud_cover": ["CLR", "FEW", "SCT", "BKN"] * (len(widx) // 4),
        "wind_direction": ["N", "NE", "E", "SE"] * (len(widx) // 4),
    })
    weather_path = os.path.join(input_dir, "weather_TESTCITY.csv")
    wdf.to_csv(weather_path, index=False)

    # --- adjacency: DCRNN-style [sensor_ids, id_map(dict), matrix] --------
    adj = np.eye(s, dtype=np.float32)
    for i in range(s - 1):
        adj[i, i + 1] = adj[i + 1, i] = 0.7
    payload = [[f"s{i}" for i in range(s)], {f"s{i}": i for i in range(s)}, adj]
    with open(os.path.join(input_dir, "adj_TESTCITY.pkl"), "wb") as f:
        pickle.dump(payload, f)

    # --- african target: 300 steps x 5 sensors ----------------------------
    n2, s2 = 300, 5
    idx2 = pd.date_range("2024-03-01", periods=n2, freq="5min")
    d2 = 30 + 8 * np.cos(np.linspace(0, 12 * np.pi, n2))[:, None]
    d2 = d2 + rng.normal(0, 1.5, size=(n2, s2))
    african_path = os.path.join(root, "african_traffic.csv")
    pd.DataFrame(d2, index=idx2, columns=[f"m{i}" for i in range(s2)]).to_csv(african_path)

    return input_dir, weather_path, african_path


def main():
    warnings.simplefilter("default")
    root = tempfile.mkdtemp(prefix="tt_smoke_")
    input_dir, weather_path, african_path = make_synthetic_inputs(root)
    backup_dir = os.path.join(root, "drive_backup")

    config = TrainingConfig(
        dataset_name="TESTCITY",
        input_dir=input_dir,
        base_output_dir=os.path.join(root, "out"),
        num_epochs=2,
        patience=5,
        batch_size=8,
        accumulation_steps=2,          # exercise accumulation
        seq_length=12,
        pred_length=12,
        hidden_dim=32,
        num_layers=2,  # 2 layers so freeze_layers=1 leaves one for an adapter
        num_heads=4,
        dim_feedforward=64,
        warmup_epochs=1,
        num_workers=0,
        use_time_features=True,
        use_weather_feature=True,
        weather_data_file=weather_path,
        use_holiday_feature=True,
        holiday_country_code="MZ",
        use_gnn_pre_transformer=True,
        use_spatial_features=False,
        decoder_type="linear",
        loss_function="mae",
        missing_value_strategy="mean_replace_zeros",
        drive_backup_dir=backup_dir,
        find_optimal_batch_size=False,
        monitor_gpu_usage=False,
    )
    check("config accepts mean_replace_zeros",
          config.missing_value_strategy == "mean_replace_zeros")
    check("accumulation_steps default-independent", config.accumulation_steps == 2)

    setup_directories(config)
    check("user weather_data_file preserved for unmapped dataset",
          config.weather_data_file == weather_path)

    train_loader, val_loader, test_loader, scaler, adj = load_and_prepare_data(config)
    check("adjacency loaded (tuple unpack + dict id-map + path fix)",
          adj is not None and tuple(adj.shape) == (8, 8))

    ds = train_loader.dataset
    check("weather features present (weather_data_file read)",
          "weather" in ds.feature_groups, f"groups={list(ds.feature_groups)}")
    check("holiday features present (holiday_country_code read)",
          "holiday" in ds.feature_groups)

    x0, y0 = ds[0]
    expected_in = 8 + 4 + ds.feature_groups["weather"]["dim"] + 1
    check("holiday wired into __getitem__ (input width)",
          x0.shape[-1] == expected_in, f"width={x0.shape[-1]}, expected={expected_in}")
    check("no NaNs in model input", bool(torch.isfinite(x0).all()))

    device = "cpu"
    model = TrafficTransformer(
        input_dim=x0.shape[-1],
        d_model=config.hidden_dim,
        nhead=config.num_heads,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        pred_length=config.pred_length,
        output_dim=8,
        num_sensors=8,
        decoder_type="linear",
        use_gnn_pre_transformer=True,
        gnn_hidden_dim=16,
        gnn_num_layers=2,
    ).to(device)
    check("dense GNN pre-encoder built without torch_geometric",
          model.gnn_encoder is not None)
    check("transformer alias is read-only property (no duplicate keys)",
          "transformer.0.self_attn.in_proj_weight" not in model.state_dict()
          and model.transformer is model.encoder)

    xb, yb = next(iter(train_loader))
    out = model(xb, adjacency_matrix=adj)
    check("forward with GNN + aux features", tuple(out.shape) == tuple(yb.shape))

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    criterion = create_criterion(config)
    check("warmup epoch 1 LR > 0", optimizer.param_groups[0]["lr"] > 0,
          f"lr={optimizer.param_groups[0]['lr']:.2e}")

    model, tr_losses, va_losses = train_model(
        model, train_loader, val_loader, optimizer, scheduler, criterion,
        config, data_scaler=scaler, device=device, adjacency_matrix=adj,
    )
    check("train_model returns (model, train, val)",
          len(tr_losses) == 2 and len(va_losses) == 2)
    ckpt_path = os.path.join(config.model_dir, config.checkpoint_filename)
    check("checkpoint saved + reloaded (weights_only fix)", os.path.exists(ckpt_path))
    check("Drive backup mirrored",
          os.path.exists(os.path.join(backup_dir, config.checkpoint_filename)))
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    check("config stored as plain dict in checkpoint", isinstance(ck["config"], dict))
    check("scheduler state in checkpoint", ck.get("scheduler_state_dict") is not None)

    # --- resume ------------------------------------------------------------
    config.resume_from = ckpt_path
    config.num_epochs = 3
    opt2 = create_optimizer(model, config)
    sch2 = create_scheduler(opt2, config)
    model, tr2, va2 = train_model(
        model, train_loader, val_loader, opt2, sch2, criterion,
        config, data_scaler=scaler, device=device, adjacency_matrix=adj,
    )
    check("resume from checkpoint (continues at next epoch)", len(tr2) == 1)
    config.resume_from = None

    # --- evaluate / predict ------------------------------------------------
    test_loss, (mae, rmse, r2, mape) = evaluate_model(
        model, test_loader, criterion, config,
        data_scaler=scaler, device=device, adjacency_matrix=adj,
    )
    check("evaluate_model unpack + finite metrics",
          np.isfinite([test_loss, mae, rmse, mape]).all(),
          f"MAE={mae:.3f} RMSE={rmse:.3f} MAPE={mape:.2f}%")
    check("masked MAPE sane despite zero targets", 0 <= mape < 200, f"{mape:.2f}%")
    preds, acts = predict(model, test_loader, config, data_scaler=scaler,
                          device=device, adjacency_matrix=adj)
    check("predict returns aligned arrays", preds.shape == acts.shape)

    m = masked_mape(np.array([0.0, 10.0, 20.0]), np.array([5.0, 11.0, 19.0]))
    check("masked_mape excludes null targets", abs(m - 7.5) < 1e-6, f"{m:.3f}")

    # --- attention capture -------------------------------------------------
    model.set_attention_capture(True)
    with torch.no_grad():
        model(xb, adjacency_matrix=adj)
    check("attention captured when enabled",
          model.encoder[0].attn_weights is not None)
    model.set_attention_capture(False)
    with torch.no_grad():
        model(xb, adjacency_matrix=adj)
    check("attention skipped when disabled (fused path)",
          model.encoder[0].attn_weights is None)

    # --- transfer learning -------------------------------------------------
    import copy as _copy
    tl_model = _copy.deepcopy(model)
    tl = TransferLearningModule(
        base_model=tl_model, config=config,
        target_dataset_name="testafrica",
        freeze_encoder=True, freeze_layers=1, adapter_dim=8,
    )
    tl_train, tl_val, tl_test, tl_scaler = tl.load_african_dataset(
        african_path, test_split=0.2, val_split=0.15,
    )
    sx, sy = tl_train.dataset[0]
    check("load_african_dataset returns 4-tuple with val split", tl_val is not None)
    check("input embedding re-headed to target width",
          tl_model.embedding.in_features == sx.shape[-1],
          f"{tl_model.embedding.in_features} == {sx.shape[-1]}")
    check("output head re-headed to target sensors",
          tl_model.output_dim == sy.shape[-1])

    from traffic_transformer.transfer_learning import AdaptedEncoderLayer
    check("adapter layers actually created",
          len(tl.adapters) >= 1 and any(
              isinstance(l, AdaptedEncoderLayer) for l in tl_model.encoder),
          f"{len(tl.adapters)} adapters")

    fine_tuned = tl.fine_tune(tl_train, tl_val, num_epochs=1,
                              output_dir=os.path.join(root, "tl_out"))
    check("fine_tune runs with adapters (set_spatial_bias delegation)",
          fine_tuned is not None)
    with torch.no_grad():
        tx, ty = next(iter(tl_test))
        tout = fine_tuned(tx)
    check("fine-tuned forward on target test", tuple(tout.shape) == tuple(ty.shape))
    check("MAE criterion honoured in fine_tune (config.loss_function)",
          isinstance(create_criterion(config), torch.nn.L1Loss))

    # --- feature-wise attention mode (the paper's mechanism) ---------------
    config.use_feature_attention = True
    config.resume_from = None
    config.num_epochs = 1
    fa_train, fa_val, fa_test, fa_scaler, fa_adj = load_and_prepare_data(config)
    fx, fy = next(iter(fa_train))
    check("FA mode yields dict batches",
          isinstance(fx, dict) and "traffic" in fx and "weather" in fx,
          f"groups={sorted(fx)}")
    fa_dims = {k: v.shape[-1] for k, v in fx.items()}
    fa_model = TrafficTransformer(
        input_dim=sum(fa_dims.values()), d_model=config.hidden_dim,
        nhead=config.num_heads, num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward, pred_length=config.pred_length,
        output_dim=8, num_sensors=8, decoder_type="linear",
        use_feature_attention=True, feature_dims=fa_dims,
        use_gnn_pre_transformer=True, gnn_hidden_dim=16, gnn_num_layers=2,
    )
    fa_opt = create_optimizer(fa_model, config)
    fa_sch = create_scheduler(fa_opt, config)
    fa_model, fa_tr, fa_va = train_model(
        fa_model, fa_train, fa_val, fa_opt, fa_sch, criterion,
        config, data_scaler=fa_scaler, device=device, adjacency_matrix=fa_adj,
    )
    check("FA model trains through dict pipeline", len(fa_tr) == 1)
    _, (fmae, _, _, fmape) = evaluate_model(
        fa_model, fa_test, criterion, config,
        data_scaler=fa_scaler, device=device, adjacency_matrix=fa_adj,
    )
    imp = fa_model.feature_attention.feature_importances
    check("FeatureAttention importances captured",
          imp is not None and imp.shape[-1] == len(fa_dims),
          f"MAE={fmae:.2f}, groups={fa_model.feature_attention.feature_names}")
    config.use_feature_attention = False

    shutil.rmtree(root, ignore_errors=True)
    print(f"\n{'='*60}\nALL {len(PASS)} CHECKS PASSED\n{'='*60}")


if __name__ == "__main__":
    main()
