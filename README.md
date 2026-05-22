# visu-predict

**English** · [Português (Portugal)](README.pt-PT.md)

Transformer-based traffic forecasting toolkit. Predicts short-horizon traffic
speed/flow for sensor networks (METR-LA, PEMS-BAY, PEMS-03/04/07/08, custom
datasets) using an encoder–decoder Transformer with optional graph neural
network pre-processing, feature-wise attention over time / weather / spatial /
lagged features, and a transfer-learning path for low-resource targets.

> Originally developed as a Google Colab research notebook
> (`VISU_Traffic_Transformer_*.ipynb`). This repository is a refactor of that
> notebook into a maintainable Python package.

## Features

- **Encoder–decoder Transformer** with linear, MLP, or autoregressive
  Transformer decoder.
- **Feature-wise attention**: separate embeddings + transformer per feature
  group (traffic, time-of-day, holidays, weather, lagged, spatial), fused
  with context-adaptive gating and cross-attention.
- **Optional GNN pre-encoder** (GCN or GAT, via `torch-geometric`).
- **Spatial bias** in encoder self-attention from sensor coordinates /
  adjacency.
- **Weather integration** with nearest-time alignment to traffic timestamps.
- **Transfer learning**: load a pre-trained checkpoint, freeze first N encoder
  layers, fine-tune at a lower learning rate.
- **Mixed precision** training with gradient accumulation.

## Installation

```bash
pip install -e .
# Optional extras
pip install -e ".[gnn,holidays,dev]"
```

Requires Python ≥ 3.10 and a recent PyTorch (2.x).

## Quickstart

1. Place a traffic CSV (timestamp index, one column per sensor) somewhere
   accessible.
2. Copy `configs/example.yaml` and edit at minimum:
   - `base_output_dir` — where checkpoints and plots are written.
   - `dataset_name` — used to locate adjacency / coordinate files in
     `<base_output_dir>/inputs/`.
3. Train:

```bash
visu-predict train --config configs/example.yaml --data path/to/traffic.csv
```

Or programmatically:

```python
from visu_predict import load_config
from visu_predict.runner import run_training

config = load_config("configs/example.yaml")
result = run_training(config, data_path="path/to/traffic.csv")
# result: dict with 'model', 'train_losses', 'val_losses', 'predictions', 'actuals'
```

## Configuration

All knobs live in `TrainingConfig` (`src/visu_predict/config.py`).
See `configs/example.yaml` for an annotated starting point. Highlights:

| Key | Default | Notes |
|-----|---------|-------|
| `dataset_name` | `METR-LA` | Used by `paths.find_adjacency_matrix` / `find_coordinates` |
| `seq_length` / `pred_length` | `12` / `12` | Input window and forecast horizon (steps) |
| `hidden_dim` / `num_heads` | `336` / `16` | Auto-adjusted to be even and divisible |
| `decoder_type` | `linear` | `linear`, `mlp`, or `transformer` (autoregressive) |
| `use_gnn_pre_transformer` | `false` | Requires `torch-geometric` |
| `use_weather_feature` | `false` | Provide `weather_data_file` |
| `enable_transfer_learning` | `false` | Provide `source_model_path` |

## Repository layout

```
src/visu_predict/
├── config.py              # TrainingConfig dataclass + YAML loader
├── paths.py               # Adjacency / coordinate file discovery
├── data.py                # TrafficDataset, prepare_data
├── features/
│   ├── weather.py         # WeatherIntegration
│   └── spatial.py         # SpatialIntegration, adjacency utilities
├── models/
│   ├── transformer.py     # TrafficTransformer
│   ├── attention.py       # FeatureAttention
│   ├── positional.py      # PositionalEncoding
│   ├── gnn.py             # GCNEncoder (optional)
│   └── lr.py              # CosineWarmupLR
├── training/
│   ├── train.py           # train / evaluate / predict
│   ├── losses.py          # quantile, hybrid, robust MAPE
│   └── transfer.py        # checkpoint loading + freezing
├── utils/                 # logging, GPU info, seeding
├── viz.py                 # training-curve and prediction plots
├── runner.py              # End-to-end pipeline
└── cli.py                 # `visu-predict train ...`
```

## Differences from the original notebook

The refactor preserves the model architecture and training behaviour while
cleaning up:

- Removed all Colab-specific code (`drive.mount`, hardcoded Google Drive paths,
  inline `!pip install`).
- Replaced the `TeeLogger` stdout-hijack with the standard `logging` module.
- Consolidated two parallel weather-loading implementations into one
  (`features/weather.py`).
- Replaced the singleton `SpatialIntegration` with a regular class.
- Removed the embedded LSTM baseline and the in-notebook PDF report generator
  (can be reintroduced as a separate package if needed).
- Added type hints, validation in `TrainingConfig.__post_init__`, a CLI, and
  a CI workflow.

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

## License

MIT, © Almo Intellect. See [LICENSE](LICENSE).

## Authors

- Lauro Mota — `lauro.mota@almo.co.mz` (primary author)
