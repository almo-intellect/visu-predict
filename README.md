# VISU Predict

**English** · [Português (Portugal)](README.pt-PT.md)

[![CI](https://github.com/almo-intellect/visu-predict/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/almo-intellect/visu-predict/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/almo-intellect/visu-predict/blob/main/notebooks/benchmark_colab.ipynb)

**Traffic forecasting for road sensor networks.** VISU Predict reads the last hour of
readings from every sensor (12 five-minute steps) and forecasts the next hour for all of
them at once. It uses a node-level spatio-temporal Transformer, V19 of the VISU traffic model.

The repository includes:

- **The standard METR-LA / PEMS-BAY protocol**, so scores compare directly with
  published results.
- **The tools to reproduce them:** data download, training recipes, seed statistics and
  ensembles.

## Results at a glance

Test MAE in mph at 15 / 30 / 60 minutes (lower is better). V19 rows are means over three
training seeds.

| Model | METR-LA | PEMS-BAY |
|---|---|---|
| **V19** | **2.70 / 3.01 / 3.38** | **1.35 / 1.65 / 1.90** |
| V19 + day/week history lags | 2.73 / 3.07 / 3.46 | 1.33 / 1.58 / 1.80 |
| V19, ensemble of 3 seeds | 2.66 / 2.97 / 3.33 | 1.32 / 1.61 / 1.85 |
| STAEformer (CIKM 2023) | 2.65 / 2.97 / 3.34 | 1.31 / 1.62 / 1.88 |
| Graph WaveNet (IJCAI 2019) | 2.69 / 3.07 / 3.53 | 1.30 / 1.63 / 1.95 |
| DCRNN (ICLR 2018) | 2.77 / 3.15 / 3.60 | 1.38 / 1.74 / 2.07 |
| V18 (previous VISU model) | 3.64 / 3.79 / 4.11 | 2.15 / 2.24 / 2.39 |
| Repeat the last reading | 4.02 / 5.09 / 6.80 | 1.59 / 2.17 / 3.04 |

- **Against V18:** average error is 22% lower on METR-LA and 30% lower on PEMS-BAY, with
  about half the parameters.
- **Against the field:** a few percent behind the best peer-reviewed models.

Full tables (RMSE, MAPE, seed spread), the comparison with published models, the METR-LA
ranking and the ablations are in **[docs/results.md](docs/results.md)**.

## Quick start

```bash
pip install "visu-predict @ git+https://github.com/almo-intellect/visu-predict"

visu-predict download                       # METR-LA + PEMS-BAY (160 MB) into ./data
visu-predict baselines --dataset METR-LA    # sanity check: persistence scores 5.14 average MAE
visu-predict train --dataset METR-LA        # train and test V19; results in ./runs/<run-name>/
```

- **GPU.** Training uses the GPU automatically. Install a CUDA build of PyTorch first if
  needed (see [pytorch.org](https://pytorch.org/get-started/locally/)). On an A100 or L4,
  add `--precision bf16 --compile` for the fastest training.
- **CPU.** CPU works for the tests and for trying things out, but a full run needs a GPU
  (about one to a few hours).
- **No local GPU?** Use the [Colab notebook](notebooks/benchmark_colab.ipynb).

Each run writes to `runs/<run-name>/`:

| File | Content |
|---|---|
| `results.json` | test MAE / RMSE / MAPE per horizon, training options, timings |
| `train_log.txt`, `history.json` | per-epoch log (flushed each epoch, so it can be watched on Google Drive) |
| `best.pt` | best weights, self-contained (model configuration and scaler included) |
| `last.pt` | full training state, used by `--resume` |
| `test_predictions.npz` | with `--save-predictions`; needed by `visu-predict ensemble` |

## Commands

| Command | What it does |
|---|---|
| `visu-predict download` | Fetch the METR-LA / PEMS-BAY input files ([DATA.md](DATA.md)) |
| `visu-predict baselines --dataset D` | Score persistence and the historical average |
| `visu-predict train --dataset D` | Train with early stopping and report test metrics |
| `visu-predict evaluate runs/<run>` | Re-score a trained run, optionally saving its predictions |
| `visu-predict aggregate runs/` | Mean ± standard deviation over runs that differ only by seed |
| `visu-predict ensemble runs/a runs/b ...` | Score the average of several runs' predictions |
| `visu-predict queue --queue file.json` | Run many trainings, a few at a time on one GPU, resuming interrupted ones |
| `visu-predict weather` | Build hourly ERA5 weather files for a dataset |

Every command has `--help`. Common `train` options:

| Option | Purpose |
|---|---|
| `--seed 43` | another training seed |
| `--history-lags 288 2016` | add the readings from the same time yesterday and last week |
| `--weather`, `--holidays` | extra inputs |
| `--graph-bias` | road-graph attention prior |
| `--model legacy` | train V18 under the same protocol |
| `--epochs`, `--lr` | override the training recipe |

## Your own data

Put `MYCITY.csv` (first column: timestamps; then one column per sensor) in `data/`, then:

```bash
visu-predict baselines --dataset MYCITY
visu-predict train --dataset MYCITY
```

Readings of `0` count as missing. An adjacency matrix (`adj_MYCITY.pkl` / `.npy`) is only
needed for `--graph-bias`. To add weather, give the network's location:
`visu-predict weather --datasets MYCITY --lat -25.97 --lon 32.57 --tz Africa/Maputo`.

To start from a model trained on a large network, see
[transfer to another network](docs/architecture.md#transfer-to-another-network). All file
formats are in [DATA.md](DATA.md).

## Python API

```python
import torch
from visu_predict import STTrainConfig, STTransformer, fit, load_checkpoint, load_st_benchmark

data = load_st_benchmark("data", "METR-LA", batch_size=16, device="cuda")
model = STTransformer(num_nodes=data.num_nodes, steps_per_day=data.steps_per_day)
results = fit(model, data, STTrainConfig(max_epochs=200, precision="bf16"),
              run_dir="runs/metr-la", device="cuda")
print(results["test"]["h12"])          # 60-minute MAE / RMSE / MAPE

# later: rebuild the trained model and forecast one batch
model, scaler, _ = load_checkpoint("runs/metr-la/best.pt", device="cuda")
batch = next(iter(data.test))          # x: (B, 12, N, 1) scaled readings, tod / dow: calendar indices
with torch.no_grad():
    forecast = scaler.inverse_transform(model(batch["x"], batch["tod"], batch["dow"]))   # (B, 12, N) mph
```

## How it works

```mermaid
flowchart LR
    A["Last 12 readings<br/>of every sensor"] --> B["One token per<br/>(time step, sensor)"]
    B --> C["Temporal attention<br/>within each sensor"]
    C --> D["Spatial attention<br/>across sensors"]
    D --> E["Per-sensor head:<br/>next 12 steps"]
```

Each (time step, sensor) token combines the reading with time-of-day and day-of-week
embeddings and a learned per-sensor embedding. Three temporal and three spatial attention
layers follow (STAEformer design, 1.26 M parameters on METR-LA).

Training uses the standard protocol: chronological 70/10/20 split, masked MAE on raw
speeds, test windows identical to the literature. Details are in
[docs/architecture.md](docs/architecture.md).

## Reproducing the published numbers

```bash
visu-predict download
visu-predict queue --queue configs/paper_runs.json --max-concurrent 3   # all 17 runs
visu-predict aggregate runs/
visu-predict ensemble runs/METR-LA_st_base runs/METR-LA_st_base_s43 runs/METR-LA_st_base_s44
```

The queue skips finished runs and resumes interrupted ones, so it can simply be restarted
after a Colab disconnect. The metrics of the published runs are in [`results/`](results).

## Project layout

```
src/visu_predict/
├── data.py          # benchmark data pipeline: windows, splits, scaling, calendar, weather
├── model.py         # STTransformer (+ graph prior, transfer to other networks)
├── training.py      # training loop, early stopping, checkpoints, evaluation, baselines
├── metrics.py       # masked MAE / RMSE / MAPE per horizon
├── benchmark.py     # train / baselines / evaluate commands
├── analysis.py      # seed statistics and ensembles
├── job_queue.py     # multi-run queue
├── weather.py       # ERA5 weather builder
├── download.py      # dataset download
├── cli.py           # `visu-predict` command
└── legacy/          # V18 TrafficTransformer (pip install "visu-predict[legacy]")
configs/paper_runs.json   # every run behind docs/results.md
notebooks/                # Colab notebook
results/                  # metrics of the published runs
docs/                     # results and architecture
tests/                    # pytest suite (CPU, about 30 s)
```

## Versions

- **`main`**: V19, package version 0.2.0 ([CHANGELOG](CHANGELOG.md)).
- **Tag [`v0.1.0`](https://github.com/almo-intellect/visu-predict/tree/v0.1.0)**: the
  previous package (V14 encoder-decoder Transformer with optional GNN pre-encoder).
- **`COLAB_NOBASELINE_V*` branches**: snapshots of the Colab working folders. V19 there is
  the same model as `main`, in its original file layout.
- **`sota-upgrade`**: experimental extensions of the 0.1.0 package (mixture of experts,
  Mamba, patching, pre-training), not benchmarked under this protocol.

## Development

```bash
git clone https://github.com/almo-intellect/visu-predict && cd visu-predict
pip install -e ".[dev,legacy]"
pytest                 # about 30 s on CPU; VISU_DATA_DIR=<data folder> adds the real-data checks
ruff check src tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, © Almo Intellect. See [LICENSE](LICENSE).

## Authors

- Lauro Mota (`lauro.mota@almo.co.mz`), primary author
