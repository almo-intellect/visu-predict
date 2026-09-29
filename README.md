# V19 — Node-level spatio-temporal transformer + benchmark protocol

> **About this branch.** `COLAB_NOBASELINE_V19` is a snapshot of the V19 Colab working folder, in the same
> style as the other `COLAB_NOBASELINE_*` branches: it builds on the V18 `traffic_transformer` package, not on
> the restructured `src/visu_predict` package on `main` / `sota-upgrade`. The benchmark protocol, masked
> metrics, rebuilt weather inputs and 3-seed results here are independent of that line of work.
>
> **Data is not versioned.** METR-LA / PEMS-BAY (`<dataset>.csv` + `adj_<dataset>.pkl`) go in an input
> folder passed with `--input-dir` (see `DATA.md` on `main` for sources). Rebuild the weather inputs with
> `python fix_weather.py <input-dir>`. Aggregated results are in `results/`.

V19 keeps the whole V18 package (legacy `TrafficTransformer`, feature attention,
weather, transfer learning, reports — all 34 V18 smoke checks still pass) and adds
a standard-protocol benchmark pipeline plus a new model that targets
state-of-the-art accuracy on METR-LA / PEMS-BAY.

## Why a new architecture

Under the standard protocol the V18 model scores **2.25 average MAE on PEMS-BAY**
(15/30/60 min: 2.15 / 2.24 / 2.40). Simply repeating the last observation scores
1.59 / 2.18 / 3.04 (avg 2.17), and published SOTA is ~1.52. The cause is structural:
V18 embeds all 325 sensors of a timestep into one token, attends over 12 time
tokens, mean-pools and maps a single vector to all 12 x 325 outputs, so it has no
sensor identity, no weight sharing across sensors and no spatial reasoning; it
overfits (train MAE 1.33 vs val 2.20) and cannot track current conditions per
sensor. More capacity made it worse (38M params: val 0.037 vs 0.035 at 2.8M).

## Results (Sep 2026, Colab A100; V19 rows are means of 3 seeds)

Test MAE in mph at 15 / 30 / 60 min and averaged over all 12 steps (masked, standard protocol).
Full tables with RMSE/MAPE: `results/benchmark/*/results.json`; report: https://claude.ai/artifact/CxGXZmPWdp2KxqwGsdJ3A9

| Model | PEMS-BAY 15/30/60 | PEMS-BAY avg | METR-LA 15/30/60 | METR-LA avg |
|---|---|---|---|---|
| **V19 STTransformer** (3 seeds) | 1.35 / 1.65 / 1.90 | **1.583 ± 0.027** | 2.70 / 3.01 / 3.38 | **2.979 ± 0.023** |
| V19 + day/week history lags (3 seeds) | 1.33 / 1.58 / 1.80 | 1.525 ± 0.010 | 2.73 / 3.07 / 3.46 | 3.035 ± 0.045 |
| V19 + road-graph prior | 1.37 / 1.69 / 1.97 | 1.63 | 2.66 / 2.99 / 3.39 | 2.97 |
| STAEformer (published) | 1.31 / 1.62 / 1.88 | 1.53 | 2.65 / 2.97 / 3.34 | 2.93 |
| V18 legacy (same protocol) | 2.15 / 2.24 / 2.39 | 2.25 | 3.64 / 3.79 / 4.11 | 3.82 |
| Persistence (repeat last reading) | 1.59 / 2.18 / 3.04 | 2.17 | 4.02 / 5.09 / 6.80 | 5.14 |

| V19 + weather (rebuilt data) | 1.34 / 1.64 / 1.89 | 1.57 | 2.73 / 3.06 / 3.51 | 3.04 |

Over three seeds V19 lands a few percent behind the best published models (STAEformer 1.53 / 2.93, HimNet 1.51 / 2.92):
ahead of DCRNN, Graph WaveNet and STID on METR-LA, level with them on PEMS-BAY, and ~30% better than V18 with half
its parameters. Seed spread (sd 0.02-0.03 MAE) exceeds several published model-to-model gaps, so single-run
comparisons in this field are not trustworthy - the first V19 PEMS-BAY run (1.552) was the luckiest of three.
History lags are a genuine gain on PEMS-BAY (60-min -0.099 mph, 6.8x the seed spread) and not on METR-LA
(+0.056 average, within noise); the road-graph prior does not help (and is ~20% slower).

**Weather does not help these benchmarks even with correct data** (+1.2% error on PEMS-BAY, +2.5% on METR-LA).
Their test periods are nearly dry (2.7% and 0.0% of hours wet, against 21.6% and 5.9% in training) and temperature
drifts 0.8-1.4 SD past the training range, so weather inputs are extrapolation with nothing to explain. On the 398
rainy PEMS-BAY test windows the base model still wins (1.37 vs 1.42 mph). Study weather where adverse weather falls
in the test period (Maputo's rainy season, or a winter split).

## What is new

| File | Content |
|---|---|
| `traffic_transformer/st_data.py` | DCRNN-convention windows + 70/10/20 split, raw targets (zeros kept & masked), train-only z-score, 5-min time-of-day and day-of-week indices (+ holiday day type), timezone-corrected weather with availability flag, optional history lags, GPU-resident batching |
| `traffic_transformer/metrics.py` | masked MAE / RMSE / MAPE per horizon (15/30/60 min) and average — directly comparable with published tables |
| `traffic_transformer/st_model.py` | `STTransformer`: every (timestep, sensor) is a token = value + time-of-day + day-type + spatio-temporal adaptive embedding (+ exogenous); temporal attention per sensor, spatial attention across sensors, mixed output head (STAEformer family). Extensions: `graph_bias` (learned per-head bias by road-graph hop distance), `history_lags`, `adapt_to_graph()` for transfer to a new network. `LegacyTransformerAdapter` runs V18 in the same harness |
| `traffic_transformer/st_training.py` | masked-MAE training on the original scale, Adam + step decay, early stopping, bf16/tf32, `torch.compile`, resumable checkpoints (`last.pt`), per-epoch flushed logs |
| `run_benchmark.py` | CLI for one experiment (`--model st_transformer | legacy | baselines`) |
| `queue_runner.py` | runs experiments listed in `results/queue.json` a few at a time on one GPU; re-reads the queue, skips finished runs, resumes interrupted ones |
| `benchmark_notebook.ipynb` | Colab entry point (mount Drive, copy code/data to SSD, run queue) |
| `test_st_pipeline.py` | 21 CPU checks (splits, masking, alignment, shapes, graph bias, lags, transfer, weather tz) |
| `speed_test.py`, `colab_speedtest.sh` | throughput check for precision / compile settings |

## Running

```bash
# tests (CPU, < 1 min)
python test_st_pipeline.py <path/to/Transformers_Input>

# naive baselines (sanity check of the protocol)
python run_benchmark.py --dataset PEMS-BAY --model baselines --input-dir <data>

# new model, STAEformer recipe (Adam 1e-3, step decay, batch 16)
python run_benchmark.py --dataset PEMS-BAY --model st_transformer --precision bf16 --compile --input-dir <data>
#   + road-graph prior         --graph-bias
#   + day / week history lags   --history-lags 288 2016
#   + weather / holidays        --weather --holidays

# legacy V18 model under the same protocol
python run_benchmark.py --dataset PEMS-BAY --model legacy --batch-size 32 --input-dir <data>
```

On Colab: open `benchmark_notebook.ipynb`, run cell 1 (setup), then either the
notebook's queue cells or

```
!cd /content/v19 && python -W ignore queue_runner.py --queue "<V19>/results/queue.json" \
    --out "<V19>/results/benchmark" --data /content/data --max-concurrent 3 --common=--save-predictions
```

Each run writes `results/benchmark/<run>/{train_log.txt, history.json, results.json,
best.pt, last.pt, test_predictions.npz}`.

## Speed notes (Colab L4, PEMS-BAY, batch 16)

tf32 195 ms/step, bf16 109, bf16 + `torch.compile` 87 ms/step (~3.3 min/epoch for one
run). The model is memory-bandwidth bound: an A100 is several times faster. The head
size of the STAEformer configuration (38) is padded to 40 inside attention so PyTorch's
fused kernels are used (identical results, see `MultiHeadSelfAttention`).

## Data caveats (important for the paper)

- **Weather was rebuilt on 2026-09-19** (`fix_weather.py`). The original exports
  (`clean_weather_data_*.csv`) are stamped in UTC while traffic is local time, PEMS-BAY
  weather ends 2017-05-31 although the test period runs to 06-30 (only ~15% covered),
  hourly values were forward-filled for up to 72 h, and METR-LA precipitation never drops
  below 0.01 with no rain labels - so **V18 weather results are not trustworthy**.
  `weather_<dataset>_era5_local.csv` replaces them: hourly ERA5 reanalysis for the network
  centre, already in local wall-clock time, covering 100% of each traffic period with no
  missing values, 12 features (temperature, precipitation, humidity, dew point, wind,
  gusts, cloud cover, pressure, rain/fog/cloud flags). The old files stay as a fallback.
  Cross-check: the old series matches the rebuilt one best when shifted -7 h (PEMS-BAY,
  r=0.77) and -6 h (METR-LA, r=0.90) versus r=0.28 / 0.37 unshifted.
- `METR-MPT-V1.csv` (Maputo) has only 18 timesteps — enough to test the transfer code
  path (`STTransformer.adapt_to_graph`), not to train or evaluate.
