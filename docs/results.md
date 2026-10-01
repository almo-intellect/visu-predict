# Results

All numbers below use the **test split of the standard protocol**:

- 12 input steps forecast the next 12 steps, at 5-minute resolution.
- Windows are split 70/10/20 in time, following DCRNN.
- Zero readings (sensor faults) are excluded from every metric.
- Errors are in mph.

"15 / 30 / 60 min" are horizons 3, 6 and 12; "average" covers all 12 steps.

V19 rows are the mean ± sample standard deviation over seeds 42, 43 and 44. Ensemble rows
score the average of the same three models' predictions. The runs are listed in
[`configs/paper_runs.json`](../configs/paper_runs.json), and their metrics are in
[`results/`](../results). See [Reproducing](#reproducing) to re-run them.

## METR-LA (207 sensors)

Test MAE:

| Model | 15 min | 30 min | 60 min | Average |
|---|---|---|---|---|
| **V19** | 2.698 ± 0.032 | 3.008 ± 0.020 | 3.382 ± 0.015 | **2.979 ± 0.023** |
| V19, ensemble of the 3 seeds | 2.662 | 2.966 | 3.330 | 2.937 |
| V19 + day/week history lags | 2.732 ± 0.038 | 3.066 ± 0.047 | 3.463 ± 0.052 | 3.035 ± 0.045 |
| V18 (previous VISU model) | 3.639 | 3.788 | 4.115 | 3.822 |
| Historical average (training data) | 4.187 | 4.187 | 4.187 | 4.187 |
| Persistence (repeat the last reading) | 4.017 | 5.094 | 6.795 | 5.140 |

RMSE / MAPE:

| Model | 15 min | 30 min | 60 min | Average |
|---|---|---|---|---|
| **V19** | 5.14 / 6.93% | 6.07 / 8.22% | 7.08 / 9.83% | 6.03 / 8.14% |
| V19, ensemble of the 3 seeds | 5.04 / 6.82% | 5.93 / 8.10% | 6.92 / 9.69% | 5.90 / 8.03% |
| V19 + day/week history lags | 5.23 / 7.03% | 6.17 / 8.39% | 7.20 / 10.04% | 6.13 / 8.31% |
| V18 | 6.84 / 10.18% | 7.22 / 10.73% | 7.95 / 11.96% | 7.30 / 10.86% |
| Persistence | 8.69 / 9.39% | 11.13 / 12.21% | 14.21 / 16.71% | 11.27 / 12.34% |

## PEMS-BAY (325 sensors)

Test MAE:

| Model | 15 min | 30 min | 60 min | Average |
|---|---|---|---|---|
| **V19** | 1.352 ± 0.039 | 1.648 ± 0.027 | 1.902 ± 0.018 | **1.583 ± 0.027** |
| V19, ensemble of the 3 seeds | 1.324 | 1.610 | 1.850 | 1.545 |
| **V19 + day/week history lags** | 1.326 ± 0.011 | 1.584 ± 0.011 | 1.803 ± 0.011 | **1.525 ± 0.010** |
| V19 + history lags, ensemble of the 3 seeds | 1.310 | 1.560 | 1.771 | 1.502 |
| V18 (previous VISU model) | 2.154 | 2.244 | 2.395 | 2.252 |
| Historical average (training data) | 2.632 | 2.631 | 2.629 | 2.631 |
| Persistence (repeat the last reading) | 1.594 | 2.175 | 3.044 | 2.173 |

RMSE / MAPE:

| Model | 15 min | 30 min | 60 min | Average |
|---|---|---|---|---|
| **V19** | 2.81 / 2.84% | 3.69 / 3.70% | 4.34 / 4.46% | 3.58 / 3.54% |
| V19, ensemble of the 3 seeds | 2.76 / 2.78% | 3.61 / 3.61% | 4.23 / 4.35% | 3.50 / 3.46% |
| V19 + day/week history lags | 2.80 / 2.81% | 3.66 / 3.59% | 4.24 / 4.27% | 3.53 / 3.44% |
| V19 + history lags, ensemble of the 3 seeds | 2.76 / 2.76% | 3.58 / 3.54% | 4.14 / 4.20% | 3.46 / 3.39% |
| V18 | 4.34 / 4.97% | 4.55 / 5.20% | 4.89 / 5.60% | 4.57 / 5.22% |
| Persistence | 3.39 / 3.24% | 4.96 / 4.65% | 6.99 / 6.83% | 5.13 / 4.67% |

## Why V18 was replaced

At 15 minutes, V18 is worse than repeating the last reading on both datasets. It packed all
sensors of a timestep into one token, so it had no notion of individual sensors and could
not follow current conditions sensor by sensor (see
[architecture.md](architecture.md#why-v18-was-replaced)). V19 cuts the average error by 22%
on METR-LA and 30% on PEMS-BAY with about half the parameters (1.26 M against 2.44 M on
METR-LA).

## Comparison with published models

Test MAE at 15 / 30 / 60 minutes, as reported in each paper under the same protocol.

| Model | Venue | Input | METR-LA | PEMS-BAY |
|---|---|---|---|---|
| DCRNN | ICLR 2018 | 12 steps | 2.77 / 3.15 / 3.60 | 1.38 / 1.74 / 2.07 |
| Graph WaveNet | IJCAI 2019 | 12 steps | 2.69 / 3.07 / 3.53 | 1.30 / 1.63 / 1.95 |
| STID | CIKM 2022 | 12 steps | 2.82 / 3.19 / 3.55 ¹ | 1.30 / 1.62 / 1.89 |
| D2STGNN | VLDB 2022 | 12 steps | 2.56 / 2.90 / 3.35 | 1.24 / 1.55 / 1.85 |
| MegaCRN | AAAI 2023 | 12 steps | 2.52 / 2.93 / 3.38 ² | 1.28 / 1.60 / 1.88 |
| STAEformer | CIKM 2023 | 12 steps | 2.65 / 2.97 / 3.34 | 1.31 / 1.62 / 1.88 |
| TESTAM | ICLR 2024 | 12 steps | 2.54 / 2.96 / 3.36 | 1.29 / 1.59 / 1.85 |
| HimNet | KDD 2024 | 12 steps | 2.60 / 2.95 / 3.37 | 1.27 / 1.57 / 1.84 |
| MLCAFormer | PLOS One 2025 | 12 steps | 2.62 / 2.93 / 3.30 | 1.28 / 1.59 / 1.86 |
| ST-SSDL | NeurIPS 2025 | 12 steps + weekly anchor | 2.60 / 2.96 / 3.37 | 1.26 / 1.57 / 1.86 |
| **V19 (3-seed mean)** | this repo | 12 steps | **2.70 / 3.01 / 3.38** | **1.35 / 1.65 / 1.90** |
| **V19, 3-seed ensemble** | this repo | 12 steps | **2.66 / 2.97 / 3.33** | **1.32 / 1.61 / 1.85** |
| STEP | KDD 2022 | 1 week | 2.61 / 2.96 / 3.37 | 1.26 / 1.55 / 1.79 |
| STD-MAE | IJCAI 2024 | 3 days | 2.62 / 2.99 / 3.40 | 1.23 / 1.53 / 1.77 |
| **V19 + day/week lags (3-seed mean)** | this repo | 12 steps + lags | 2.73 / 3.07 / 3.46 | **1.33 / 1.58 / 1.80** |
| **V19 + lags, 3-seed ensemble** | this repo | 12 steps + lags | | **1.31 / 1.56 / 1.77** |

¹ STAEformer's re-run; STID reports no METR-LA results itself.
² Other authors' re-runs get 2.62 / 3.01 / 3.48.

- **Where a single V19 model stands.** It is a few percent behind the best reviewed models.
  - **METR-LA:** level with Graph WaveNet at 15 minutes, then ahead of it and of DCRNN
    (3.38 against 3.53 and 3.60 at 60 minutes).
  - **PEMS-BAY:** it beats DCRNN at every horizon. Against Graph WaveNet it is worse at
    15 minutes and better at 60.
- **Ensembles.** The 3-seed ensemble matches STAEformer on METR-LA. On PEMS-BAY, the
  history-lag ensemble equals STD-MAE at 60 minutes (1.77), which reads three days of
  input. An ensemble averages three models, while the papers report one.
- **Differences are small.** Since 2023 the leading models sit within about 0.05 mph of
  each other, roughly the spread between two training runs of one model (see MegaCRN's
  re-runs above).
- **Unreviewed claims.** TITAN, T-Graphormer and GAMMA-Net report much lower errors in
  preprints that are not peer-reviewed, have no code, or define horizons differently.

### Ranking on METR-LA at 60 minutes (October 2026)

A wider sweep of published METR-LA results (values read from each paper's own tables)
gives these positions among **peer-reviewed models with a 12-step input**:

| | MAPE | Rank by MAPE | MAE | Rank by MAE |
|---|---|---|---|---|
| V19 (3-seed mean) | 9.83% | about 15th | 3.38 | 10th to 11th |
| V19, best seed for MAPE (seed 44) | 9.80% | 13th to 14th | 3.40 | |
| V19, 3-seed ensemble | 9.69% | 7th to 8th | 3.33 | 4th |

- **Spread of the leaders.** Their 60-minute MAPE ranges from 9.4% to 9.9% (D2STGNN 9.40%,
  STAEformer 9.70%, HimNet 9.79%, MTGNN 9.87%; Graph WaveNet is at 10.01%).
- **Ranks are fragile.** Other authors' re-runs of one model move its MAPE by 0.2 to 0.35
  points, more than many gaps between neighbours in the ranking.
- **Wider comparisons.** Counting the long-history models (STEP, STD-MAE) moves V19 down
  two places; counting preprints moves it down about eight more.

## Ablations

Single runs with seed 42, compared with the same-seed V19 run. The base model's seed spread
is 0.02 to 0.03 mph MAE, so smaller differences are noise.

| Variant | METR-LA avg MAE | PEMS-BAY avg MAE | Verdict |
|---|---|---|---|
| V19, seed 42 | 2.966 | 1.552 | reference |
| + road-graph hop-distance prior (`--graph-bias`) | 2.968 | 1.631 | no gain |
| + weather (`--weather`, rebuilt ERA5 data) | 3.041 | 1.571 | worse, see [Weather](#weather) |
| + day/week history lags (3-seed means against 3-seed means) | +0.056 | −0.058 | real gain on PEMS-BAY only |

**History lags** (`--history-lags 288 2016`) add the readings from the same time yesterday
and last week to each input step. On PEMS-BAY they cut the 60-minute error by 0.099 mph,
6.8 times the pooled seed standard deviation, so the gain is real. On METR-LA the change
(+0.056 average) is within seed noise.

## Weather

Weather inputs make both benchmarks slightly worse, even after the weather data was rebuilt
correctly (see [DATA.md](../DATA.md#weather)): +1.2% error on PEMS-BAY and +2.5% on METR-LA.
These benchmarks cannot show a weather effect:

- **The test periods are almost dry.** Rain falls in 2.7% of PEMS-BAY test hours (against
  21.6% in training) and in 0.0% of METR-LA's (against 5.9%).
- **Temperatures drift out of range.** With a chronological split, test temperatures sit
  0.8 to 1.4 standard deviations outside the training range, so weather inputs force the
  model to extrapolate.
- **Even rainy windows favour the base model.** On the 398 rainy PEMS-BAY test windows the
  base model is still better (1.37 against 1.42 mph).

To measure what weather is worth, use a dataset whose test period contains adverse weather
(a rainy season, or a winter test split), or score weather-event windows separately.

## How much of this is luck?

Seed spread (0.02 to 0.03 mph MAE) is larger than many published gaps between models:

- **One run is not enough.** The first V19 PEMS-BAY run scored 1.552, the luckiest of three;
  the 3-seed mean is 1.583.
- **Compare means.** Compare configurations by their means over at least three seeds
  (`visu-predict aggregate`). Treat a difference as real only when it clearly exceeds the
  pooled seed standard deviation.

## Reproducing

```bash
visu-predict download --dest data
visu-predict baselines --dataset METR-LA                # persistence 5.140, historical average 4.187
visu-predict queue --queue configs/paper_runs.json --max-concurrent 3
visu-predict aggregate runs/                            # the 3-seed tables above
visu-predict ensemble runs/METR-LA_st_base runs/METR-LA_st_base_s43 runs/METR-LA_st_base_s44
```

- **Precision.** The published runs trained with `--precision bf16 --compile`; test metrics
  are always computed in full precision.
- **Time.** On a Colab A100, a METR-LA run took 50 minutes to 2.5 hours and a PEMS-BAY run 1
  to 4.5 hours, depending on how many runs shared the GPU. Runs stop early, after 30 to 70
  epochs.
- **Re-scoring.** `visu-predict evaluate runs/<run>` re-scores any finished run from its
  `best.pt`. Re-scored on CPU with this package, the published METR-LA seed-42 checkpoint
  gives the recorded test MAE (2.9658).
