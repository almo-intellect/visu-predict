# Data

The repository contains no datasets. This page covers how to get the benchmark data,
the file formats, how to add weather, and how to use your own sensor network.

## Benchmark datasets

```bash
visu-predict download --dest data              # both datasets, about 160 MB
visu-predict download --datasets METR-LA       # just one
```

| Dataset | Sensors | Period | Steps (5 min) | Train / val / test windows |
|---|---|---|---|---|
| METR-LA (Los Angeles highways) | 207 | 2012-03-01 to 2012-06-27 | 34,272 | 23,974 / 3,425 / 6,850 |
| PEMS-BAY (San Francisco Bay Area) | 325 | 2017-01-01 to 2017-06-30 | 52,116 | 36,465 / 5,209 / 10,419 |

The command fetches `METR-LA.csv`, `PEMS-BAY.csv`, `adj_METR-LA.pkl` and
`adj_PEMS-BAY.pkl` from the maintainer's public Google Drive folder and checks their sizes:

> <https://drive.google.com/drive/folders/1eNGQpeHlxa7SWnpzIjeHFif4Ae15gjgs>

If the download fails (Google sometimes rate-limits large files), download the four files
from that folder in a browser and put them in `data/`.

They are the [DCRNN](https://github.com/liyaguang/DCRNN) releases of both datasets
(`metr-la.h5`, `pems-bay.h5`, `adj_mx.pkl`, `adj_mx_bay.pkl`) converted to CSV. If you
convert the original files yourself, keep the sensor order of the `.h5` columns.

- **METR-LA:** the CSV columns are the sensor ids, in the same order as the adjacency file.
- **PEMS-BAY:** the CSV columns are positions (`0` to `324`) rather than ids. Their order
  also matches the adjacency file (checked by correlating neighbouring sensors).

## File formats

For a dataset called `NAME`, the commands look in the data folder (`--data`, default `./data`) for:

| File | Required | Content |
|---|---|---|
| `NAME.csv` | yes | First column: timestamps. Then one column per sensor. One row per time step, at a regular interval (5 minutes for the benchmarks; any interval works). |
| `adj_NAME.pkl` or `adj_NAME.npy` | for `--graph-bias` | N × N adjacency matrix, sensors in the same order as the CSV columns. Pickles can hold the bare array or the DCRNN list `[sensor_ids, {sensor_id: index}, matrix]`. |
| `weather_NAME_era5_local.csv` | for `--weather` | Hourly weather in the same local time as the traffic, from `visu-predict weather`. |

Readings equal to `0` mean "missing" (sensor faults). They stay in the data, are never
used as targets, and are left out of every metric, as in the published protocol.

## Weather

```bash
visu-predict weather --data data                 # METR-LA and PEMS-BAY
```

This writes `weather_<dataset>_era5_local.csv`: hourly ERA5 reanalysis from the
[Open-Meteo archive API](https://open-meteo.com/en/docs/historical-weather-api). It
holds 12 variables (temperature, precipitation, humidity, dew point, wind, gusts, cloud
cover, pressure and rain / fog / cloud flags) for one point at the centre of each
network, in local time, covering 100% of the traffic period. No API key is needed.
Open-Meteo's free tier is for non-commercial use; read its
[terms](https://open-meteo.com/en/terms) before commercial use.

These files replace the `clean_weather_data_*.csv` exports used up to V18, which had
three defects:

- **Wrong time zone.** They were stamped in UTC while traffic is local time, so the daily
  temperature peak landed at about 22:00.
- **Incomplete.** PEMS-BAY weather ended a month before the traffic data, leaving about
  85% of the test period without weather.
- **Invented values.** Hourly readings were forward-filled for up to 72 hours, and METR-LA
  precipitation never dropped below 0.01 and had no rain labels.

If only an old export is present, the pipeline still reads it (converting from UTC) and warns.

Weather does not improve the benchmark scores, even with correct data. See
[docs/results.md](docs/results.md#weather) for why.

## Your own sensor network

1. Write `data/MYCITY.csv` in the format above (and optionally `adj_MYCITY.pkl`).
2. Check the data and get reference scores:
   ```bash
   visu-predict baselines --dataset MYCITY
   ```
3. Train:
   ```bash
   visu-predict train --dataset MYCITY
   ```
4. Optional extras:
   - **Weather:** give the network's location and the time zone of its timestamps.
     ```bash
     visu-predict weather --datasets MYCITY --lat -25.97 --lon 32.57 --tz Africa/Maputo
     visu-predict train --dataset MYCITY --weather
     ```
   - **Public holidays:** add `--holidays --holiday-country MZ` (any country code known to
     the [`holidays`](https://pypi.org/project/holidays/) package; install it with
     `pip install "visu-predict[holidays]"`).
   - **History lags:** `--history-lags 288 2016` adds the readings from the same time
     yesterday and last week. The numbers are steps, so scale them to your data interval.

The split is chronological: the first 70% of windows train, the next 10% validate and the
last 20% test. With a few weeks of data the test period is short, so read the scores with that in mind.

For networks with too little history to train from scratch, start from a model trained on
a large network and re-initialise only its sensor-specific embeddings with
`STTransformer.adapt_to_graph` (see [docs/architecture.md](docs/architecture.md#transfer-to-another-network)).
