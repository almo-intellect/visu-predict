# Data files

`visu-predict` is intentionally distributed without datasets: traffic CSVs,
adjacency pickles, sensor-location CSVs, and weather data live outside the
repository. This keeps clones small and avoids redistributing third-party
research datasets.

## Where to get the data

The maintainer publishes a Drive folder with all input files used by the
reference experiments:

> **<https://drive.google.com/drive/folders/1eNGQpeHlxa7SWnpzIjeHFif4Ae15gjgs?usp=sharing>**

Files you can expect to find there (per dataset):

| Purpose | Filename pattern |
|---------|------------------|
| Traffic time series | `METR-LA.csv`, `PEMS-BAY.csv`, `PEMS-03.csv`, … |
| Adjacency matrix | `adj_METR-LA.pkl`, `adj_PEMS-BAY.pkl`, … |
| Sensor coordinates | `graph_sensor_locations_metr_la.csv`, … |
| Cleaned weather data | `clean_weather_data_<dataset>.csv` |

## Where to put the data

After downloading, place files under your run's `inputs/` directory.
`setup_directories()` creates this directory at:

```
<base_output_dir>/inputs/
```

For example, with the default `configs/example.yaml`:

```
./outputs/inputs/METR-LA.csv
./outputs/inputs/adj_METR-LA.pkl
./outputs/inputs/graph_sensor_locations_metr_la.csv
```

`paths.find_adjacency_matrix` and `paths.find_coordinates` then locate the
right files based on `dataset_name` in your YAML config.

## Original sources

The traffic / sensor-location data comes from the public DCRNN release
([liyaguang/DCRNN](https://github.com/liyaguang/DCRNN)) for METR-LA and
PEMS-BAY, and from the STSGCN / ASTGCN releases for PEMS-03, -04, -07, and -08.
Weather data is derived from public NOAA observations matched to sensor
coordinates.

## Helper script

A small downloader is provided at `scripts/download_data.py` — see
`python scripts/download_data.py --help` for usage.
