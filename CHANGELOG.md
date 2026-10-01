# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-01 - V19

The repository now holds V19 of the VISU traffic model. The 0.1.0 package (V14)
is available under the `v0.1.0` tag.

### Added
- `STTransformer`: node-level spatio-temporal Transformer (STAEformer family). Every
  (timestep, sensor) pair is a token; temporal attention per sensor, spatial attention
  across sensors, per-sensor output head. 1.26 M parameters on METR-LA.
- Standard METR-LA / PEMS-BAY benchmark protocol: DCRNN 70/10/20 window split, raw
  targets with zeros masked, train-only normalisation, MAE / RMSE / MAPE at 15 / 30 / 60
  minutes and averaged over all steps. Results are directly comparable with published tables.
- Optional inputs: same-time-yesterday / last-week history lags, a road-graph hop-distance
  attention prior, ERA5 weather, and public holidays.
- `adapt_to_graph()` to move a trained model to a network with a different number of sensors.
- `visu-predict` command with `download`, `baselines`, `train`, `evaluate`, `aggregate`,
  `ensemble`, `queue` and `weather` subcommands.
- Self-contained checkpoints: `best.pt` stores the model class, configuration and scaler,
  and `load_checkpoint()` rebuilds the model from that file alone.
- Seed statistics (`aggregate`) and prediction ensembles (`ensemble`).
- Weather files rebuilt from hourly ERA5 reanalysis in local time (`visu-predict weather`).
- `configs/paper_runs.json`, the queue that reproduces every published run.
- Colab notebook, `docs/results.md`, `docs/architecture.md`, `CONTRIBUTING.md`.
- Tests for the data protocol, model, training loop, CLI, queue runner and the V18 package;
  CI on Python 3.10 to 3.13.

### Changed
- The V18 `TrafficTransformer` package moved to `visu_predict.legacy` (optional
  `[legacy]` extra). It can be scored under the V19 protocol with `--model legacy`.
- Package version 0.2.0; the `torch` requirement is now `>=2.3`.

### Results (test MAE in mph, mean of 3 seeds)
- METR-LA: 2.98 average (V18: 3.82), 3.38 at 60 minutes.
- PEMS-BAY: 1.58 average (V18: 2.25); 1.53 with day/week history lags.
- See `docs/results.md` for the full tables and the comparison with published models.

## [0.1.0] - 2026-05-22

### Added
- Initial public release of the `visu_predict` Python package (V14 Colab notebook
  refactored into a package).
- Modules: `config`, `data`, `features.{weather,spatial}`,
  `models.{transformer,attention,positional,gnn,lr}`,
  `training.{train,losses,transfer}`, `utils`, `viz`, `runner`, `cli`.
- CLI: `visu-predict train --config <yaml> --data <csv>`.
- Example YAML configuration in `configs/example.yaml`.
- Smoke tests covering config, dataset, model forward pass, and losses.
- GitHub Actions CI matrix on Python 3.10 / 3.11 / 3.12.
- Bilingual README (English and European Portuguese).

[0.2.0]: https://github.com/almo-intellect/visu-predict/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/almo-intellect/visu-predict/releases/tag/v0.1.0
