# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-05-22

### Added
- Initial public release as a Python package (refactor of the original
  `VISU_Traffic_Transformer` Colab notebook).
- `visu_predict` package with `config`, `data`, `features.{weather,spatial}`,
  `models.{transformer,attention,positional,gnn,lr}`, `training.{train,losses,transfer}`,
  `utils`, `viz`, `runner`, `cli` modules.
- CLI: `visu-predict train --config <yaml> --data <csv>`.
- Example YAML configuration in `configs/example.yaml`.
- Smoke tests (config, dataset, model forward, losses).
- GitHub Actions CI matrix on Python 3.10 / 3.11 / 3.12.

### Removed (relative to the source notebook)
- Colab-specific code: `drive.mount`, hardcoded `/content/drive/...` paths,
  inline `!pip install` commands.
- `TeeLogger` stdout hijack (replaced by Python `logging`).
- Embedded LSTM baseline and the in-notebook PDF report generator.
- Duplicate weather-loading implementation living in the dataset class.
- Singleton-based `SpatialIntegration`.
