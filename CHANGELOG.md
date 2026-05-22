# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-05-22

### Added
- Initial public release of the `visu_predict` Python package.
- Modules: `config`, `data`, `features.{weather,spatial}`,
  `models.{transformer,attention,positional,gnn,lr}`,
  `training.{train,losses,transfer}`, `utils`, `viz`, `runner`, `cli`.
- CLI: `visu-predict train --config <yaml> --data <csv>`.
- Example YAML configuration in `configs/example.yaml`.
- Smoke tests covering config, dataset, model forward pass, and losses.
- GitHub Actions CI matrix on Python 3.10 / 3.11 / 3.12.
- Bilingual README (English and European Portuguese).
