# Contributing

## Setup

```bash
git clone https://github.com/almo-intellect/visu-predict && cd visu-predict
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,legacy]"
```

## Before opening a pull request

```bash
ruff check src tests          # lint (ruff check --fix applies most fixes)
pytest                        # about 30 s on CPU
```

CI runs both on Python 3.10 to 3.13 for every pull request to `main`.

Tests that need the real datasets run only when `VISU_DATA_DIR` points at a folder created
by `visu-predict download`:

```bash
VISU_DATA_DIR=data pytest tests/test_data.py
```

## Changing the model or the protocol

- **Keep the protocol fixed.** Splits, masking, metrics and test windows are what make
  results comparable with the literature. A change there invalidates every number in
  `docs/results.md`.
- **Use at least three seeds.** Seed spread (0.02 to 0.03 mph MAE) is larger than many
  differences between configurations. Compare means with `visu-predict aggregate`, and
  call a gain real only when it clearly exceeds the pooled seed standard deviation.
- **Record new runs.** Add new configurations to `configs/paper_runs.json`. Update
  `docs/results.md`, `results/` and `CHANGELOG.md` when published numbers change.
- **Leave `src/visu_predict/legacy/` as it is** unless the fix concerns the V18 model itself.
  It keeps earlier experiments reproducible and is excluded from lint.

## Style

- Python 3.10+, type hints, ruff's default style (line length 120).
- Docstrings explain why, not just what. Many choices exist to keep results comparable.
- Commit messages: a short imperative summary (`fix: ...`, `feat: ...`, `docs: ...`).
