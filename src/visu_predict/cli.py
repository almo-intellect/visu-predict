"""Command-line interface for visu-predict."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from visu_predict.config import TrainingConfig, load_config
from visu_predict.runner import run_pretraining, run_training


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visu-predict",
        description="Train a transformer model on traffic time series.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a model from a YAML config")
    train_p.add_argument("--config", "-c", type=Path, required=True, help="Path to YAML config")
    train_p.add_argument("--data", "-d", type=Path, required=True, help="Path to traffic CSV")
    train_p.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Override base_output_dir from the config",
    )

    pretrain_p = sub.add_parser(
        "pretrain",
        help="Masked-reconstruction pretraining (STAE pipeline only).",
    )
    pretrain_p.add_argument("--config", "-c", type=Path, required=True, help="Path to YAML config")
    pretrain_p.add_argument("--data", "-d", type=Path, required=True, help="Path to traffic CSV")
    pretrain_p.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Override base_output_dir from the config",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config: TrainingConfig = load_config(args.config)
        if args.output is not None:
            config.base_output_dir = str(args.output)
        run_training(config, args.data)
        return 0

    if args.command == "pretrain":
        config = load_config(args.config)
        if args.output is not None:
            config.base_output_dir = str(args.output)
        run_pretraining(config, args.data)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
