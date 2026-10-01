"""
Command-line interface: ``visu-predict <command> [options]`` (or ``python -m visu_predict``).

    visu-predict download                              # METR-LA + PEMS-BAY into ./data
    visu-predict baselines --dataset METR-LA           # persistence / historical average
    visu-predict train --dataset METR-LA               # train + test the V19 model
    visu-predict evaluate runs/<run>                   # re-score a trained run
    visu-predict aggregate runs/                       # mean +/- sd over seeds
    visu-predict ensemble runs/<run1> runs/<run2> ...  # average several runs' predictions
    visu-predict queue --queue configs/paper_runs.json # many runs, a few at a time
    visu-predict weather --data data                   # rebuild the ERA5 weather files
"""

import argparse
import json
import sys

from . import __version__

EPILOG = """examples:
  visu-predict download --dest data
  visu-predict baselines --dataset METR-LA
  visu-predict train --dataset METR-LA --precision bf16 --compile --save-predictions
  visu-predict train --dataset PEMS-BAY --history-lags 288 2016 --seed 43
  visu-predict evaluate runs/<run-name>
  visu-predict aggregate runs/
  visu-predict ensemble runs/<run-seed42> runs/<run-seed43> runs/<run-seed44>

Run `visu-predict <command> --help` for the options of each command."""


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show defaults only where they say something (not None / False / [])."""

    def _get_help_string(self, action):
        if action.default is None or action.default is False or action.default == []:
            return action.help
        return super()._get_help_string(action)


def add_data_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data")
    g.add_argument("--dataset", required=True,
                   help="dataset name; reads <data>/<dataset>.csv (METR-LA, PEMS-BAY or your own)")
    g.add_argument("--data", "--data-dir", "--input-dir", dest="data_dir", default="data",
                   help="folder with the input files")
    g.add_argument("--out", "--output-dir", dest="out_dir", default="runs", help="folder for run outputs")
    g.add_argument("--run-name", default=None, help="sub-folder for this run (default: generated from the options)")
    g.add_argument("--device", default=None, help="cuda, cpu, mps, ... (default: cuda if available)")
    g.add_argument("--seed", type=int, default=42, help="random seed")
    g.add_argument("--batch-size", type=int, default=16, help="training batch size")
    g.add_argument("--in-steps", type=int, default=12, help="input window length in steps")
    g.add_argument("--out-steps", type=int, default=12, help="forecast horizon in steps")
    g.add_argument("--weather", action="store_true", help="add weather inputs (build them with `visu-predict weather`)")
    g.add_argument("--weather-file", default=None, help="weather CSV (default: weather_<dataset>_era5_local.csv)")
    g.add_argument("--holidays", action="store_true", help="treat public holidays as an 8th day type")
    g.add_argument("--holiday-country", default=None, help="country code for --holidays (METR-LA / PEMS-BAY: US)")
    g.add_argument("--history-lags", type=int, nargs="*", default=[],
                   help="add the readings this many steps before each target, e.g. 288 2016 = "
                        "same time yesterday and last week on 5-minute data")


def add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", choices=["st_transformer", "legacy", "baselines"], default="st_transformer",
                   help="st_transformer = V19; legacy = V18 (needs the [legacy] extra)")
    m = p.add_argument_group("model (st_transformer)")
    m.add_argument("--dim-input", type=int, default=24, help="value embedding size")
    m.add_argument("--dim-tod", type=int, default=24, help="time-of-day embedding size")
    m.add_argument("--dim-dow", type=int, default=24, help="day-of-week embedding size")
    m.add_argument("--dim-node", type=int, default=0, help="per-sensor embedding size")
    m.add_argument("--dim-adaptive", type=int, default=80, help="spatio-temporal adaptive embedding size")
    m.add_argument("--dim-exo", type=int, default=16, help="weather embedding size (with --weather)")
    m.add_argument("--ff-dim", type=int, default=256, help="feed-forward width")
    m.add_argument("--heads", type=int, default=4, help="attention heads")
    m.add_argument("--t-layers", type=int, default=3, help="temporal attention layers")
    m.add_argument("--s-layers", type=int, default=3, help="spatial attention layers")
    m.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    m.add_argument("--norm-first", action="store_true", help="pre-LayerNorm blocks instead of post-LN")
    m.add_argument("--graph-bias", action="store_true", help="road-graph hop-distance prior in spatial attention")
    m.add_argument("--graph-max-hops", type=int, default=6, help="hop distances beyond this share one bias")
    lg = p.add_argument_group("model (legacy)")
    lg.add_argument("--legacy-d-model", type=int, default=256, help="V18 model width")
    lg.add_argument("--legacy-layers", type=int, default=4, help="V18 encoder layers")
    lg.add_argument("--legacy-heads", type=int, default=8, help="V18 attention heads")
    t = p.add_argument_group("training (unset options follow the per-dataset recipe)")
    t.add_argument("--epochs", type=int, default=None, help="maximum epochs (recipe: 200 METR-LA, 300 PEMS-BAY)")
    t.add_argument("--patience", type=int, default=None, help="early-stopping patience (recipe: 30 / 20)")
    t.add_argument("--lr", type=float, default=None, help="learning rate (recipe: 1e-3)")
    t.add_argument("--weight-decay", type=float, default=None, help="weight decay (recipe: 3e-4 / 1e-4)")
    t.add_argument("--milestones", type=int, nargs="*", default=None,
                   help="epochs where the learning rate drops (recipe: 20 30 / 10 30)")
    t.add_argument("--lr-decay", type=float, default=0.1, help="learning-rate factor at each milestone")
    t.add_argument("--optimizer", choices=["adam", "adamw"], default=None, help="(recipe: adam)")
    t.add_argument("--scheduler", choices=["multistep", "cosine", "none"], default=None, help="(recipe: multistep)")
    t.add_argument("--warmup-epochs", type=int, default=None, help="linear warm-up epochs (recipe: 0)")
    t.add_argument("--clip-grad", type=float, default=0.0, help="gradient-norm clipping (0 = off)")
    t.add_argument("--loss", choices=["masked_mae", "masked_huber"], default="masked_mae", help="training loss")
    t.add_argument("--precision", choices=["fp32", "tf32", "bf16", "fp16"], default="tf32",
                   help="bf16 is fastest on A100 / L4 GPUs; test metrics are always computed in full precision")
    t.add_argument("--compile", action="store_true", help="torch.compile the model (faster on GPU)")
    t.add_argument("--resume", action="store_true", help="continue an interrupted run (same --run-name)")
    t.add_argument("--save-predictions", action="store_true",
                   help="write test_predictions.npz (needed by `visu-predict ensemble`)")
    t.add_argument("--log-every", type=int, default=0, help="log every N batches (0 = once per epoch)")
    t.add_argument("--max-train-batches", type=int, default=None, help=argparse.SUPPRESS)
    t.add_argument("--max-eval-batches", type=int, default=None, help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visu-predict",
        description="VISU Predict: traffic forecasting with a node-level spatio-temporal Transformer.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"visu-predict {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def command(name: str, help: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help, description=help, formatter_class=HelpFormatter)

    p = command("download", "download the METR-LA / PEMS-BAY input files")
    p.add_argument("--datasets", nargs="+", default=["METR-LA", "PEMS-BAY"], choices=["METR-LA", "PEMS-BAY"],
                   help="datasets to fetch")
    p.add_argument("--dest", default="data", help="target folder")
    p.add_argument("--force", action="store_true", help="download again even if the files exist")
    p.set_defaults(func=cmd_download)

    p = command("baselines", "score the persistence and historical-average baselines")
    add_data_args(p)
    p.set_defaults(func=cmd_baselines)

    p = command("train", "train a model and report its test metrics (standard protocol)")
    add_data_args(p)
    add_train_args(p)
    p.set_defaults(func=cmd_train)

    p = command("evaluate", "re-score a trained run on the test split")
    p.add_argument("run", help="run folder (uses its best.pt) or a checkpoint file")
    p.add_argument("--data", "--data-dir", dest="data_dir", default="data", help="folder with the input files")
    p.add_argument("--dataset", default=None, help="override the dataset recorded in the run")
    p.add_argument("--device", default=None, help="cuda, cpu, ... (default: cuda if available)")
    p.add_argument("--precision", choices=["fp32", "tf32", "bf16"], default=None,
                   help="default: tf32 on GPU, fp32 on CPU")
    p.add_argument("--batch-size", type=int, default=64, help="evaluation batch size")
    p.add_argument("--save-predictions", metavar="NPZ", default=None, help="write the predictions to this file")
    p.add_argument("--json", metavar="FILE", default=None, help="write the metrics to this JSON file")
    p.set_defaults(func=cmd_evaluate)

    p = command("aggregate", "mean and standard deviation over runs that differ only by seed")
    p.add_argument("runs", nargs="+", help="run folders, or folders containing runs")
    p.add_argument("--json", metavar="FILE", default=None, help="also write the statistics to this file")
    p.set_defaults(func=cmd_aggregate)

    p = command("ensemble", "score the average of several runs' test predictions")
    p.add_argument("runs", nargs="+", help="run folders with test_predictions.npz")
    p.add_argument("--json", metavar="FILE", default=None, help="also write the result to this file")
    p.set_defaults(func=cmd_ensemble)

    p = command("queue", "run many trainings from a JSON queue, a few at a time on one GPU")
    p.add_argument("--queue", required=True, help="JSON list of {name, args}, e.g. configs/paper_runs.json")
    p.add_argument("--data", "--data-dir", dest="data_dir", default="data", help="folder with the input files")
    p.add_argument("--out", "--output-dir", dest="out_dir", default="runs", help="folder for run outputs")
    p.add_argument("--max-concurrent", type=int, default=2, help="runs sharing the GPU at once")
    p.add_argument("--common", default="--save-predictions",
                   help="options given to every run (a run's own args take precedence)")
    p.add_argument("--poll", type=float, default=120, help="seconds between status checks")
    p.add_argument("--busy-minutes", type=float, default=15.0,
                   help="leave runs alone whose log changed this recently (another process owns them)")
    p.add_argument("--stagger", type=float, default=15.0, help="seconds between launches")
    p.set_defaults(func=cmd_queue)

    p = command("weather", "build hourly ERA5 weather files (Open-Meteo) for a dataset")
    p.add_argument("--data", "--data-dir", dest="data_dir", default="data", help="folder with <dataset>.csv")
    p.add_argument("--datasets", nargs="+", default=["METR-LA", "PEMS-BAY"], help="datasets to cover")
    p.add_argument("--out", dest="out_dir", default=None, help="output folder (default: the data folder)")
    p.add_argument("--lat", type=float, default=None, help="latitude, for datasets without a built-in location")
    p.add_argument("--lon", type=float, default=None, help="longitude")
    p.add_argument("--tz", default=None, help="IANA timezone of the traffic timestamps, e.g. Africa/Maputo")
    p.set_defaults(func=cmd_weather)
    return parser


# =============================================================================
# Command handlers
# =============================================================================

def _device(a) -> str:
    if a.device:
        return a.device
    from .benchmark import default_device

    return default_device()


def cmd_download(a) -> int:
    from .download import download

    download(a.datasets, a.dest, force=a.force)
    print(f"\nData ready in {a.dest}/. Next: visu-predict baselines --dataset {a.datasets[0]} --data {a.dest}")
    return 0


def cmd_baselines(a) -> int:
    from .benchmark import run_baselines

    a.device = _device(a)
    run_baselines(a)
    return 0


def cmd_train(a) -> int:
    from .benchmark import run_train

    a.device = _device(a)
    run_train(a)
    return 0


def cmd_evaluate(a) -> int:
    from .benchmark import run_evaluate

    a.device = _device(a)
    run_evaluate(a)
    return 0


def cmd_aggregate(a) -> int:
    from .analysis import aggregate, find_runs, format_aggregate

    runs = find_runs(a.runs)
    if not runs:
        print("no finished runs found (folders with a results.json)", file=sys.stderr)
        return 1
    groups = aggregate(runs)
    print(format_aggregate(groups))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(groups, f, indent=1)
    return 0


def cmd_ensemble(a) -> int:
    from .analysis import ensemble, format_ensemble

    result = ensemble(a.runs)
    print(format_ensemble(result))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0


def cmd_queue(a) -> int:
    from .job_queue import run_queue

    outcome = run_queue(a.queue, a.data_dir, a.out_dir, max_concurrent=a.max_concurrent, common=a.common,
                        poll=a.poll, busy_minutes=a.busy_minutes, stagger=a.stagger)
    return 0 if all(outcome.values()) else 1


def cmd_weather(a) -> int:
    from .weather import build_weather

    if (a.lat is not None or a.lon is not None or a.tz) and len(a.datasets) != 1:
        print("--lat/--lon/--tz describe one dataset: pass exactly one name with --datasets", file=sys.stderr)
        return 2
    for ds in a.datasets:
        build_weather(ds, a.data_dir, a.out_dir, lat=a.lat, lon=a.lon, tz=a.tz)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
