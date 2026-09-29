"""Aggregate the seeds of each V19 configuration: mean +/- sample std per metric.

    python aggregate_seeds.py <results/benchmark dir> [out.json]
"""
import json
import os
import sys

import numpy as np

OUT = sys.argv[1]
DEST = sys.argv[2] if len(sys.argv) > 2 else "final/seed_stats.json"

CONFIGS = {
    ("PEMS-BAY", "base"): ["PEMS-BAY_st_base", "PEMS-BAY_st_base_s43", "PEMS-BAY_st_base_s44"],
    ("PEMS-BAY", "hist"): ["PEMS-BAY_st_hist", "PEMS-BAY_st_hist_s43", "PEMS-BAY_st_hist_s44"],
    ("METR-LA", "base"): ["METR-LA_st_base", "METR-LA_st_base_s43", "METR-LA_st_base_s44"],
    ("METR-LA", "hist"): ["METR-LA_st_hist", "METR-LA_st_hist_s43", "METR-LA_st_hist_s44"],
}
LABEL = {"base": "V19 STTransformer", "hist": "V19 + day/week history lags"}
HORIZONS = ("h3", "h6", "h12", "all")
METRICS = ("mae", "rmse", "mape")

stats = {}
for (ds, cfg), runs in CONFIGS.items():
    acc = {h: {m: [] for m in METRICS} for h in HORIZONS}
    seeds, params = [], None
    for r in runs:
        p = os.path.join(OUT, r, "results.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        seeds.append(d["train_config"]["seed"])
        params = d.get("params")
        for h in HORIZONS:
            for m in METRICS:
                acc[h][m].append(d["test"][h][m])
    if not seeds:
        continue
    entry = {"runs": len(seeds), "seeds": seeds, "params": params}
    for h in HORIZONS:
        entry[h] = {}
        for m in METRICS:
            a = np.array(acc[h][m])
            entry[h][m] = {"mean": float(a.mean()),
                           "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                           "values": [round(x, 4) for x in a]}
    stats.setdefault(ds, {})[cfg] = entry

notes = []
for ds in ("PEMS-BAY", "METR-LA"):
    b, hh = stats.get(ds, {}).get("base"), stats.get(ds, {}).get("hist")
    if b and hh and b["runs"] > 1 and hh["runs"] > 1:
        for h, name in (("h12", "60-min"), ("all", "average")):
            diff = hh[h]["mae"]["mean"] - b[h]["mae"]["mean"]
            pooled = float(np.sqrt((b[h]["mae"]["std"] ** 2 + hh[h]["mae"]["std"] ** 2) / 2))
            ratio = abs(diff) / pooled if pooled > 0 else float("inf")
            verdict = "larger than seed noise" if ratio >= 2 else "within seed noise"
            notes.append({"dataset": ds, "horizon": name, "diff": round(diff, 4),
                          "pooled_sd": round(pooled, 4), "ratio": round(ratio, 1), "verdict": verdict,
                          "text": f"{ds} {name}: history lags {diff:+.3f} mph vs base "
                                  f"(pooled seed sd {pooled:.3f}, {ratio:.1f}x) - {verdict}"})

os.makedirs(os.path.dirname(DEST) or ".", exist_ok=True)
json.dump({"stats": stats, "notes": notes}, open(DEST, "w"), indent=1)

print("| Dataset | Configuration | seeds | 15 min | 30 min | 60 min | Average |")
print("|---|---|---|---|---|---|---|")
for ds in ("PEMS-BAY", "METR-LA"):
    for cfg in ("base", "hist"):
        e = stats.get(ds, {}).get(cfg)
        if e:
            cells = " | ".join(f"{e[h]['mae']['mean']:.3f} ± {e[h]['mae']['std']:.3f}" for h in HORIZONS)
            print(f"| {ds} | {LABEL[cfg]} | {e['runs']} ({','.join(str(s) for s in e['seeds'])}) | {cells} |")
print()
for n in notes:
    print(n["text"])
print(f"\nwrote {DEST}")
