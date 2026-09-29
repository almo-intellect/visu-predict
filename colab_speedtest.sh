#!/bin/bash
# Stop running benchmark processes, refresh code from Drive, run the speed test.
V19="/content/drive/Shareddrives/Almo-2002-R&D/1 - RD-Traffic-Prediction/Transformer_Versions/COLAB_NOBASELINE_V19"
pkill -f run_benchmark.py && echo "stopped running benchmark processes" || echo "no benchmark processes running"
sleep 5
OUT="$V19/results/benchmark"; mkdir -p "$OUT/_aborted_wave1"
for r in PEMS-BAY_st_base PEMS-BAY_st_graph METR-LA_st_base METR-LA_st_graph; do
  if [ -d "$OUT/$r" ] && [ ! -f "$OUT/$r/results.json" ]; then mv "$OUT/$r" "$OUT/_aborted_wave1/$r"; fi
done
rm -rf /content/v19 && mkdir -p /content/v19
cp -r "$V19/traffic_transformer" /content/v19/ && cp "$V19"/*.py "$V19"/*.sh /content/v19/ 2>/dev/null
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
cd /content/v19 && python -W ignore speed_test.py /content/data PEMS-BAY 2>&1 | grep -v "^Loaded adjacency"
