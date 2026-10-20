#!/usr/bin/env bash
# Full pipeline. Data prep is idempotent; protocols run in priority order and are skipped when their
# results/pred_<protocol>.parquet exists. The ablation protocol runs fold by fold so it can be resumed.
set -euo pipefail
cd "$(dirname "$0")/src"
PY=${PYTHON:-python}
DATA=${SPLICETAC_DATA:-"$(cd .. && pwd)/data"}
export SPLICETAC_DATA="$DATA"
mkdir -p ../logs
[ -f "$DATA/processed/variants_v1.parquet" ] || $PY prep.py
[ -f "$DATA/processed/windows_v1.npz" ] || $PY features.py
[ -f "$DATA/processed/pangolin_features_v1.parquet" ] || $PY pangolin_features.py
for p in ${PROTOCOLS:-gene study bench random}; do
  [ -f "../results/pred_$p.parquet" ] && continue
  echo "=== $p $(date)"
  $PY run.py "$p" 2>&1 | grep --line-buffered -v -i warning | tee "../logs/run_$p.log"
done
if [ ! -f ../results/pred_ablation.parquet ]; then
  for k in 0 1 2 3 4; do
    [ -f "../results/pred_ablation_f$k.parquet" ] && continue
    echo "=== ablation fold $k $(date)"
    $PY run.py ablation --only_fold "$k" 2>&1 | grep --line-buffered -v -i warning | tee -a ../logs/run_ablation.log
  done
  $PY -c "import pandas as pd, glob; pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('../results/pred_ablation_f*.parquet'))]).to_parquet('../results/pred_ablation.parquet')"
fi
$PY analyze.py
echo "=== done $(date)"
