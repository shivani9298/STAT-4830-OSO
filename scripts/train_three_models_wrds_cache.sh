#!/usr/bin/env bash
# Sequential training: GRU, LSTM, Transformer from a prepared WRDS cache (no live WRDS pull).
#
# Loss defaults come from run_ipo_optimizer_wrds.py DEFAULTS merged with
# results/ipo_optimizer_best_config.json (tuned keys like window_len / lambdas;
# training mechanics always from DEFAULTS: lr, batch, epochs, patience).
#
# Current DEFAULTS include, among others:
#   lambda_log_return=0.2, train_segment_len=63, lambda_segment_log=0.1
# Omit CLI overrides below to stay on those defaults; pass explicit flags to pin values.
#
# Usage:
#   cd /path/to/STAT-4830-OSO && source .venv/bin/activate
#   ./scripts/train_three_models_wrds_cache.sh
#
#   IPO_CACHE_DIR=results/cache_wrds_maxhist ./scripts/train_three_models_wrds_cache.sh
#   ./scripts/train_three_models_wrds_cache.sh --selection-metric mean_excess_vs_ew
#
# With uv (no prior activate):
#   uv run bash scripts/train_three_models_wrds_cache.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# VM / containers: matplotlib and fontconfig need a writable cache (avoid $HOME permission issues).
if [[ -z "${MPLCONFIGDIR:-}" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/.cache/mpl"
fi
mkdir -p "$MPLCONFIGDIR"
if [[ -z "${XDG_CACHE_HOME:-}" ]] && { [[ ! -e "${HOME}/.cache" ]] || [[ ! -w "${HOME}/.cache" ]]; }; then
  export XDG_CACHE_HOME="$ROOT_DIR/.cache"
  mkdir -p "$XDG_CACHE_HOME/fontconfig"
fi

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
CACHE_DIR="${IPO_CACHE_DIR:-$ROOT_DIR/results/cache_wrds_maxhist}"
TS="$(date -u +"%Y%m%d_%H%M%S")"
META_FILE="$LOG_DIR/train_three_models_cache_meta_${TS}.txt"

if [[ ! -f "$CACHE_DIR/prepared_df.pkl" ]] || [[ ! -f "$CACHE_DIR/prepared_meta.json" ]]; then
  echo "ERROR: Cache missing under $CACHE_DIR (need prepared_df.pkl and prepared_meta.json)."
  exit 1
fi

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

{
  echo "train_three_models_wrds_cache.sh"
  echo "timestamp_utc=$TS"
  echo "cache_dir=$CACHE_DIR"
  echo "python=$PYTHON"
  (command -v git >/dev/null 2>&1 && git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null) || echo "git_commit=unknown"
} | tee "$META_FILE"

echo "" | tee -a "$META_FILE"
echo "Effective merged config (GRU; LSTM/Transformer share this unless ipo_optimizer_best_config_{lstm,transformer}.json exists):" | tee -a "$META_FILE"
ROOT_DIR_EXPORT="$ROOT_DIR" "$PYTHON" - <<'PY' | tee -a "$META_FILE"
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ["ROOT_DIR_EXPORT"])
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from scripts.run_ipo_optimizer_wrds import DEFAULTS, load_best_config

print(
    "DEFAULTS snippet:",
    {k: DEFAULTS[k] for k in (
        "lambda_log_return", "train_segment_len", "lambda_segment_log",
        "lambda_turnover", "window_len", "lr", "batch_size", "epochs",
    )},
)
print(json.dumps(load_best_config("gru"), indent=2, default=float))
PY

for model in gru lstm transformer; do
  log="$LOG_DIR/train_${model}_cache_${TS}.log"
  echo ""
  echo "=== START $model $(date -u +"%Y-%m-%dT%H:%M:%SZ") log=$log ==="
  "$PYTHON" -u scripts/run_ipo_optimizer_wrds.py \
    --use-cache \
    --cache-dir "$CACHE_DIR" \
    --model "$model" \
    "$@" 2>&1 | tee "$log"
done

echo ""
echo "Done. Artifacts under results/ (*_lstm / *_transformer suffixes; GRU unsuffixed)."
echo "Compare plots: python scripts/plot_model_comparison_vm.py --artifacts-dir . --out-dir \"figures/old diagrams/model_comparison_fresh\""
