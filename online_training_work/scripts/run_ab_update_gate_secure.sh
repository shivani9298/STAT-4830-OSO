#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Create it first with: uv venv .venv"
  exit 1
fi

source ".venv/bin/activate"

if [[ -z "${WRDS_PASSWORD:-}" ]]; then
  read -r -s -p "Enter WRDS password (input hidden): " WRDS_PASSWORD
  echo
  export WRDS_PASSWORD
fi

python3 scripts/sweep_online_settings.py \
  --cadences W \
  --lookbacks 252 504 \
  --epochs-step 2 \
  --gate-modes cadence confidence \
  --gate-min-val-improvement 0.0 \
  --gate-min-relative-improvement 0.0 \
  --gate-min-history-windows 252 \
  --out-csv results/ab_update_gate_results.csv

# Do not keep password in shell environment after run.
unset WRDS_PASSWORD
