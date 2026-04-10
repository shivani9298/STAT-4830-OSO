#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

END_DATE="${IPO_END_DATE:-2025-12-31}"
EXTRA_ARGS=("$@")

if [[ -z "${WRDS_USERNAME:-}" ]]; then
  echo "ERROR: WRDS_USERNAME is not set."
  echo "Set WRDS_USERNAME (and WRDS_PASSWORD or ~/.pgpass) before launching background WRDS jobs."
  exit 1
fi

launch_model() {
  local model="$1"
  local runner="$2"
  local log_file="$LOG_DIR/train_wrds_${model}.log"
  local cmd=(python3 -u "$ROOT_DIR/$runner" --max-history --end-date "$END_DATE")
  if ((${#EXTRA_ARGS[@]})); then
    cmd+=("${EXTRA_ARGS[@]}")
  fi
  nohup "${cmd[@]}" >"$log_file" 2>&1 &

  local pid=$!
  echo "$model started (pid=$pid) log=$log_file"
}

launch_model "gru" "scripts/run_ipo_optimizer_wrds_gru.py"
launch_model "lstm" "scripts/run_ipo_optimizer_wrds_lstm.py"
launch_model "transformer" "scripts/run_ipo_optimizer_wrds_transformer.py"

echo "All model trainings launched in background."
