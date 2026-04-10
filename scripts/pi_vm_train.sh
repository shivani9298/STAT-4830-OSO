#!/usr/bin/env bash
# One-shot setup + train on a Linux GPU VM (e.g. Prime Intellect).
# Run from repo root:  bash scripts/pi_vm_train.sh
#
# Before first run on the VM, copy the prepared cache from your laptop (large):
#   rsync -avz results/cache_wrds_maxhist/ user@VM:~/STAT-4830-OSO/results/cache_wrds_maxhist/
# And ensure results/ipo_optimizer_best_config.json is present (or remove it to use pure DEFAULTS).
#
# PyTorch wheel index must match the VM CUDA driver. See https://pytorch.org — common:
#   cu124  CUDA 12.4   |  cu121  CUDA 12.1   |  cu118  CUDA 11.8   |  cpu
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.cache/mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "warning: nvidia-smi not found — training will fall back to CPU (very slow)."
else
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv venv .venv
  # shellcheck source=/dev/null
  source .venv/bin/activate
  echo "Installing PyTorch from $TORCH_INDEX ..."
  uv pip install torch --index-url "$TORCH_INDEX"
  uv pip install pandas==2.1.4 sqlalchemy==1.4.54 wrds==3.1.6 python-dotenv matplotlib numpy
  echo "Dependencies installed."
else
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    # shellcheck source=/dev/null
    source "$ROOT_DIR/.venv/bin/activate"
  fi
fi

exec bash "$ROOT_DIR/scripts/train_three_models_wrds_cache.sh" "$@"
