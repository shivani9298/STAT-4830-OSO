#!/usr/bin/env python3
"""
Run three Transformer + sector-portfolio models on full WRDS data (2010–2024), one
hyperparameter change each (same spirit as the short-horizon ablation):

  1. batch_32   — ``batch_size``: 32 (rest from TRANSFORMER_CONFIG)
  2. hidden_64  — ``hidden_size``: 64
  3. lambda_vol_1.0 — ``lambda_vol``: 1.0

Requires: ``.env`` WRDS credentials. Sectors from Compustat GICS unless
``IPO_SECTOR_SOURCE=yfinance``.

Environment (optional):
  IPO_SECTOR_SOURCE=compustat   (default in run_ipo_optimizer_wrds)
  IPO_EXPORT_ATTENTION=1

Outputs are copied after each run to:
  results/transformer_sector_runs/<run_id>/
  figures/transformer_sector_runs/<run_id>/

Each run also saves ``results/ipo_optimizer_training_history.json`` (slim epoch losses);
that file is copied per run. Plot all three side-by-side::

  python scripts/TRANSFORMER_plot_sector_triplet_losses.py

Usage (repo root):
  python scripts/TRANSFORMER_run_three_sector_models.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

RUNS: list[tuple[str, str]] = [
    ("batch_32", "local/ipo_optimizer_transformer_batch_32.json"),
    ("hidden_64", "local/ipo_optimizer_transformer_hidden_64.json"),
    ("lambda_vol_1.0", "local/ipo_optimizer_transformer_lambda_vol_1.json"),
]


def _copy_outputs(run_id: str) -> None:
    res_dst = ROOT / "results" / "transformer_sector_runs" / run_id
    fig_dst = ROOT / "figures" / "transformer_sector_runs" / run_id
    res_dst.mkdir(parents=True, exist_ok=True)
    fig_dst.mkdir(parents=True, exist_ok=True)

    loss_png = ROOT / "figures" / "ipo_optimizer_loss.png"
    if loss_png.exists():
        shutil.copy2(loss_png, fig_dst / "ipo_optimizer_loss.png")

    attn = ROOT / "figures" / "ipo_optimizer_attention_layer0.png"
    if attn.exists():
        shutil.copy2(attn, fig_dst / "ipo_optimizer_attention_layer0.png")

    for name in (
        "ipo_optimizer_summary_by_sector.txt",
        "ipo_optimizer_summary.txt",
    ):
        p = ROOT / "results" / name
        if p.exists():
            shutil.copy2(p, res_dst / name)

    for p in (ROOT / "results").glob("ipo_optimizer_weights_sector_*.csv"):
        shutil.copy2(p, res_dst / p.name)
    w = ROOT / "results" / "ipo_optimizer_weights.csv"
    if w.exists():
        shutil.copy2(w, res_dst / w.name)

    hist = ROOT / "results" / "ipo_optimizer_training_history.json"
    if hist.exists():
        shutil.copy2(hist, res_dst / "ipo_optimizer_training_history.json")

    npz = ROOT / "results" / "ipo_optimizer_attention.npz"
    if npz.exists():
        shutil.copy2(npz, res_dst / npz.name)


def main() -> int:
    os.chdir(ROOT)
    # Child inherits env; must be set before child imports numpy/torch (Windows OpenMP).
    if sys.platform == "win32":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("IPO_MODEL_TYPE", "transformer")
    os.environ.setdefault("IPO_SECTOR_SOURCE", "compustat")
    os.environ.setdefault("IPO_EXPORT_ATTENTION", "1")

    py = sys.executable
    main_script = ROOT / "run_ipo_optimizer_wrds.py"

    for run_id, rel_cfg in RUNS:
        cfg_path = ROOT / rel_cfg
        if not cfg_path.is_file():
            print(f"Missing config {cfg_path}", file=sys.stderr)
            return 1
        os.environ["IPO_LOCAL_CONFIG"] = str(cfg_path)
        print(f"\n========== Run: {run_id}  config={rel_cfg} ==========\n", flush=True)
        r = subprocess.run([py, "-u", str(main_script)], cwd=str(ROOT))
        if r.returncode != 0:
            print(f"[run_three] Stopping after failure in {run_id}", file=sys.stderr)
            return r.returncode
        _copy_outputs(run_id)
        print(f"[run_three] Saved artifacts under results/transformer_sector_runs/{run_id}/", flush=True)

    print("\n[run_three] All three runs finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
