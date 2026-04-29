#!/usr/bin/env python3
"""
Sweep online update cadence and lookback settings.

Runs `run_ipo_optimizer_wrds.py` multiple times using env overrides and records
headline metrics from `results/ipo_optimizer_summary.txt`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "run_ipo_optimizer_wrds.py"
SUMMARY = ROOT / "results" / "ipo_optimizer_summary.txt"
ONLINE_PATH = ROOT / "results" / "ipo_optimizer_online_path.csv"
UPDATE_BENEFIT = ROOT / "results" / "update_benefit_summary.json"


def parse_summary(summary_path: Path) -> dict:
    txt = summary_path.read_text()
    out: dict[str, float] = {}
    pats = {
        "net_total_return": r"Net of costs:\s+Total return:\s+([-\d\.]+)%",
        "net_ann_return": r"Net of costs:\s+.*?Return \(annualized\):\s+([-\d\.]+)%",
        "net_ann_vol": r"Net of costs:\s+.*?Volatility \(annual\):\s+([-\d\.]+)%",
        "net_sharpe": r"Net of costs:\s+.*?Sharpe \(annualized\):\s+([-\d\.]+)",
        "net_max_dd": r"Net of costs:\s+.*?Max drawdown:\s+([-\d\.]+)%",
    }
    for k, p in pats.items():
        m = re.search(p, txt, flags=re.S)
        out[k] = float(m.group(1)) if m else float("nan")
    return out


def run_one(
    cadence: str,
    lookback: int,
    epochs_step: int,
    gate_mode: str,
    gate_min_val_improvement: float,
    gate_min_relative_improvement: float,
    gate_min_history_windows: int,
) -> dict:
    env = os.environ.copy()
    env["IPO_MODE"] = "online"
    env["IPO_ONLINE_UPDATE_FREQ"] = cadence
    env["IPO_ONLINE_LOOKBACK"] = str(lookback)
    env["IPO_ONLINE_EPOCHS_STEP"] = str(epochs_step)
    env["IPO_ONLINE_STOP_ON_DETERIORATION"] = "1"
    env["IPO_UPDATE_GATE_MODE"] = str(gate_mode)
    env["IPO_GATE_MIN_VAL_IMPROVEMENT"] = str(gate_min_val_improvement)
    env["IPO_GATE_MIN_RELATIVE_IMPROVEMENT"] = str(gate_min_relative_improvement)
    env["IPO_GATE_MIN_HISTORY_WINDOWS"] = str(gate_min_history_windows)
    # Use the currently active interpreter (e.g., project .venv), not system python.
    cmd = [sys.executable, str(RUNNER)]
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    row = {
        "cadence": cadence,
        "lookback": lookback,
        "epochs_step": epochs_step,
        "gate_mode": gate_mode,
        "gate_min_val_improvement": gate_min_val_improvement,
        "gate_min_relative_improvement": gate_min_relative_improvement,
        "gate_min_history_windows": gate_min_history_windows,
        "exit_code": proc.returncode,
    }
    if proc.returncode == 0 and SUMMARY.exists():
        row.update(parse_summary(SUMMARY))
        if ONLINE_PATH.exists():
            path_df = pd.read_csv(ONLINE_PATH)
            row["n_updates_applied"] = int(path_df["was_model_updated"].astype(int).sum())
            row["n_decisions"] = int(len(path_df))
        if UPDATE_BENEFIT.exists():
            try:
                b = json.loads(UPDATE_BENEFIT.read_text())
                row["update_accept_rate"] = float(b.get("update_accept_rate", float("nan")))
                row["mean_post_update_return"] = float(
                    b.get("mean_post_update_return", float("nan"))
                )
                row["mean_post_no_update_return"] = float(
                    b.get("mean_post_no_update_return", float("nan"))
                )
                row["difference_update_minus_no_update"] = float(
                    b.get("difference_update_minus_no_update", float("nan"))
                )
            except Exception:
                pass
    else:
        row["stderr_tail"] = (proc.stderr or "")[-800:]
        row["stdout_tail"] = (proc.stdout or "")[-800:]
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cadences", nargs="+", default=["D", "W", "M"])
    p.add_argument("--lookbacks", nargs="+", type=int, default=[0, 252, 504])
    p.add_argument("--epochs-step", type=int, default=2)
    p.add_argument("--gate-modes", nargs="+", default=["cadence", "confidence"])
    p.add_argument("--gate-min-val-improvement", type=float, default=0.0)
    p.add_argument("--gate-min-relative-improvement", type=float, default=0.0)
    p.add_argument("--gate-min-history-windows", type=int, default=252)
    p.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "results" / "online_sweep_results.csv",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for gm in args.gate_modes:
        for c in args.cadences:
            for lb in args.lookbacks:
                print(
                    f"Running gate={gm} cadence={c} lookback={lb} epochs_step={args.epochs_step} ...",
                    flush=True,
                )
                rows.append(
                    run_one(
                        cadence=c,
                        lookback=lb,
                        epochs_step=args.epochs_step,
                        gate_mode=gm,
                        gate_min_val_improvement=args.gate_min_val_improvement,
                        gate_min_relative_improvement=args.gate_min_relative_improvement,
                        gate_min_history_windows=args.gate_min_history_windows,
                    )
                )
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
