#!/usr/bin/env python3
"""
Walk-forward style WRDS training: for each anchor date, run run_ipo_optimizer_wrds.py
with calendar splits (validation ends at anchor, test starts at anchor).

Copies each run's selection JSON to results/wf_<anchor>_selection_metrics*.json and
writes results/walk_forward_summary_paths.csv with validation/test path metrics.

Example:
  python3 scripts/walk_forward_train_wrds.py --use-cache --max-history \\
    --anchors 2018-01-01 2020-01-01 2022-01-01 \\
    --models gru lstm --val-months 18 \\
    --selection-metric val_retail_composite --selection-drawdown-penalty 1.0
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "run_ipo_optimizer_wrds.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--anchors",
        nargs="+",
        required=True,
        help="Test-window start dates YYYY-MM-DD (validation uses data before this).",
    )
    p.add_argument(
        "--val-months",
        type=int,
        default=12,
        help="Approximate calendar months of validation before each anchor.",
    )
    p.add_argument("--models", nargs="+", default=["gru"])
    p.add_argument("--use-cache", action="store_true")
    p.add_argument("--cache-dir", default=str(ROOT / "results" / "cache_wrds"))
    p.add_argument("--start-date", default="2010-01-01")
    p.add_argument("--end-date", default="2025-12-31")
    p.add_argument("--max-history", action="store_true")
    p.add_argument("--selection-metric", default="val_retail_composite")
    p.add_argument("--selection-drawdown-penalty", type=float, default=1.0)
    p.add_argument("remainder", nargs=argparse.REMAINDER, help="Extra args after --")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    extras = list(args.remainder)
    if extras and extras[0] == "--":
        extras = extras[1:]

    flat_rows: list[dict] = []
    for anchor in args.anchors:
        ts = pd.Timestamp(anchor)
        val_ts = ts - pd.DateOffset(months=int(args.val_months))
        val_start = val_ts.strftime("%Y-%m-%d")
        test_start = ts.strftime("%Y-%m-%d")
        tag = test_start.replace("-", "")

        for model in args.models:
            cmd = [
                sys.executable,
                "-u",
                str(RUNNER),
                "--model",
                model,
                "--start-date",
                args.start_date,
                "--end-date",
                args.end_date,
                "--val-start",
                val_start,
                "--test-start",
                test_start,
                "--selection-metric",
                args.selection_metric,
                "--selection-drawdown-penalty",
                str(args.selection_drawdown_penalty),
            ]
            if args.max_history:
                cmd.append("--max-history")
            if args.use_cache:
                cmd.extend(["--use-cache", "--cache-dir", args.cache_dir])
            cmd.extend(extras)
            print(f"[walk-forward] anchor={test_start} val_start={val_start} model={model}", flush=True)
            print("  " + " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=ROOT)

            suf = "" if model == "gru" else f"_{model}"
            src = ROOT / "results" / f"ipo_optimizer_selection_metrics{suf}.json"
            dst = ROOT / "results" / f"wf_{tag}_selection_metrics{suf}.json"
            shutil.copy2(src, dst)

            with open(src) as f:
                payload = json.load(f)
            pt = payload.get("final_path_metrics_test") or {}
            pv = payload.get("final_path_metrics_validation") or {}
            flat_rows.append(
                {
                    "anchor": test_start,
                    "val_start": val_start,
                    "model": model,
                    "selection_metric": payload.get("selection_metric"),
                    "val_sharpe": pv.get("sharpe"),
                    "val_sortino": pv.get("sortino"),
                    "val_compound_return": pv.get("compound_return"),
                    "val_max_drawdown": pv.get("max_drawdown"),
                    "test_sharpe": pt.get("sharpe"),
                    "test_sortino": pt.get("sortino"),
                    "test_compound_return": pt.get("compound_return"),
                    "test_max_drawdown": pt.get("max_drawdown"),
                    "saved_json": str(dst),
                }
            )

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "walk_forward_summary_paths.csv"
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
