#!/usr/bin/env python3
"""
Regenerate the **per-window (dated) validation** chart: cumulative wealth vs 50/50 — *not* epoch loss.

Requires ``ipo_optimizer_returns_val*.csv`` (written on the next WRDS run) merged with weights.
If you only have ``ipo_optimizer_weights.csv`` from an older run, re-run training once so
``export_window_returns_csv`` saves returns, or merge weights with any CSV that has
``date``, ``market_return``, ``ipo_return``.

  python3 scripts/replot_daily_vs_50_50.py
  python3 scripts/replot_daily_vs_50_50.py --model gru --out figures/ipo_optimizer/gru/validation_returns_vs_equal_weight.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.export import model_csv_suffix
from src.plot_loss import plot_cumulative_returns_vs_equal_weight


def _suffix_for_arg(model: str) -> str:
    return model_csv_suffix(model)


def load_merged(results_dir: Path, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    suf = _suffix_for_arg(model)
    w_candidates = [
        results_dir / f"ipo_optimizer_weights_val{suf}.csv",
        results_dir / "ipo_optimizer_weights.csv",
    ]
    w_path = next((p for p in w_candidates if p.is_file()), None)
    r_path = results_dir / f"ipo_optimizer_returns_val{suf}.csv"
    if w_path is None:
        raise FileNotFoundError(
            f"No weights CSV found (tried {w_candidates[0].name} and {w_candidates[1].name})"
        )
    if not r_path.is_file():
        raise FileNotFoundError(
            f"Missing {r_path.name}. Run ``scripts/run_ipo_optimizer_wrds.py`` once (or merge returns "
            "with dates aligned to weights)."
        )
    w = pd.read_csv(w_path)
    r = pd.read_csv(r_path)
    if "date" not in w.columns:
        raise ValueError(f"{w_path} must have a date column")
    w["date"] = pd.to_datetime(w["date"])
    r["date"] = pd.to_datetime(r["date"])
    df = w.merge(r, on="date", how="inner").sort_values("date")
    if len(df) == 0:
        raise ValueError("No overlapping dates between weights and returns CSVs")
    cols_w = [c for c in df.columns if c.startswith("weight_") or c.startswith("weight")]
    if "weight_market" in df.columns and "weight_IPO" in df.columns:
        W = df[["weight_market", "weight_IPO"]].values.astype(np.float64)
    elif len(cols_w) >= 2:
        W = df[cols_w[:2]].values.astype(np.float64)
    else:
        raise ValueError("Could not find weight_market / weight_IPO columns")
    if "market_return" not in df.columns or "ipo_return" not in df.columns:
        raise ValueError("Returns CSV must include market_return and ipo_return")
    R = df[["market_return", "ipo_return"]].values.astype(np.float64)
    dates = df["date"].values
    return W, R, dates


def main() -> int:
    p = argparse.ArgumentParser(description="Replot validation cumulative wealth vs 50/50 from CSVs.")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--model", type=str, default="gru", help="gru | lstm | transformer (CSV suffix)")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: figures/ipo_optimizer/<model>/validation_returns_vs_equal_weight.png)",
    )
    p.add_argument("--no-excess-panel", action="store_true", help="Single panel only (wealth lines)")
    args = p.parse_args()

    try:
        W, R, dates = load_merged(args.results_dir, args.model)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    model_tag = str(args.model).strip().lower() or "gru"
    out = args.out
    if out is None:
        out = ROOT / "figures" / "ipo_optimizer" / model_tag / "validation_returns_vs_equal_weight.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    title = "Validation: cumulative growth vs 50/50 market/IPO"
    path = plot_cumulative_returns_vs_equal_weight(
        W,
        R,
        dates,
        out,
        title=title,
        excess_panel=not args.no_excess_panel,
    )
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
