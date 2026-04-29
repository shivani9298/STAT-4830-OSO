#!/usr/bin/env python3
"""
Plot cumulative return trajectories for cadence vs confidence and static 50/50.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cadence-csv",
        type=Path,
        default=ROOT / "results" / "online_path_cadence_lb504.csv",
    )
    p.add_argument(
        "--confidence-csv",
        type=Path,
        default=ROOT / "results" / "online_path_confidence_lb504.csv",
    )
    p.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "figures" / "online_evaluation" / "ab_return_trajectory_vs_5050.png",
    )
    return p.parse_args()


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").set_index("date")
    required = {"net_ret", "realized_market", "realized_ipo"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def _cumret(r: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + r) - 1.0


def _pct(x: np.ndarray) -> np.ndarray:
    return x * 100.0


def main() -> int:
    args = parse_args()
    cad = _load(args.cadence_csv)
    conf = _load(args.confidence_csv)

    idx = cad.index.intersection(conf.index)
    if len(idx) == 0:
        raise ValueError("No overlapping dates between cadence and confidence paths.")
    cad = cad.loc[idx]
    conf = conf.loc[idx]

    # Static 50/50 daily return, no rebalance cost approximation.
    ret_5050 = 0.5 * cad["realized_market"].to_numpy(dtype=float) + 0.5 * cad["realized_ipo"].to_numpy(dtype=float)

    cad_cum = _pct(_cumret(cad["net_ret"].to_numpy(dtype=float)))
    conf_cum = _pct(_cumret(conf["net_ret"].to_numpy(dtype=float)))
    eq_cum = _pct(_cumret(ret_5050))

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(idx, cad_cum, color="#1f77b4", lw=2.2, label="Cadence gate (net)")
    ax.plot(idx, conf_cum, color="#ff7f0e", lw=2.0, label="Confidence gate (net)")
    ax.plot(idx, eq_cum, color="#2ca02c", lw=1.8, ls="--", label="Static 50/50")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_title("Return Trajectory: Cadence vs Confidence vs Static 50/50")
    ax.set_ylabel("Cumulative Return (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {args.out_png}")
    print(f"Final cadence cumulative return: {cad_cum[-1]:.2f}%")
    print(f"Final confidence cumulative return: {conf_cum[-1]:.2f}%")
    print(f"Final static 50/50 cumulative return: {eq_cum[-1]:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
