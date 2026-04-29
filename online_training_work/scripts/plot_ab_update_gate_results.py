#!/usr/bin/env python3
"""
Plot A/B online update-gate sweep results from CSV.

If performance metrics are present (e.g., net_total_return/net_sharpe), generate
a 2x2 metric comparison chart. Otherwise, generate a run-status heatmap based on
exit codes so failed sweeps are still visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in-csv",
        type=Path,
        default=ROOT / "results" / "ab_update_gate_results.csv",
        help="A/B sweep results CSV path",
    )
    p.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "figures" / "online_evaluation" / "ab_update_gate_results.png",
        help="Output figure path",
    )
    return p.parse_args()


def plot_status_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    tbl = (
        df.assign(status=np.where(df["exit_code"].fillna(1).astype(int) == 0, 1, 0))
        .pivot_table(index="gate_mode", columns="lookback", values="status", aggfunc="max")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    mat = tbl.to_numpy(dtype=float)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(tbl.columns)))
    ax.set_xticklabels([str(c) for c in tbl.columns])
    ax.set_yticks(np.arange(len(tbl.index)))
    ax.set_yticklabels([str(i) for i in tbl.index])
    ax.set_xlabel("Lookback windows")
    ax.set_ylabel("Gate mode")
    ax.set_title("A/B Sweep Run Status (1=success, 0=failed)")

    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            ax.text(c, r, f"{int(mat[r, c])}", ha="center", va="center", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Run status")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(df: pd.DataFrame, out_path: Path) -> None:
    metric_specs = [
        ("net_total_return", "Net Total Return (%)", True),
        ("net_sharpe", "Net Sharpe", False),
        ("net_max_dd", "Net Max Drawdown (%)", True),
        ("update_accept_rate", "Update Accept Rate", False),
    ]

    gdf = df.copy()
    gdf["label"] = gdf["gate_mode"].astype(str) + " | lb=" + gdf["lookback"].astype(str)
    x = np.arange(len(gdf))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.ravel()
    for ax, (col, title, as_pct) in zip(axes, metric_specs):
        y = pd.to_numeric(gdf[col], errors="coerce").to_numpy(dtype=float)
        bars = ax.bar(x, y, color=["#1f77b4" if m == "cadence" else "#ff7f0e" for m in gdf["gate_mode"]])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(gdf["label"], rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        if as_pct:
            ax.axhline(0, color="k", lw=0.8, ls=":")
        for b, val in zip(bars, y):
            if np.isfinite(val):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    fig.suptitle("A/B Sweep: Cadence vs Confidence", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.in_csv.is_file():
        raise FileNotFoundError(f"Missing input CSV: {args.in_csv}")

    df = pd.read_csv(args.in_csv)
    metrics_needed = {"net_total_return", "net_sharpe", "net_max_dd", "update_accept_rate"}
    has_metrics = metrics_needed.issubset(df.columns)

    if has_metrics and (df["exit_code"].fillna(1).astype(int) == 0).any():
        plot_metrics(df[df["exit_code"].fillna(1).astype(int) == 0], args.out_png)
    else:
        plot_status_heatmap(df, args.out_png)

    print(f"Wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
