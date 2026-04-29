#!/usr/bin/env python3
"""
Create evaluation figures for online-policy backtest outputs.

Input:
  - results/ipo_optimizer_online_path.csv

Outputs (default):
  - figures/online_evaluation/online_cumulative_returns.png
  - figures/online_evaluation/online_drawdowns.png
  - figures/online_evaluation/online_allocations_updates.png
  - figures/online_evaluation/online_turnover_cost.png
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
        "--online-csv",
        type=Path,
        default=ROOT / "results" / "ipo_optimizer_online_path.csv",
        help="Online path CSV from run_ipo_optimizer_wrds.py",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "figures" / "online_evaluation",
        help="Directory to write generated figures",
    )
    p.add_argument(
        "--cost-bps",
        type=float,
        default=5.0,
        help="Cost rate (bps) for baseline net-return approximations",
    )
    return p.parse_args()


def cumulative_from_returns(r: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + r) - 1.0


def drawdown_from_returns(r: np.ndarray) -> np.ndarray:
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    return wealth / peak - 1.0


def load_online(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").set_index("date")
    required = {
        "weight_market",
        "weight_ipo",
        "realized_market",
        "realized_ipo",
        "gross_ret",
        "cost",
        "net_ret",
        "turnover",
        "was_model_updated",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    return df


def add_baselines(df: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    out = df.copy()
    r_m = out["realized_market"].to_numpy(dtype=float)
    r_i = out["realized_ipo"].to_numpy(dtype=float)
    out["ret_market"] = r_m
    out["ret_ipo"] = r_i
    out["ret_5050"] = 0.5 * r_m + 0.5 * r_i

    # Approximate baseline net returns using same turnover/cost framework.
    cost_rate = float(cost_bps) / 1e4
    # Market-only and IPO-only have one initial rebalance then zero turnover.
    turn_m = np.zeros_like(r_m)
    turn_i = np.zeros_like(r_i)
    turn_5050 = np.zeros_like(r_i)
    turn_m[0] = 1.0
    turn_i[0] = 1.0
    turn_5050[0] = 0.0
    out["ret_market_net"] = out["ret_market"] - turn_m * cost_rate
    out["ret_ipo_net"] = out["ret_ipo"] - turn_i * cost_rate
    out["ret_5050_net"] = out["ret_5050"] - turn_5050 * cost_rate
    return out


def plot_cumulative(df: pd.DataFrame, out_path: Path) -> None:
    t = df.index
    model_net = cumulative_from_returns(df["net_ret"].to_numpy(dtype=float)) * 100
    model_gross = cumulative_from_returns(df["gross_ret"].to_numpy(dtype=float)) * 100
    mkt = cumulative_from_returns(df["ret_market_net"].to_numpy(dtype=float)) * 100
    ipo = cumulative_from_returns(df["ret_ipo_net"].to_numpy(dtype=float)) * 100
    eq = cumulative_from_returns(df["ret_5050_net"].to_numpy(dtype=float)) * 100

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, model_net, color="#1f77b4", lw=2.2, label="Model (net)")
    ax.plot(t, model_gross, color="#1f77b4", lw=1.2, ls="--", alpha=0.8, label="Model (gross)")
    ax.plot(t, eq, color="#2ca02c", lw=1.5, ls="--", label="Equal 50/50 (net approx)")
    ax.plot(t, mkt, color="#ff7f0e", lw=1.4, ls="--", label="Market only (net approx)")
    ax.plot(t, ipo, color="#9467bd", lw=1.4, ls="--", label="IPO only (net approx)")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Online Backtest: Cumulative Returns")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_drawdowns(df: pd.DataFrame, out_path: Path) -> None:
    t = df.index
    dd_model = drawdown_from_returns(df["net_ret"].to_numpy(dtype=float)) * 100
    dd_mkt = drawdown_from_returns(df["ret_market_net"].to_numpy(dtype=float)) * 100
    dd_ipo = drawdown_from_returns(df["ret_ipo_net"].to_numpy(dtype=float)) * 100
    dd_eq = drawdown_from_returns(df["ret_5050_net"].to_numpy(dtype=float)) * 100

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, dd_model, color="#1f77b4", lw=2.0, label="Model (net)")
    ax.plot(t, dd_eq, color="#2ca02c", lw=1.4, ls="--", label="Equal 50/50 (net approx)")
    ax.plot(t, dd_mkt, color="#ff7f0e", lw=1.2, ls="--", label="Market only (net approx)")
    ax.plot(t, dd_ipo, color="#9467bd", lw=1.2, ls="--", label="IPO only (net approx)")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Online Backtest: Drawdown Comparison")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_allocations_updates(df: pd.DataFrame, out_path: Path) -> None:
    t = df.index
    w_m = df["weight_market"].to_numpy(dtype=float) * 100
    w_i = df["weight_ipo"].to_numpy(dtype=float) * 100
    upd_mask = df["was_model_updated"].to_numpy(dtype=int) > 0

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, w_m, color="#1f77b4", lw=1.8, label="Weight market")
    ax.plot(t, w_i, color="#ff7f0e", lw=1.8, label="Weight IPO")
    if upd_mask.any():
        ax.scatter(
            t[upd_mask],
            w_i[upd_mask],
            color="#d62728",
            s=14,
            alpha=0.7,
            label="Model update dates",
            zorder=5,
        )
    ax.set_ylabel("Weight (%)")
    ax.set_title("Online Backtest: Allocation Path and Update Dates")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_turnover_cost(df: pd.DataFrame, out_path: Path) -> None:
    t = df.index
    turn = df["turnover"].to_numpy(dtype=float)
    cost = df["cost"].to_numpy(dtype=float) * 10000.0  # return points in bps
    roll = pd.Series(turn, index=t).rolling(21, min_periods=1).mean()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(t, turn, color="#1f77b4", lw=1.2, alpha=0.8, label="Daily turnover")
    axes[0].plot(t, roll.values, color="#d62728", lw=1.8, label="21-day avg turnover")
    axes[0].set_ylabel("Turnover")
    axes[0].set_title("Online Backtest: Turnover")
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, cost, color="#9467bd", lw=1.3, label="Daily trading cost (bps)")
    axes[1].set_ylabel("Cost (bps of capital)")
    axes[1].set_title("Online Backtest: Transaction Cost")
    axes[1].legend(fontsize=9, loc="upper left")
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.online_csv.is_file():
        raise FileNotFoundError(f"Missing online path CSV: {args.online_csv}")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_online(args.online_csv)
    df = add_baselines(df, cost_bps=args.cost_bps)

    p1 = out_dir / "online_cumulative_returns.png"
    p2 = out_dir / "online_drawdowns.png"
    p3 = out_dir / "online_allocations_updates.png"
    p4 = out_dir / "online_turnover_cost.png"

    plot_cumulative(df, p1)
    plot_drawdowns(df, p2)
    plot_allocations_updates(df, p3)
    plot_turnover_cost(df, p4)

    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    print(f"Wrote {p3}")
    print(f"Wrote {p4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
