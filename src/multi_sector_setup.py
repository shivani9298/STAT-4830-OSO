"""
Helpers for a global multi-sector allocator:

- One model output per day with a single softmax over
  [market, sector_1, ..., sector_G].
- Rolling windows target realized returns for all assets at once.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .export import baseline_stats_same_R, portfolio_stats


def build_rolling_windows_multi_asset(
    df: pd.DataFrame,
    window_len: int,
    feature_cols: list[str],
    return_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build windows for a global allocator with n_assets = len(return_cols).

    Returns:
        X: (N, T, F)
        R: (N, n_assets)
        dates: (N,)
    """
    arr_x = df[feature_cols].values.astype(np.float32)
    arr_x = np.nan_to_num(arr_x, nan=0.0, posinf=0.0, neginf=0.0)
    arr_r = df[return_cols].values.astype(np.float32)
    arr_r = np.nan_to_num(arr_r, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(df)
    if n <= window_len:
        return (
            np.zeros((0, window_len, len(feature_cols)), np.float32),
            np.zeros((0, len(return_cols)), np.float32),
            np.array([], dtype=object),
        )

    X_list, R_list, d_list = [], [], []
    for i in range(window_len, n):
        X_list.append(arr_x[i - window_len : i])
        R_list.append(arr_r[i])
        d_list.append(df.index[i])
    return np.stack(X_list, axis=0), np.stack(R_list, axis=0), np.array(d_list)


def export_multi_sector_outputs(
    dates: np.ndarray,
    weights: np.ndarray,
    returns: np.ndarray,
    asset_names: list[str],
    out_dir: str | Path,
    prefix: str = "multisector_allocator",
) -> dict:
    """
    Export weights and a compact summary for the global multi-sector allocator.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w_df = pd.DataFrame(
        weights,
        index=pd.DatetimeIndex(dates),
        columns=[f"weight_{name}" for name in asset_names],
    )
    w_df.index.name = "date"
    w_path = out_dir / f"{prefix}_weights.csv"
    w_df.to_csv(w_path)

    stats = portfolio_stats(weights, returns)
    avg_w = np.mean(weights, axis=0)
    max_w = np.max(weights, axis=0)

    summary_path = out_dir / f"{prefix}_summary.txt"
    with summary_path.open("w") as f:
        f.write("Global Multi-Sector Allocator - Summary\n")
        f.write("======================================\n")
        f.write(f"Assets: {', '.join(asset_names)}\n")
        f.write(f"Mean daily return:     {stats['mean_return_daily']:.4%}\n")
        f.write(f"Volatility (daily):    {stats['volatility_daily']:.4%}\n")
        f.write(f"Sharpe (annualized):   {stats['sharpe_annualized']:.2f}\n")
        f.write(f"Max drawdown:          {stats['max_drawdown']:.2%}\n")
        f.write(f"Avg turnover:          {stats['avg_turnover']:.4f}\n\n")
        f.write("Average weights by asset:\n")
        for name, val in zip(asset_names, avg_w):
            f.write(f"  {name:24s} {val:.2%}\n")
        f.write("\nPeak weights by asset:\n")
        for name, val in zip(asset_names, max_w):
            f.write(f"  {name:24s} {val:.2%}\n")
        f.write("\nStatic baselines (same R_val, same dates; zero turnover vs model):\n")
        for label, bs in baseline_stats_same_R(returns):
            f.write(
                f"  {label}\n"
                f"    Sharpe={bs['sharpe_annualized']:.2f}  "
                f"TotalRet={bs['total_return']:.2%}  "
                f"MaxDD={bs['max_drawdown']:.2%}\n"
            )

    return {
        "weights_path": str(w_path),
        "summary_path": str(summary_path),
        "stats": stats,
    }

