"""
Inference and export: daily weights CSV and summary stats for retail trader.
"""
from __future__ import annotations

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from .model import AllocatorNet


@torch.no_grad()
def predict_weights(
    model: AllocatorNet,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Run model over windows; return (N, n_assets) weights."""
    model.eval()
    out = []
    for start in range(0, X.shape[0], batch_size):
        x = torch.as_tensor(X[start : start + batch_size], device=device, dtype=torch.float32)
        w = model(x)
        out.append(w.cpu().numpy())
    return np.concatenate(out, axis=0)


def portfolio_stats(weights: np.ndarray, R: np.ndarray) -> dict:
    """Compute mean return, vol, Sharpe, max drawdown, avg turnover from weights and returns."""
    port_ret = (weights * R).sum(axis=1)
    mean_ret = float(np.mean(port_ret))
    vol = float(np.std(port_ret))
    sharpe = (mean_ret / vol * np.sqrt(252)) if vol > 1e-8 else 0.0
    # Annual return (compound) and annual volatility
    n_days = len(port_ret)
    total_ret = float(np.prod(1 + port_ret) - 1) if n_days > 0 else 0.0
    ann_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1 if n_days > 0 else 0.0
    ann_vol = vol * np.sqrt(252) if vol > 1e-8 else 0.0
    wealth = np.cumprod(1 + port_ret)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth / peak) - 1.0
    max_dd = float(np.min(dd))
    turnover = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    avg_turnover = float(np.mean(turnover)) if len(turnover) > 0 else 0.0
    return {
        "mean_return_daily": mean_ret,
        "volatility_daily": vol,
        "total_return": total_ret,
        "return_annualized": ann_ret,
        "volatility_annualized": ann_vol,
        "sharpe_annualized": sharpe,
        "max_drawdown": max_dd,
        "avg_turnover": avg_turnover,
    }


def export_weights_csv(
    dates: np.ndarray,
    weights: np.ndarray,
    out_path: str | Path,
    asset_names: Optional[list[str]] = None,
) -> None:
    """Write CSV with columns date, weight_market, weight_IPO (or asset_names)."""
    if asset_names is None:
        asset_names = [f"weight_{i}" for i in range(weights.shape[1])]
    if len(asset_names) == 2:
        asset_names = ["weight_market", "weight_IPO"]
    df = pd.DataFrame(
        weights,
        index=pd.DatetimeIndex(dates),
        columns=asset_names[: weights.shape[1]],
    )
    df.index.name = "date"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)


def export_summary(
    stats: dict,
    weights: np.ndarray,
    out_path: str | Path,
) -> None:
    """Write a short summary for the retail trader."""
    avg_ipo = float(np.mean(weights[:, 1])) if weights.shape[1] > 1 else 0.0
    pct_high_ipo = float(np.mean(weights[:, 1] > 0.2)) * 100.0 if weights.shape[1] > 1 else 0.0
    lines = [
        "IPO Portfolio Optimizer — Summary",
        "==================================",
        f"Total return:          {stats.get('total_return', 0):.2%}",
        f"Return (annualized):  {stats.get('return_annualized', 0):.2%}",
        f"Volatility (annual):  {stats.get('volatility_annualized', 0):.2%}",
        f"Mean daily return:    {stats.get('mean_return_daily', 0):.4%}",
        f"Volatility (daily):   {stats.get('volatility_daily', 0):.4%}",
        f"Sharpe (annualized):  {stats.get('sharpe_annualized', 0):.2f}",
        f"Max drawdown:         {stats.get('max_drawdown', 0):.2%}",
        f"Avg turnover:         {stats.get('avg_turnover', 0):.4f}",
        f"Average IPO weight:   {avg_ipo:.2%}",
        f"% days IPO weight>20%: {pct_high_ipo:.1f}%",
    ]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))
