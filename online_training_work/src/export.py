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
def predict_sector_head_weights(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Model forward (B, T, F) -> (B, G, 2). Returns (N, G, 2) numpy."""
    model.eval()
    out = []
    for start in range(0, X.shape[0], batch_size):
        x = torch.as_tensor(X[start : start + batch_size], device=device, dtype=torch.float32)
        w = model(x)
        out.append(w.cpu().numpy())
    return np.concatenate(out, axis=0)


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
    weights = np.asarray(weights, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if np.isnan(weights).any() or np.isnan(R).any():
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
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


def portfolio_stats_from_path(
    net_returns: np.ndarray,
    *,
    turnover: Optional[np.ndarray] = None,
) -> dict:
    """Aggregate statistics from a realized return path."""
    r = np.asarray(net_returns, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return {
            "mean_return_daily": 0.0,
            "volatility_daily": 0.0,
            "total_return": 0.0,
            "return_annualized": 0.0,
            "volatility_annualized": 0.0,
            "sharpe_annualized": 0.0,
            "max_drawdown": 0.0,
            "avg_turnover": 0.0,
        }
    mean_ret = float(np.mean(r))
    vol = float(np.std(r))
    sharpe = (mean_ret / vol * np.sqrt(252)) if vol > 1e-8 else 0.0
    n_days = len(r)
    total_ret = float(np.prod(1 + r) - 1.0)
    ann_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1 if n_days > 0 else 0.0
    ann_vol = vol * np.sqrt(252) if vol > 1e-8 else 0.0
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth / peak) - 1.0
    max_dd = float(np.min(dd))
    if turnover is None:
        avg_turnover = 0.0
    else:
        t = np.asarray(turnover, dtype=np.float64).reshape(-1)
        avg_turnover = float(np.mean(t)) if t.size > 0 else 0.0
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


def evaluate_online_path(
    weights_path: np.ndarray,
    R_path: np.ndarray,
    *,
    cost_bps: float = 0.0,
    initial_weights: Optional[np.ndarray] = None,
) -> dict:
    """
    Evaluate a sequential online policy path with transaction costs.

    Returns gross/net path arrays and summary stats.
    """
    w = np.asarray(weights_path, dtype=np.float64)
    r = np.asarray(R_path, dtype=np.float64)
    if w.ndim != 2 or r.ndim != 2 or w.shape[0] != r.shape[0]:
        raise ValueError("weights_path and R_path must both be (N, n_assets) with equal N")
    if w.shape[0] == 0:
        empty = np.array([], dtype=np.float64)
        return {
            "gross_returns": empty,
            "net_returns": empty,
            "turnover": empty,
            "cost": empty,
            "gross_stats": portfolio_stats_from_path(empty),
            "net_stats": portfolio_stats_from_path(empty),
        }

    prev = np.asarray(initial_weights, dtype=np.float64) if initial_weights is not None else w[0]
    if prev.shape != w[0].shape:
        raise ValueError("initial_weights shape must match one row of weights_path")

    turnover = np.zeros((w.shape[0],), dtype=np.float64)
    for i in range(w.shape[0]):
        turnover[i] = float(np.abs(w[i] - prev).sum())
        prev = w[i]

    gross = np.sum(w * r, axis=1)
    cost_rate = float(cost_bps) / 1e4
    cost = turnover * cost_rate
    net = gross - cost
    return {
        "gross_returns": gross,
        "net_returns": net,
        "turnover": turnover,
        "cost": cost,
        "gross_stats": portfolio_stats_from_path(gross, turnover=turnover),
        "net_stats": portfolio_stats_from_path(net, turnover=turnover),
    }


def export_online_path_csv(
    dates: np.ndarray,
    weights: np.ndarray,
    realized_returns: np.ndarray,
    metrics: dict,
    out_path: str | Path,
    *,
    was_model_updated: Optional[np.ndarray] = None,
) -> None:
    """Export per-date online action/evaluation log."""
    w = np.asarray(weights, dtype=np.float64)
    r = np.asarray(realized_returns, dtype=np.float64)
    gross = np.asarray(metrics["gross_returns"], dtype=np.float64)
    net = np.asarray(metrics["net_returns"], dtype=np.float64)
    cost = np.asarray(metrics["cost"], dtype=np.float64)
    turn = np.asarray(metrics["turnover"], dtype=np.float64)
    n = w.shape[0]
    if was_model_updated is None:
        upd = np.zeros((n,), dtype=bool)
    else:
        upd = np.asarray(was_model_updated).astype(bool)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "weight_market": w[:, 0],
            "weight_ipo": w[:, 1] if w.shape[1] > 1 else 0.0,
            "realized_market": r[:, 0],
            "realized_ipo": r[:, 1] if r.shape[1] > 1 else 0.0,
            "gross_ret": gross,
            "cost": cost,
            "net_ret": net,
            "turnover": turn,
            "was_model_updated": upd.astype(int),
        }
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


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


def model_csv_suffix(model_tag: str) -> str:
    """``gru`` -> ``""``; ``lstm`` -> ``"_lstm"`` (matches compare / holdout scripts)."""
    t = str(model_tag).strip().lower() or "gru"
    return "" if t == "gru" else f"_{t}"


def export_window_returns_csv(
    dates: np.ndarray,
    R: np.ndarray,
    out_path: str | Path,
) -> None:
    """
    One row per rolling window (same order as ``export_weights_csv``): ``date``, ``market_return``,
    ``ipo_return``, ``equal_weight_return`` (50/50 blend of the two assets that day).

    ``R`` can be ``(N, 2)`` or sector tensor ``(N, G, 2)`` (means across sleeves for a single benchmark series).
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 3:
        R = R.mean(axis=1)
    if R.ndim != 2 or R.shape[1] < 2:
        raise ValueError("R must be (N, 2) or (N, G, 2)")
    m = R[:, 0]
    ipo = R[:, 1]
    eq = 0.5 * (m + ipo)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "market_return": m,
            "ipo_return": ipo,
            "equal_weight_return": eq,
        }
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _stats_line(name: str, s: dict) -> str:
    """Format a stats dict as a compact line."""
    return (f"  {name:12} Total={s['total_return']:.2%}  AnnRet={s['return_annualized']:.2%}  "
            f"AnnVol={s['volatility_annualized']:.2%}  Sharpe={s['sharpe_annualized']:.2f}  MaxDD={s['max_drawdown']:.2%}")


def export_summary(
    stats: dict,
    weights: np.ndarray,
    out_path: str | Path,
    R: Optional[np.ndarray] = None,
) -> None:
    """Write a short summary for the retail trader.
    If R is provided (N, 2) with [market_return, ipo_return], includes baseline comparison.
    """
    avg_ipo = float(np.mean(weights[:, 1])) if weights.shape[1] > 1 else 0.0
    pct_high_ipo = float(np.mean(weights[:, 1] > 0.2)) * 100.0 if weights.shape[1] > 1 else 0.0
    lines = [
        "IPO Portfolio Optimizer — Summary",
        "==================================",
        "",
        "Model Portfolio:",
        f"  Total return:          {stats.get('total_return', 0):.2%}",
        f"  Return (annualized):  {stats.get('return_annualized', 0):.2%}",
        f"  Volatility (annual):  {stats.get('volatility_annualized', 0):.2%}",
        f"  Sharpe (annualized):  {stats.get('sharpe_annualized', 0):.2f}",
        f"  Max drawdown:         {stats.get('max_drawdown', 0):.2%}",
        f"  Avg turnover:         {stats.get('avg_turnover', 0):.4f}",
        f"  Average IPO weight:   {avg_ipo:.2%}",
        f"  % days IPO weight>20%: {pct_high_ipo:.1f}%",
    ]
    if R is not None and R.shape[1] >= 2:
        # Baselines: SPY-only [1,0], IPO-only [0,1], Equal-weight [0.5, 0.5]
        w_spy = np.tile([1.0, 0.0], (R.shape[0], 1))
        w_ipo = np.tile([0.0, 1.0], (R.shape[0], 1))
        w_eq = np.tile([0.5, 0.5], (R.shape[0], 1))
        s_spy = portfolio_stats(w_spy, R)
        s_ipo = portfolio_stats(w_ipo, R)
        s_eq = portfolio_stats(w_eq, R)
        lines.extend([
            "",
            "Baseline Comparison (same period):",
            _stats_line("Market only:", s_spy),
            _stats_line("IPO only:", s_ipo),
            _stats_line("Equal 50/50:", s_eq),
        ])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))


def export_online_summary(
    net_stats: dict,
    gross_stats: dict,
    out_path: str | Path,
    *,
    baseline_stats: Optional[dict[str, dict]] = None,
) -> None:
    """Write a compact summary for online path results."""
    lines = [
        "IPO Portfolio Optimizer — Online Summary",
        "========================================",
        "",
        "Net of costs:",
        f"  Total return:          {net_stats.get('total_return', 0):.2%}",
        f"  Return (annualized):  {net_stats.get('return_annualized', 0):.2%}",
        f"  Volatility (annual):  {net_stats.get('volatility_annualized', 0):.2%}",
        f"  Sharpe (annualized):  {net_stats.get('sharpe_annualized', 0):.2f}",
        f"  Max drawdown:         {net_stats.get('max_drawdown', 0):.2%}",
        f"  Avg turnover:         {net_stats.get('avg_turnover', 0):.4f}",
        "",
        "Gross (before costs):",
        f"  Total return:          {gross_stats.get('total_return', 0):.2%}",
        f"  Return (annualized):  {gross_stats.get('return_annualized', 0):.2%}",
        f"  Volatility (annual):  {gross_stats.get('volatility_annualized', 0):.2%}",
        f"  Sharpe (annualized):  {gross_stats.get('sharpe_annualized', 0):.2f}",
        f"  Max drawdown:         {gross_stats.get('max_drawdown', 0):.2%}",
    ]
    if baseline_stats:
        lines.extend(["", "Baselines (same online dates):"])
        for name, s in baseline_stats.items():
            lines.append(_stats_line(name, s))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))


def export_sector_group_outputs(
    dates: np.ndarray,
    weights: np.ndarray,
    R: np.ndarray,
    sector_labels: list[str],
    out_dir: str | Path,
    sanitize_fn=None,
) -> None:
    """
    ``weights``: (N, G, 2), ``R``: (N, G, 2). Writes one CSV per sector and a combined summary.

    ``sanitize_fn``: optional(str) -> str for filenames; default keeps alnum + underscore.
    """
    from re import sub

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _safe(s: str) -> str:
        if sanitize_fn:
            return sanitize_fn(s)
        return sub(r"[^A-Za-z0-9_]+", "_", s).strip("_") or "sector"

    summary_lines = [
        "IPO Optimizer — Per-sector portfolios (market vs sector IPO basket)",
        "Each sector: separate softmax over [market, IPO basket]; shared RNN state.",
        "",
    ]
    for g, label in enumerate(sector_labels):
        tag = _safe(label)
        w_g = weights[:, g, :]
        R_g = R[:, g, :]
        stats_g = portfolio_stats(w_g, R_g)
        export_weights_csv(
            dates,
            w_g,
            out_dir / f"ipo_optimizer_weights_sector_{tag}.csv",
            asset_names=["weight_market", f"weight_ipo_{tag}"],
        )
        summary_lines.append(f"Sector: {label}")
        summary_lines.append(
            f"  Sharpe={stats_g['sharpe_annualized']:.2f}  MaxDD={stats_g['max_drawdown']:.2%}  "
            f"AnnRet={stats_g['return_annualized']:.2%}  Avg IPO weight={float(np.mean(w_g[:, 1])):.2%}"
        )
        summary_lines.append("")
    (out_dir / "ipo_optimizer_summary_by_sector.txt").write_text("\n".join(summary_lines))
