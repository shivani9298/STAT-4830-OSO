#!/usr/bin/env python3
"""
Train a GLOBAL allocator over multiple sector IPO portfolios + market.

This setup is intentionally separate from run_ipo_optimizer_wrds.py.

Context window: **126 trading days** is the default (and recommended) setting; it is
applied after load_best_config() so tuning JSON does not override it.

Model output each day:
  softmax([market, sector_1, ..., sector_G])  -> shape (G + 1,)

So unlike per-sector heads, this is one allocation simplex across all sleeves.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_ipo_optimizer_wrds import DEFAULTS, END_DATE, START_DATE, load_best_config  # noqa: E402
from src.wrds_data import get_connection  # noqa: E402
from src.data_layer import train_val_split  # noqa: E402
from src.export import predict_weights  # noqa: E402
from src.multisector_data import prepare_multisector_data  # noqa: E402
from src.multi_sector_setup import (  # noqa: E402
    build_rolling_windows_multi_asset,
    export_multi_sector_outputs,
)
from src.train import run_training  # noqa: E402


VAL_FRAC = 0.20
# Canonical context length for this allocator (trading days).
DEFAULT_CONTEXT_WINDOW = 126


def _output_prefix(window_len: int) -> str:
    """Stable names for the default 126d run; disambiguate other lengths with _w{N}."""
    if window_len == DEFAULT_CONTEXT_WINDOW:
        return "multisector_allocator"
    return f"multisector_allocator_w{window_len}"


def _plot_losses(history: list[dict], fig_dir: Path, prefix: str, window_len: int) -> tuple[Path, Path]:
    epochs_x = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig_loss = f"{prefix}_loss.png"
    fig_semilog = f"{prefix}_loss_semilog.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_x, train_loss, label="Train loss", marker="o", markersize=3)
    ax.plot(epochs_x, val_loss, label="Validation loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Global Multi-Sector Allocator: Loss (context={window_len}d)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / fig_loss, dpi=150)
    plt.close(fig)

    t_plot = [max(abs(x), 1e-8) for x in train_loss]
    v_plot = [max(abs(x), 1e-8) for x in val_loss]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs_x, t_plot, label="Train loss", marker="o", markersize=3)
    ax.semilogy(epochs_x, v_plot, label="Validation loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (abs, semilog)")
    ax.set_title(f"Global Multi-Sector Allocator: Loss Semilog (context={window_len}d)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / fig_semilog, dpi=150)
    plt.close(fig)

    return fig_dir / fig_loss, fig_dir / fig_semilog


def _plot_daily_portfolio_return_validation(
    dates_val: np.ndarray,
    weights_val: np.ndarray,
    returns_val: np.ndarray,
    fig_dir: Path,
    prefix: str,
    window_len: int,
) -> Path:
    """
    Per calendar day in the held-out split: realized portfolio return using model weights.

    port_ret_t = sum_i w_{t,i} * r_{t,i}  (not the training val_loss objective).
    """
    dates_idx = pd.DatetimeIndex(dates_val)
    port_ret = np.sum(weights_val * returns_val, axis=1)

    fig_name = f"{prefix}_daily_portfolio_return_validation.png"
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(dates_idx, port_ret, label="Daily portfolio return", linewidth=1.2)
    if len(port_ret) >= 10:
        smooth = pd.Series(port_ret, index=dates_idx).rolling(21, min_periods=5).mean()
        ax.plot(dates_idx, smooth.values, label="21-day moving average", linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(f"Global Multi-Sector Allocator: Daily portfolio return (validation, {window_len}d context)")
    ax.set_xlabel("Date (validation split)")
    ax.set_ylabel("Daily portfolio return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = fig_dir / fig_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _write_comparison(
    results_dir: Path,
    rows: list[tuple[int, dict, int, int]],
) -> Path:
    """
    rows: (window_len, stats dict, n_train, n_val)
    """
    path = results_dir / "multisector_allocator_compare_w126_w252.txt"
    lines = [
        "Global Multi-Sector Allocator — context window comparison (same data prep, same val_frac)",
        "=" * 72,
        "",
        "Note: validation metrics are on the held-out time split using the trained model weights.",
        "",
    ]
    for wlen, stats, n_train, n_val in rows:
        lines.append(f"Context window: {wlen} trading days")
        lines.append(f"  Train windows: {n_train}  |  Val windows: {n_val}")
        lines.append(f"  Mean daily return:     {stats['mean_return_daily']:.4%}")
        lines.append(f"  Volatility (daily):    {stats['volatility_daily']:.4%}")
        lines.append(f"  Sharpe (annualized):   {stats['sharpe_annualized']:.2f}")
        lines.append(f"  Total return (period): {stats['total_return']:.2%}")
        lines.append(f"  Max drawdown:          {stats['max_drawdown']:.2%}")
        lines.append(f"  Avg turnover:          {stats['avg_turnover']:.4f}")
        lines.append("")
    if len(rows) == 2:
        a, b = rows[0], rows[1]
        s0, s1 = a[1], b[1]
        lines.append("Deltas (second row minus first row, by metric order above):")
        lines.append(
            f"  Sharpe: {s1['sharpe_annualized'] - s0['sharpe_annualized']:+.2f}  "
            f"MaxDD: {s1['max_drawdown'] - s0['max_drawdown']:+.2%}  "
            f"Total ret: {s1['total_return'] - s0['total_return']:+.2%}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def run_multisector_for_window(
    *,
    df,
    sector_labels: list,
    sector_ret_cols: list,
    feature_cols: list[str],
    context_window_len: int,
) -> dict:
    """Train and export for one context length; returns stats and paths."""
    return_cols = ["market_return"] + list(sector_ret_cols)
    prefix = _output_prefix(context_window_len)

    cfg = DEFAULTS.copy()
    cfg.update(load_best_config())
    cfg["window_len"] = context_window_len
    cfg["val_frac"] = VAL_FRAC

    print(f"\n=== Context window = {context_window_len} ({prefix}) ===")
    print(f"Using {len(sector_labels)} sector sleeves")
    print(f"Assets in softmax: {1 + len(sector_labels)} (market + sectors)")
    print(f"Config: {cfg}")

    X, R, dates = build_rolling_windows_multi_asset(
        df=df,
        window_len=cfg["window_len"],
        feature_cols=feature_cols,
        return_cols=return_cols,
    )
    X_train, R_train, d_train, X_val, R_val, d_val = train_val_split(
        X, R, dates, val_frac=cfg["val_frac"]
    )
    print(f"Train windows: {X_train.shape[0]}, Val windows: {X_val.shape[0]}")

    data = {
        "X_train": X_train,
        "R_train": R_train,
        "dates_train": d_train,
        "X_val": X_val,
        "R_val": R_val,
        "dates_val": d_val,
        "n_assets": R.shape[1],
        "window_len": cfg["window_len"],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[MULTI] Starting global multi-sector training...", flush=True)
    model, history = run_training(
        data,
        device=device,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        lr_decay=cfg.get("lr_decay", 0.1),
        batch_size=cfg["batch_size"],
        patience=cfg["patience"],
        lambda_vol=cfg["lambda_vol"],
        lambda_cvar=cfg["lambda_cvar"],
        lambda_turnover=cfg.get("lambda_turnover", 0.01),
        lambda_path=cfg.get("lambda_path", 0.01),
        lambda_diversify=cfg.get("lambda_diversify", 0.0),
        min_weight=cfg.get("min_weight", 0.1),
        lambda_vol_excess=cfg.get("lambda_vol_excess", 0.0),
        target_vol_annual=cfg.get("target_vol_annual", 0.25),
        lambda_vs_ew=cfg.get("lambda_vs_ew", 0.0),
        hidden_size=cfg["hidden_size"],
        model_type=cfg.get("model_type", "gru"),
    )
    print(f"[MULTI] Trained for {len(history)} epochs")

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    p_loss, p_semilog = _plot_losses(history, fig_dir, prefix, context_window_len)
    print(f"Saved loss plots to {p_loss} and {p_semilog}")

    hist_df = pd.DataFrame(history)
    hist_path = ROOT / "results" / f"{prefix}_training_history.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_df.to_csv(hist_path, index=False)

    weights = predict_weights(model, X_val, device=device)
    asset_names = ["market"] + [f"sector_{s}" for s in sector_labels]
    out = export_multi_sector_outputs(
        dates=d_val,
        weights=weights,
        returns=R_val,
        asset_names=asset_names,
        out_dir=ROOT / "results",
        prefix=prefix,
    )

    print(f"Saved history: {hist_path}")
    print(f"Saved weights: {out['weights_path']}")
    print(f"Saved summary: {out['summary_path']}")
    p_daily = _plot_daily_portfolio_return_validation(
        d_val, weights, R_val, fig_dir, prefix, context_window_len
    )
    print(f"Saved daily portfolio return plot: {p_daily}")
    s = out["stats"]
    print(
        f"[MULTI] Metrics  Sharpe={s['sharpe_annualized']:.2f}  "
        f"MaxDD={s['max_drawdown']:.2%}  AvgTurnover={s['avg_turnover']:.4f}"
    )
    return {
        "window_len": context_window_len,
        "prefix": prefix,
        "stats": s,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "history_path": hist_path,
        "summary_path": Path(out["summary_path"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train global multi-sector GRU allocator.")
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=f"Context window length in trading days (default: {DEFAULT_CONTEXT_WINDOW}).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both 126 and 252 day windows on the same prepared data and write comparison file.",
    )
    args = parser.parse_args(argv)

    print("Connecting to WRDS...")
    conn = get_connection()
    print("Connected.")

    try:
        data_prep = prepare_multisector_data(conn, start=START_DATE, end=END_DATE)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not data_prep.get("sector_portfolios"):
        raise RuntimeError("Expected sector_portfolios=True but got False from data prep.")

    df = data_prep["df"]
    sector_labels = data_prep["sector_labels"]
    sector_ret_cols = data_prep["sector_ret_cols"]
    feature_cols = data_prep["feature_cols"]

    windows = [126, 252] if args.compare else [args.window]

    compare_rows: list[tuple[int, dict, int, int]] = []
    for wlen in windows:
        r = run_multisector_for_window(
            df=df,
            sector_labels=sector_labels,
            sector_ret_cols=sector_ret_cols,
            feature_cols=feature_cols,
            context_window_len=wlen,
        )
        compare_rows.append((wlen, r["stats"], r["n_train"], r["n_val"]))

    if args.compare and len(compare_rows) == 2:
        cmp_path = _write_comparison(ROOT / "results", compare_rows)
        print(f"\nWrote comparison: {cmp_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
