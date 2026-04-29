#!/usr/bin/env python3
"""
Train a GLOBAL allocator over multiple sector IPO portfolios + market.

This setup is intentionally separate from run_ipo_optimizer_wrds.py.

Model output each day:
  softmax([market, sector_1, ..., sector_G])  -> shape (G + 1,)

So unlike per-sector heads, this is one allocation simplex across all sleeves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_ipo_optimizer_wrds import (  # noqa: E402
    DEFAULTS,
    END_DATE,
    SECTOR_PORTFOLIOS,
    START_DATE,
    close_wrds_connection,
    get_connection,
    load_best_config,
    prepare_data,
)
from src.data_layer import train_val_split  # noqa: E402
from src.export import predict_weights  # noqa: E402
from src.multi_sector_setup import (  # noqa: E402
    build_rolling_windows_multi_asset,
    export_multi_sector_outputs,
)
from src.train import run_training  # noqa: E402


VAL_FRAC = 0.20
CONTEXT_WINDOW_LEN = 126
OUTPUT_PREFIX = "multisector_allocator"
FIG_LOSS = "multisector_allocator_loss.png"
FIG_LOSS_SEMILOG = "multisector_allocator_loss_semilog.png"
FIG_VAL_TIME = "multisector_allocator_validation_loss_over_time.png"


def _plot_losses(history: list[dict], fig_dir: Path) -> None:
    epochs_x = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_x, train_loss, label="Train loss", marker="o", markersize=3)
    ax.plot(epochs_x, val_loss, label="Validation loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Global Multi-Sector Allocator: Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / FIG_LOSS, dpi=150)
    plt.close(fig)

    t_plot = [max(abs(x), 1e-8) for x in train_loss]
    v_plot = [max(abs(x), 1e-8) for x in val_loss]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs_x, t_plot, label="Train loss", marker="o", markersize=3)
    ax.semilogy(epochs_x, v_plot, label="Validation loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (abs, semilog)")
    ax.set_title("Global Multi-Sector Allocator: Loss (Semilog)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / FIG_LOSS_SEMILOG, dpi=150)
    plt.close(fig)


def _plot_validation_loss_over_time(
    dates_val: np.ndarray,
    weights_val: np.ndarray,
    returns_val: np.ndarray,
    fig_dir: Path,
) -> None:
    """
    Plot per-day validation loss proxy over validation period.

    For comparability and readability, use negative realized portfolio return:
      loss_t = -sum_i w_{t,i} * r_{t,i}
    """
    dates_idx = pd.DatetimeIndex(dates_val)
    port_ret = np.sum(weights_val * returns_val, axis=1)
    loss_t = -port_ret

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(dates_idx, loss_t, label="Daily validation loss (-portfolio return)", linewidth=1.2)
    if len(loss_t) >= 10:
        smooth = pd.Series(loss_t, index=dates_idx).rolling(21, min_periods=5).mean()
        ax.plot(dates_idx, smooth.values, label="21-day moving average", linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title("Global Multi-Sector Allocator: Validation Loss Over Time")
    ax.set_xlabel("Validation date")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / FIG_VAL_TIME, dpi=150)
    plt.close(fig)


def main() -> int:
    print("Connecting to WRDS...")
    conn = get_connection()
    print("Connected.")

    try:
        # Reuse existing sector portfolio construction, but train one global allocator.
        data_prep = prepare_data(conn, start=START_DATE, end=END_DATE, sector_portfolios=SECTOR_PORTFOLIOS)
    finally:
        close_wrds_connection(conn)

    if not data_prep.get("sector_portfolios"):
        raise RuntimeError("Expected sector_portfolios=True but got False from data prep.")

    df = data_prep["df"]
    sector_labels = data_prep["sector_labels"]
    sector_ret_cols = data_prep["sector_ret_cols"]
    feature_cols = data_prep["feature_cols"]
    return_cols = ["market_return"] + list(sector_ret_cols)

    cfg = DEFAULTS.copy()
    cfg.update(load_best_config())
    cfg["window_len"] = CONTEXT_WINDOW_LEN
    cfg["val_frac"] = VAL_FRAC

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
        batch_size=cfg["batch_size"],
        patience=cfg["patience"],
        lambda_vol=cfg["lambda_vol"],
        lambda_cvar=cfg["lambda_cvar"],
        lambda_diversify=cfg.get("lambda_diversify", 0.0),
        min_weight=cfg.get("min_weight", 0.1),
        lambda_vol_excess=cfg.get("lambda_vol_excess", 0.0),
        target_vol_annual=cfg.get("target_vol_annual", 0.25),
        hidden_size=cfg["hidden_size"],
        model_type="gru",
        verbose=True,
        log_every=1,
    )
    print(f"[MULTI] Trained for {len(history)} epochs")

    fig_dir = ROOT / "figures" / "old diagrams"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_losses(history, fig_dir)
    print(f"Saved loss plots to {fig_dir / FIG_LOSS} and {fig_dir / FIG_LOSS_SEMILOG}")

    # Save history separately to avoid clobbering existing files.
    hist_df = pd.DataFrame(history)
    hist_path = ROOT / "results" / f"{OUTPUT_PREFIX}_training_history.csv"
    hist_path.parent.mkdir(exist_ok=True)
    hist_df.to_csv(hist_path, index=False)

    weights = predict_weights(model, X_val, device=device)
    asset_names = ["market"] + [f"sector_{s}" for s in sector_labels]
    out = export_multi_sector_outputs(
        dates=d_val,
        weights=weights,
        returns=R_val,
        asset_names=asset_names,
        out_dir=ROOT / "results",
        prefix=OUTPUT_PREFIX,
    )

    print(f"Saved history: {hist_path}")
    print(f"Saved weights: {out['weights_path']}")
    print(f"Saved summary: {out['summary_path']}")
    _plot_validation_loss_over_time(d_val, weights, R_val, fig_dir)
    print(f"Saved validation-time loss plot: {fig_dir / FIG_VAL_TIME}")
    s = out["stats"]
    print(
        f"[MULTI] Metrics  Sharpe={s['sharpe_annualized']:.2f}  "
        f"MaxDD={s['max_drawdown']:.2%}  AvgTurnover={s['avg_turnover']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

