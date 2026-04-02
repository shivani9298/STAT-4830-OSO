#!/usr/bin/env python3
"""
Run IPO optimizer with LSTM using the same setup as the GRU runner.

This script intentionally does not overwrite the standard GRU outputs.
It writes LSTM-specific result/figure filenames.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from run_ipo_optimizer_wrds import (
    get_connection,
    prepare_data,
    load_best_config,
)
from src.data_layer import build_rolling_windows, train_val_split
from src.train import run_training
from src.export import predict_weights, portfolio_stats, export_weights_csv, export_summary
from src.policy_layer import ipo_tilt_to_position_scale, policy_rule


def main() -> int:
    print("Connecting to WRDS...")
    conn = get_connection()
    print("Connected.")

    data_prep = prepare_data(conn)
    df = data_prep["df"]
    feature_cols = data_prep["feature_cols"]

    cfg = load_best_config()
    print(f"Hyperparameters: {cfg}")

    X, R, dates = build_rolling_windows(
        df, window_len=cfg["window_len"], feature_cols=feature_cols
    )
    X_train, R_train, d_train, X_val, R_val, d_val = train_val_split(
        X, R, dates, val_frac=cfg["val_frac"]
    )

    data = {
        "X_train": X_train,
        "R_train": R_train,
        "dates_train": d_train,
        "X_val": X_val,
        "R_val": R_val,
        "dates_val": d_val,
        "feature_cols": feature_cols,
        "df": df,
        "n_assets": 2,
        "window_len": cfg["window_len"],
    }
    print(f"Train windows: {X_train.shape[0]}, Val windows: {X_val.shape[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        lambda_diversify=cfg.get("lambda_diversify", 1.0),
        min_weight=cfg.get("min_weight", 0.1),
        lambda_vol_excess=cfg.get("lambda_vol_excess", 1.0),
        target_vol_annual=cfg.get("target_vol_annual", 0.25),
        hidden_size=cfg["hidden_size"],
        model_type="lstm",
    )
    print(f"Trained for {len(history)} epochs")

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv(ROOT / "results" / "training_history_lstm.csv", index=False)

    epochs_x = [h["epoch"] for h in history]
    train_loss = np.array([h["train_loss"] for h in history])
    val_loss = np.array([h["val_loss"] for h in history])
    smooth_window = min(10, len(train_loss))
    train_smooth = np.convolve(train_loss, np.ones(smooth_window) / smooth_window, mode="same")
    t_raw = np.clip(np.abs(train_loss), 1e-8, None)
    t_smooth = np.clip(np.abs(train_smooth), 1e-8, None)
    v_plot = np.clip(np.abs(val_loss), 1e-8, None)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(epochs_x, t_raw, alpha=0.25, color="#1f77b4", linewidth=0.8)
    ax.semilogy(epochs_x, t_smooth, color="#1f77b4", linewidth=2, label="Train loss (10-ep smoothed)")
    ax.semilogy(
        epochs_x,
        v_plot,
        color="#ff7f0e",
        linewidth=2,
        marker="s",
        markersize=2,
        label="Validation loss",
    )
    ax.axvline(
        x=1,
        color="red",
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
        label=f"LR drop ×{cfg.get('lr_decay', 0.1):.1f} (epoch 1)",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|Loss|  (log scale)")
    ax.set_title(
        f"LSTM Training — {len(history)} epochs  |  "
        f"lr={cfg['lr']:.0e}→{cfg['lr'] * cfg.get('lr_decay', 0.1):.0e}  "
        f"batch={cfg['batch_size']}  "
        f"λ_path={cfg.get('lambda_path', 0):.0e}  λ_turn={cfg.get('lambda_turnover', 0):.0e}"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "ipo_optimizer_loss_lstm.png", dpi=150)
    fig.savefig(fig_dir / "ipo_optimizer_loss_semilog_lstm.png", dpi=150)
    plt.close()
    print(f"Saved loss plot to {fig_dir / 'ipo_optimizer_loss_semilog_lstm.png'}")

    weights = predict_weights(model, data["X_val"], device)
    stats = portfolio_stats(weights, data["R_val"])

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    weights_path = out_dir / "ipo_optimizer_weights_lstm.csv"
    summary_path = out_dir / "ipo_optimizer_summary_lstm.txt"

    export_weights_csv(data["dates_val"], weights, weights_path)
    export_summary(stats, weights, summary_path, R=data["R_val"])
    print(f"Exported weights to {weights_path}")
    print(f"Exported summary to {summary_path}")

    avg_ipo = float(weights[:, 1].mean()) if weights.shape[1] >= 2 else 0.0
    scale = ipo_tilt_to_position_scale(avg_ipo)
    print(policy_rule(avg_ipo))
    print(f"Suggested position scale: {scale:.2f}")
    print(f"Metrics: Sharpe={stats['sharpe_annualized']:.2f}, MaxDD={stats['max_drawdown']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

