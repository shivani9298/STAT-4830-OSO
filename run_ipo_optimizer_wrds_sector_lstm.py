#!/usr/bin/env python3
"""
Train sector-aware LSTM on the current sector-enabled WRDS setup.

This mirrors run_ipo_optimizer_wrds.py but switches sector-head model_type to LSTM
and writes outputs to LSTM-specific files/directories to avoid overwriting GRU artifacts.
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
    CONTEXT_WINDOW_LEN,
    SECTOR_PORTFOLIOS,
    close_wrds_connection,
    get_connection,
    load_best_config,
    prepare_data,
)
from src.data_layer import build_rolling_windows_sector_heads, train_val_split
from src.export import export_sector_group_outputs, predict_sector_head_weights, portfolio_stats
from src.policy_layer import ipo_tilt_to_position_scale, policy_rule
from src.train import run_training_sector_heads


def main() -> int:
    print("Connecting to WRDS...")
    conn = get_connection()
    print("Connected.")
    print("[IPO] Using sector-aware LSTM setup...", flush=True)

    try:
        data_prep = prepare_data(conn, sector_portfolios=SECTOR_PORTFOLIOS)
    finally:
        close_wrds_connection(conn)

    if not data_prep.get("sector_portfolios"):
        raise RuntimeError("Sector portfolios are required for this runner.")

    df = data_prep["df"]
    feature_cols = data_prep["feature_cols"]
    sector_labels = data_prep["sector_labels"]
    sector_ret_cols = data_prep["sector_ret_cols"]

    cfg = load_best_config()
    cfg["window_len"] = CONTEXT_WINDOW_LEN
    print(f"Hyperparameters: {cfg}")

    X, R, dates = build_rolling_windows_sector_heads(
        df,
        window_len=cfg["window_len"],
        feature_cols=feature_cols,
        sector_ret_cols=sector_ret_cols,
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
        "n_sectors": len(sector_labels),
        "window_len": cfg["window_len"],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history = run_training_sector_heads(
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
        lambda_vol_excess=cfg.get("lambda_vol_excess", 1.0),
        target_vol_annual=cfg.get("target_vol_annual", 0.25),
        hidden_size=cfg["hidden_size"],
        model_type="lstm",
        verbose=True,
        log_every=1,
    )
    print(f"Trained for {len(history)} epochs")

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv(ROOT / "results" / "training_history_sector_lstm.csv", index=False)

    epochs_x = [h["epoch"] for h in history]
    train_loss = np.array([h["train_loss"] for h in history])
    val_loss = np.array([h["val_loss"] for h in history])
    t_plot = np.clip(np.abs(train_loss), 1e-8, None)
    v_plot = np.clip(np.abs(val_loss), 1e-8, None)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs_x, t_plot, label="Train loss", marker="o", markersize=3)
    ax.semilogy(epochs_x, v_plot, label="Validation loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Sector LSTM: Training and Validation Loss ({len(sector_labels)} sector heads)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "ipo_optimizer_loss_sector_lstm.png", dpi=150)
    fig.savefig(fig_dir / "ipo_optimizer_loss_sector_semilog_lstm.png", dpi=150)
    plt.close(fig)
    print(f"Saved LSTM sector loss plot to {fig_dir / 'ipo_optimizer_loss_sector_semilog_lstm.png'}")

    weights = predict_sector_head_weights(model, X_val, device)
    out_dir = ROOT / "results" / "lstm_sector"
    export_sector_group_outputs(d_val, weights, R_val, sector_labels, out_dir)
    print(f"Exported sector outputs to {out_dir}")

    avg_ipo = float(np.mean(weights[:, :, 1])) if weights.ndim == 3 else 0.0
    scale = ipo_tilt_to_position_scale(avg_ipo)
    print(policy_rule(avg_ipo))
    print(f"Suggested position scale (avg across sector IPO sleeves): {scale:.2f}")
    for idx in range(weights.shape[1]):
        st = portfolio_stats(weights[:, idx, :], R_val[:, idx, :])
        print(f"  [{sector_labels[idx]}] Sharpe={st['sharpe_annualized']:.2f}  MaxDD={st['max_drawdown']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

