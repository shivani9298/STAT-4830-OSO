#!/usr/bin/env python3
"""
Train GRU and LSTM IPO allocators on the same WRDS data split, save histories, plot comparison.

Usage:
  python3 scripts/compare_gru_lstm_ipo.py              # window_len from ipo_optimizer_best_config.json
  python3 scripts/compare_gru_lstm_ipo.py --window 126

Outputs (tagged with actual window, e.g. _w126):
  results/ipo_optimizer_training_history_gru_w126.csv
  results/ipo_optimizer_training_history_lstm_w126.csv
  results/ipo_optimizer_gru_lstm_metrics_w126.txt
  figures/ipo_optimizer_gru_vs_lstm_loss_w126.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from run_ipo_optimizer_wrds import load_best_config, prepare_data
from src.data_layer import build_rolling_windows, train_val_split
from src.export import predict_weights, portfolio_stats
from src.train import run_training
from src.wrds_data import get_connection


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    w = min(window, len(y))
    if w < 2:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def _semilog_abs(y: np.ndarray) -> np.ndarray:
    return np.clip(np.abs(y.astype(float)), 1e-8, None)


def main() -> int:
    parser = argparse.ArgumentParser(description="GRU vs LSTM IPO optimizer on same data.")
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Override context window length (trading days). Default: from load_best_config().",
    )
    args = parser.parse_args()

    print("Connecting to WRDS...")
    conn = get_connection()
    print("Connected.")
    data_prep = prepare_data(conn)
    try:
        conn.close()
    except Exception:
        pass

    df = data_prep["df"]
    feature_cols = data_prep["feature_cols"]
    cfg = load_best_config()
    if args.window is not None:
        cfg["window_len"] = args.window
        print(f"Override: window_len = {cfg['window_len']}")
    tag = f"_w{cfg['window_len']}"
    print(f"Hyperparameters (shared): {cfg}")

    X, R, dates = build_rolling_windows(df, window_len=cfg["window_len"], feature_cols=feature_cols)
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
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    histories: dict[str, list] = {}
    metrics_lines: list[str] = []

    for model_type in ("gru", "lstm"):
        print(f"\n=== Training {model_type.upper()} ===", flush=True)
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
            lambda_vs_ew=cfg.get("lambda_vs_ew", 0.0),
            hidden_size=cfg["hidden_size"],
            model_type=model_type,
        )
        histories[model_type] = history
        hist_path = results_dir / f"ipo_optimizer_training_history_{model_type}{tag}.csv"
        pd.DataFrame(history).to_csv(hist_path, index=False)
        print(f"Saved {hist_path} ({len(history)} epochs)")

        weights = predict_weights(model, X_val, device=device)
        stats = portfolio_stats(weights, R_val)
        metrics_lines.append(
            f"{model_type.upper()}: Sharpe={stats['sharpe_annualized']:.2f}  "
            f"MaxDD={stats['max_drawdown']:.2%}  TotalRet={stats['total_return']:.2%}  "
            f"epochs={len(history)}"
        )

    metrics_path = results_dir / f"ipo_optimizer_gru_lstm_metrics{tag}.txt"
    metrics_path.write_text(
        f"GRU vs LSTM — same data, same hyperparameters (model_type differs only)\n"
        f"window_len={cfg['window_len']}\n\n"
        + "\n".join(metrics_lines)
        + "\n"
    )
    print(f"\nWrote {metrics_path}")

    # --- Plot: two panels, semilog |loss| ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    styles = {"gru": "#1f77b4", "lstm": "#ff7f0e"}

    ax = axes[0]
    for mt in ("gru", "lstm"):
        h = histories[mt]
        epochs = np.array([x["epoch"] for x in h])
        y = np.array([x["val_loss"] for x in h])
        ax.semilogy(epochs, _semilog_abs(y), label=mt.upper(), color=styles[mt], linewidth=2, marker="s", markersize=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|Loss| (log scale)")
    ax.set_title("Validation |loss| (semilog)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for mt in ("gru", "lstm"):
        h = histories[mt]
        epochs = np.array([x["epoch"] for x in h])
        y = _smooth(np.array([x["train_loss"] for x in h]), min(10, len(h)))
        ax.semilogy(epochs, _semilog_abs(y), label=mt.upper(), color=styles[mt], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|Loss| (log scale)")
    ax.set_title("Train |loss| — 10-epoch smoothed (semilog)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"IPO optimizer: GRU vs LSTM (window={cfg['window_len']}, batch={cfg['batch_size']}, "
        f"lr={cfg['lr']:.0e})",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = figures_dir / f"ipo_optimizer_gru_vs_lstm_loss{tag}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
