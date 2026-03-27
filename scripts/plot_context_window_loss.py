#!/usr/bin/env python3
"""
Train GRU allocators at each context window length and produce a detailed
loss-function plot: one subplot per window showing train vs val loss with
smoothed curves, plus a combined overlay and a component breakdown panel.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.data_layer import add_optional_features, build_rolling_windows, train_val_split
from src.train import run_training
from src.export import predict_weights, portfolio_stats

WINDOW_LENGTHS = [42, 63, 84, 126, 252]
SMOOTH = 10

TRAIN_CFG = {
    "epochs": 200,
    "lr": 3e-4,
    "lr_decay": 0.1,
    "batch_size": 256,
    "patience": 200,
    "lambda_vol": 0.5,
    "lambda_cvar": 0.5,
    "lambda_turnover": 0.0,
    "lambda_path": 0.0001,
    "lambda_vol_excess": 1.0,
    "target_vol_annual": 0.25,
    "hidden_size": 64,
    "lambda_diversify": 0.0,
    "min_weight": 0.1,
}
VAL_FRAC = 0.2


def load_returns() -> pd.DataFrame:
    mcap = pd.read_csv(
        ROOT / "results" / "ipo_180day_mcap_returns.csv",
        index_col=0, parse_dates=True,
    )
    df = pd.DataFrame({
        "market_return": mcap["SPY_Only"].values,
        "ipo_return": mcap["IPO_Only"].values,
    }, index=mcap.index)
    df["market_return"] = df["market_return"].clip(-0.10, 0.10)
    df["ipo_return"] = df["ipo_return"].clip(-0.50, 0.50)
    return df.dropna()


def smooth(arr, w=SMOOTH):
    k = min(w, len(arr))
    return np.convolve(arr, np.ones(k) / k, mode="same")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_returns()
    print(f"Loaded {len(df)} days  ({df.index[0].date()} → {df.index[-1].date()})")

    all_histories = {}
    all_stats = {}

    for wl in WINDOW_LENGTHS:
        print(f"\nTraining window_len={wl} …")
        torch.manual_seed(42)
        np.random.seed(42)

        df_feat = add_optional_features(df.copy())
        feature_cols = list(df_feat.columns)
        X, R, dates = build_rolling_windows(df_feat, window_len=wl, feature_cols=feature_cols)
        X_tr, R_tr, d_tr, X_va, R_va, d_va = train_val_split(X, R, dates, val_frac=VAL_FRAC)

        data = {
            "X_train": X_tr, "R_train": R_tr, "dates_train": d_tr,
            "X_val": X_va, "R_val": R_va, "dates_val": d_va,
            "feature_cols": feature_cols, "df": df_feat,
            "n_assets": 2, "window_len": wl,
        }
        model, history = run_training(
            data, device=device, model_type="gru",
            epochs=TRAIN_CFG["epochs"], lr=TRAIN_CFG["lr"],
            lr_decay=TRAIN_CFG["lr_decay"], batch_size=TRAIN_CFG["batch_size"],
            patience=TRAIN_CFG["patience"], lambda_vol=TRAIN_CFG["lambda_vol"],
            lambda_cvar=TRAIN_CFG["lambda_cvar"],
            lambda_turnover=TRAIN_CFG["lambda_turnover"],
            lambda_path=TRAIN_CFG["lambda_path"],
            lambda_vol_excess=TRAIN_CFG["lambda_vol_excess"],
            target_vol_annual=TRAIN_CFG["target_vol_annual"],
            hidden_size=TRAIN_CFG["hidden_size"],
            lambda_diversify=TRAIN_CFG["lambda_diversify"],
            min_weight=TRAIN_CFG["min_weight"],
        )
        weights = predict_weights(model, X_va, device)
        stats = portfolio_stats(weights, R_va)
        all_histories[wl] = history
        all_stats[wl] = stats
        print(f"  → {len(history)} epochs, Sharpe={stats['sharpe_annualized']:.2f}")

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    colors = {42: "#1f77b4", 63: "#ff7f0e", 84: "#2ca02c", 126: "#d62728", 252: "#9467bd"}

    # ── FIGURE 1: per-window loss subplots ────────────────────────────────────
    n = len(WINDOW_LENGTHS)
    fig1, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, wl in zip(axes, WINDOW_LENGTHS):
        h = all_histories[wl]
        ep = np.array([e["epoch"] for e in h])
        tl = np.array([e["train_loss"] for e in h])
        vl = np.array([e["val_loss"] for e in h])

        ax.semilogy(ep, np.clip(np.abs(tl), 1e-8, None), alpha=0.2, color=colors[wl], lw=0.8)
        ax.semilogy(ep, np.clip(np.abs(smooth(tl)), 1e-8, None),
                    color=colors[wl], lw=2, label="Train (smoothed)")
        ax.semilogy(ep, np.clip(np.abs(vl), 1e-8, None),
                    color="black", lw=1.5, marker=".", markersize=2, label="Validation")
        ax.axvline(x=1, color="red", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_title(f"{wl}d  ({wl/252:.1f} yr)\nSharpe {all_stats[wl]['sharpe_annualized']:.2f}",
                     fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("|Loss|  (log scale)")
    fig1.suptitle("Loss Convergence per Context Window", fontsize=13, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig(fig_dir / "context_window_loss_per_window.png", dpi=150)
    plt.close(fig1)
    print(f"\nSaved: figures/context_window_loss_per_window.png")

    # ── FIGURE 2: overlaid train & val loss ──────────────────────────────────
    fig2, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(14, 5))

    for wl in WINDOW_LENGTHS:
        h = all_histories[wl]
        ep = np.array([e["epoch"] for e in h])
        tl = np.abs(np.array([e["train_loss"] for e in h]))
        vl = np.abs(np.array([e["val_loss"] for e in h]))
        c = colors[wl]
        ax_t.semilogy(ep, np.clip(smooth(tl), 1e-8, None), lw=2, color=c,
                      label=f"{wl}d (Sharpe {all_stats[wl]['sharpe_annualized']:.2f})")
        ax_v.semilogy(ep, np.clip(vl, 1e-8, None), lw=2, color=c,
                      label=f"{wl}d")

    ax_t.set_xlabel("Epoch"); ax_t.set_ylabel("|Train Loss| (log)")
    ax_t.set_title("Smoothed Training Loss"); ax_t.legend(fontsize=8); ax_t.grid(True, alpha=0.3)
    ax_v.set_xlabel("Epoch"); ax_v.set_ylabel("|Val Loss| (log)")
    ax_v.set_title("Validation Loss"); ax_v.legend(fontsize=8); ax_v.grid(True, alpha=0.3)
    fig2.suptitle("Loss Curves Overlaid — All Context Windows", fontsize=13, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(fig_dir / "context_window_loss_overlay.png", dpi=150)
    plt.close(fig2)
    print(f"Saved: figures/context_window_loss_overlay.png")

    # ── FIGURE 3: component breakdown (val) across windows ───────────────────
    comp_keys = [
        ("val_mean_return", "Mean Return"),
        ("val_cvar", "CVaR"),
        ("val_volatility", "Variance"),
        ("val_vol_excess", "Vol-Excess"),
        ("val_turnover", "Turnover"),
        ("val_weight_path", "Weight Path"),
    ]

    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 8))
    axes3 = axes3.ravel()

    for idx, (key, label) in enumerate(comp_keys):
        ax = axes3[idx]
        for wl in WINDOW_LENGTHS:
            h = all_histories[wl]
            ep = [e["epoch"] for e in h]
            vals = [e.get(key, 0.0) for e in h]
            ax.plot(ep, vals, lw=1.5, color=colors[wl], label=f"{wl}d", alpha=0.85)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig3.suptitle("Validation Loss Components by Context Window", fontsize=13, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig(fig_dir / "context_window_loss_components.png", dpi=150)
    plt.close(fig3)
    print(f"Saved: figures/context_window_loss_components.png")

    # ── FIGURE 4: final-epoch component bar chart ────────────────────────────
    bar_keys = ["val_mean_return", "val_volatility", "val_cvar", "val_vol_excess"]
    bar_labels = ["Mean Return\n(reward)", "Variance\n(penalty)", "CVaR\n(tail risk)", "Vol Excess\n(penalty)"]
    bar_colors = ["#2ca02c", "#ff7f0e", "#e377c2", "#bcbd22"]

    fig4, ax4 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(WINDOW_LENGTHS))
    width = 0.18
    offsets = np.arange(len(bar_keys)) - (len(bar_keys) - 1) / 2

    for i, (key, lab, col) in enumerate(zip(bar_keys, bar_labels, bar_colors)):
        vals = []
        for wl in WINDOW_LENGTHS:
            h = all_histories[wl]
            vals.append(h[-1].get(key, 0.0))
        ax4.bar(x + offsets[i] * width, vals, width, label=lab, color=col, alpha=0.85)

    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{wl}d\n({wl/252:.1f} yr)" for wl in WINDOW_LENGTHS])
    ax4.set_xlabel("Context Window")
    ax4.set_ylabel("Component Value (final epoch)")
    ax4.set_title("Final-Epoch Loss Components by Context Window", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")
    fig4.tight_layout()
    fig4.savefig(fig_dir / "context_window_loss_final_bars.png", dpi=150)
    plt.close(fig4)
    print(f"Saved: figures/context_window_loss_final_bars.png")

    print("\nDone — 4 figures saved to figures/")


if __name__ == "__main__":
    main()
