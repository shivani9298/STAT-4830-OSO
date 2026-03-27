#!/usr/bin/env python3
"""
Compare GRU allocator performance across different context window lengths.

Uses locally cached return data (no WRDS needed). Trains one model per window
length, evaluates on the same validation period, and produces comparison plots.
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
import matplotlib.dates as mdates

from src.data_layer import add_optional_features, build_rolling_windows, train_val_split
from src.train import run_training
from src.export import predict_weights, portfolio_stats

WINDOW_LENGTHS = [42, 63, 84, 126, 252]

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
    """Load market + IPO returns from the cached mcap returns CSV."""
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
    df = df.dropna()
    return df


def run_single_window(df: pd.DataFrame, window_len: int, device: torch.device) -> dict:
    """Train and evaluate one model with a given window length."""
    df_feat = add_optional_features(df.copy())
    feature_cols = list(df_feat.columns)
    X, R, dates = build_rolling_windows(df_feat, window_len=window_len, feature_cols=feature_cols)

    if X.shape[0] < 50:
        print(f"  [SKIP] window_len={window_len}: only {X.shape[0]} samples")
        return {}

    X_train, R_train, d_train, X_val, R_val, d_val = train_val_split(
        X, R, dates, val_frac=VAL_FRAC,
    )
    print(f"  window={window_len}: train={X_train.shape[0]}, val={X_val.shape[0]}")

    data = {
        "X_train": X_train, "R_train": R_train, "dates_train": d_train,
        "X_val": X_val, "R_val": R_val, "dates_val": d_val,
        "feature_cols": feature_cols, "df": df_feat,
        "n_assets": 2, "window_len": window_len,
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

    weights = predict_weights(model, X_val, device)
    stats = portfolio_stats(weights, R_val)

    port_ret = (weights * R_val).sum(axis=1)
    avg_ipo_wt = float(weights[:, 1].mean())

    return {
        "window_len": window_len,
        "stats": stats,
        "weights": weights,
        "port_ret": port_ret,
        "dates_val": d_val,
        "R_val": R_val,
        "avg_ipo_weight": avg_ipo_wt,
        "history": history,
        "n_train": X_train.shape[0],
        "n_val": X_val.shape[0],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_returns()
    print(f"Loaded {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")

    results = {}
    for wl in WINDOW_LENGTHS:
        print(f"\n{'='*60}")
        print(f"Training with window_len = {wl} ({wl/252:.1f} years)")
        print(f"{'='*60}")
        torch.manual_seed(42)
        np.random.seed(42)
        res = run_single_window(df, wl, device)
        if res:
            results[wl] = res

    if not results:
        print("No results produced.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("CONTEXT WINDOW COMPARISON — SUMMARY")
    print(f"{'='*80}")
    header = f"{'Window':>8} {'Days':>5} {'Train':>6} {'Val':>5} {'TotRet':>8} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>7} {'MaxDD':>8} {'AvgIPO':>7}"
    print(header)
    print("-" * len(header))
    for wl in sorted(results.keys()):
        r = results[wl]
        s = r["stats"]
        print(
            f"{wl:>6}d  {wl:>5} {r['n_train']:>6} {r['n_val']:>5}"
            f" {s['total_return']:>7.2%} {s['return_annualized']:>7.2%}"
            f" {s['volatility_annualized']:>7.2%} {s['sharpe_annualized']:>6.2f}"
            f" {s['max_drawdown']:>7.2%} {r['avg_ipo_weight']:>6.1%}"
        )

    # Also print baselines for reference (use the longest-window val period for fair comparison)
    ref_wl = max(results.keys())
    ref = results[ref_wl]
    R_ref = ref["R_val"]
    for name, w_vec in [("Market only", [1.0, 0.0]), ("IPO only", [0.0, 1.0]), ("Equal 50/50", [0.5, 0.5])]:
        w_base = np.tile(w_vec, (R_ref.shape[0], 1))
        bs = portfolio_stats(w_base, R_ref)
        print(
            f"{'':>8} {'base':>5} {'':>6} {R_ref.shape[0]:>5}"
            f" {bs['total_return']:>7.2%} {bs['return_annualized']:>7.2%}"
            f" {bs['volatility_annualized']:>7.2%} {bs['sharpe_annualized']:>6.2f}"
            f" {bs['max_drawdown']:>7.2%} {'—':>7}  {name}"
        )

    # ── PLOT 1: Sharpe & Return vs. window length ────────────────────────────
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)

    wls = sorted(results.keys())
    sharpes = [results[w]["stats"]["sharpe_annualized"] for w in wls]
    tot_rets = [results[w]["stats"]["total_return"] * 100 for w in wls]
    ann_vols = [results[w]["stats"]["volatility_annualized"] * 100 for w in wls]
    max_dds = [results[w]["stats"]["max_drawdown"] * 100 for w in wls]
    avg_ipos = [results[w]["avg_ipo_weight"] * 100 for w in wls]
    n_trains = [results[w]["n_train"] for w in wls]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(wls, sharpes, "o-", color="#1f77b4", lw=2, markersize=8)
    ax.set_xlabel("Context Window (trading days)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio vs. Context Window")
    ax.grid(True, alpha=0.3)
    for w, s in zip(wls, sharpes):
        ax.annotate(f"{s:.2f}", (w, s), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, fontweight="bold")

    ax = axes[0, 1]
    ax.plot(wls, tot_rets, "s-", color="#2ca02c", lw=2, markersize=8)
    ax.set_xlabel("Context Window (trading days)")
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Total Return vs. Context Window")
    ax.grid(True, alpha=0.3)
    for w, r in zip(wls, tot_rets):
        ax.annotate(f"{r:.1f}%", (w, r), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)

    ax = axes[1, 0]
    ax.plot(wls, ann_vols, "^-", color="#ff7f0e", lw=2, markersize=8, label="Ann. Volatility")
    ax.plot(wls, [-d for d in max_dds], "v-", color="#d62728", lw=2, markersize=8, label="|Max Drawdown|")
    ax.set_xlabel("Context Window (trading days)")
    ax.set_ylabel("(%)")
    ax.set_title("Risk Metrics vs. Context Window")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    color_ipo = "#9467bd"
    color_n = "#8c564b"
    ax.bar([w - 3 for w in wls], avg_ipos, width=6, color=color_ipo, alpha=0.7, label="Avg IPO Weight (%)")
    ax.set_xlabel("Context Window (trading days)")
    ax.set_ylabel("Avg IPO Weight (%)", color=color_ipo)
    ax.tick_params(axis="y", labelcolor=color_ipo)
    ax.set_title("IPO Allocation & Training Samples")
    ax2 = ax.twinx()
    ax2.plot(wls, n_trains, "D-", color=color_n, lw=2, markersize=7, label="# Train Samples")
    ax2.set_ylabel("Training Samples", color=color_n)
    ax2.tick_params(axis="y", labelcolor=color_n)
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Context Window Length Comparison — GRU Allocator", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "context_window_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: figures/context_window_comparison.png")

    # ── PLOT 2: Cumulative returns per window length ─────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(wls)))

    for wl, color in zip(wls, colors):
        r = results[wl]
        cum = (1 + r["port_ret"]).cumprod() - 1
        ax2.plot(r["dates_val"], cum * 100, lw=2, color=color,
                 label=f"Window={wl}d ({wl/252:.1f}yr)")

    # Baselines using the shortest-window val dates (largest val set)
    shortest_wl = min(results.keys())
    ref = results[shortest_wl]
    base_ret_m = ref["R_val"][:, 0]
    base_ret_i = ref["R_val"][:, 1]
    base_ret_eq = 0.5 * base_ret_m + 0.5 * base_ret_i
    ax2.plot(ref["dates_val"], ((1 + base_ret_eq).cumprod() - 1) * 100,
             lw=1.5, ls="--", color="gray", label="Equal 50/50 (baseline)")

    ax2.axhline(0, color="k", lw=0.6, ls=":")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig2.autofmt_xdate()
    ax2.set_ylabel("Cumulative Return (%)")
    ax2.set_title("Cumulative Returns by Context Window — Validation Period")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(fig_dir / "context_window_cumulative.png", dpi=150)
    plt.close(fig2)
    print(f"Saved: figures/context_window_cumulative.png")

    # ── PLOT 3: Rolling 21-day Sharpe by window length ───────────────────────
    ROLL = 21
    fig3, ax3 = plt.subplots(figsize=(12, 5))

    for wl, color in zip(wls, colors):
        r = results[wl]
        pr = pd.Series(r["port_ret"], index=r["dates_val"])
        roll_mean = pr.rolling(ROLL).mean()
        roll_std = pr.rolling(ROLL).std()
        roll_sharpe = (roll_mean / roll_std) * np.sqrt(252)
        ax3.plot(r["dates_val"], roll_sharpe.values, lw=1.5, color=color,
                 label=f"Window={wl}d", alpha=0.85)

    ax3.axhline(0, color="k", lw=0.6, ls=":")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig3.autofmt_xdate()
    ax3.set_ylabel("Rolling Sharpe (annualized)")
    ax3.set_title(f"Rolling {ROLL}-day Sharpe Ratio by Context Window — Validation Period")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-15, 25)
    fig3.tight_layout()
    fig3.savefig(fig_dir / "context_window_rolling_sharpe.png", dpi=150)
    plt.close(fig3)
    print(f"Saved: figures/context_window_rolling_sharpe.png")

    # ── PLOT 4: Training loss convergence by window length ───────────────────
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5))

    for wl, color in zip(wls, colors):
        r = results[wl]
        h = r["history"]
        epochs = [e["epoch"] for e in h]
        train_l = [abs(e["train_loss"]) for e in h]
        val_l = [abs(e["val_loss"]) for e in h]
        ax4a.semilogy(epochs, train_l, lw=1.5, color=color, alpha=0.7, label=f"W={wl}d")
        ax4b.semilogy(epochs, val_l, lw=1.5, color=color, alpha=0.7, label=f"W={wl}d")

    ax4a.set_xlabel("Epoch")
    ax4a.set_ylabel("|Train Loss| (log)")
    ax4a.set_title("Training Loss Convergence")
    ax4a.legend(fontsize=8)
    ax4a.grid(True, alpha=0.3)

    ax4b.set_xlabel("Epoch")
    ax4b.set_ylabel("|Val Loss| (log)")
    ax4b.set_title("Validation Loss Convergence")
    ax4b.legend(fontsize=8)
    ax4b.grid(True, alpha=0.3)

    fig4.suptitle("Loss Convergence by Context Window Length", fontsize=13, fontweight="bold")
    fig4.tight_layout()
    fig4.savefig(fig_dir / "context_window_loss_convergence.png", dpi=150)
    plt.close(fig4)
    print(f"Saved: figures/context_window_loss_convergence.png")

    # ── Save results CSV ─────────────────────────────────────────────────────
    rows = []
    for wl in wls:
        r = results[wl]
        s = r["stats"]
        rows.append({
            "window_len": wl,
            "window_years": round(wl / 252, 2),
            "n_train": r["n_train"],
            "n_val": r["n_val"],
            "total_return": round(s["total_return"], 4),
            "return_annualized": round(s["return_annualized"], 4),
            "volatility_annualized": round(s["volatility_annualized"], 4),
            "sharpe_annualized": round(s["sharpe_annualized"], 4),
            "max_drawdown": round(s["max_drawdown"], 4),
            "avg_ipo_weight": round(r["avg_ipo_weight"], 4),
        })
    pd.DataFrame(rows).to_csv(ROOT / "results" / "context_window_comparison.csv", index=False)
    print(f"Saved: results/context_window_comparison.csv")


if __name__ == "__main__":
    main()
