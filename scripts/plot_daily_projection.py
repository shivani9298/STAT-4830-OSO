#!/usr/bin/env python3
"""
Daily projection plots for the trained allocator (GRU / LSTM).

**What “daily projection” means here**

The network outputs weights using only *past* data up to day *t*; the portfolio return we plot
for day *t* is w_t · r_t using *that day’s* realized benchmark returns. Wealth is built by
compounding those **one-day** returns in order:

    W_T = Π_t (1 + r_portfolio_t) − 1

There is **no** separate model that forecasts returns many days ahead—this figure is the
day-by-day implied wealth path from the saved weights and realized daily returns.

Reads default artifacts from ``results/`` and writes PNGs under ``figures/daily_projection/``:

* ``daily_projection_gru.png`` / ``daily_projection_lstm.png`` — cumulative return + wealth index
* ``train_val_loss_gru.png`` / ``train_val_loss_lstm.png`` — train vs validation loss
* ``train_val_loss_gru_lstm.png`` — both models side by side
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_aligned(
    weights_path: Path,
    returns_path: Path,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    w = pd.read_csv(weights_path, parse_dates=["date"]).set_index("date").sort_index()
    r = pd.read_csv(returns_path, index_col=0, parse_dates=True).sort_index()
    if "SPY_Only" not in r.columns or "IPO_Only" not in r.columns:
        raise ValueError(f"Expected SPY_Only, IPO_Only in {returns_path}")
    idx = w.index.intersection(r.index)
    w = w.loc[idx]
    spy = r.loc[idx, "SPY_Only"].astype(float)
    ipo = r.loc[idx, "IPO_Only"].astype(float)
    pr = w["weight_market"] * spy + w["weight_IPO"] * ipo
    eq = 0.5 * spy + 0.5 * ipo
    return pr, eq, spy, ipo, pd.Series(idx, index=idx)


def plot_daily_projection(
    out_path: Path,
    title_tag: str,
    pr: pd.Series,
    eq: pd.Series,
    spy: pd.Series,
    ipo: pd.Series,
) -> None:
    """Two panels: cumulative return from daily chaining; wealth index (starts at 1)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    def cum(x: pd.Series) -> pd.Series:
        return (1.0 + x).cumprod() - 1.0

    def wealth(x: pd.Series) -> pd.Series:
        return (1.0 + x).cumprod()

    t = pr.index
    ax1.plot(t, cum(pr) * 100, label="Model (daily chain)", color="#1f77b4", lw=2)
    ax1.plot(t, cum(eq) * 100, label="50/50", color="gray", ls="--", lw=1.4)
    ax1.plot(t, cum(spy) * 100, label="Market (SPY/DIA)", color="#2ca02c", lw=1.2, alpha=0.85)
    ax1.plot(t, cum(ipo) * 100, label="IPO index only", color="#ff7f0e", lw=1.2, alpha=0.85)
    ax1.axhline(0, color="k", lw=0.5, alpha=0.35)
    ax1.set_ylabel("Cumulative return (%)")
    ax1.set_title(
        f"{title_tag} — cumulative path from one-day returns (no multi-day return forecast)"
    )
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, wealth(pr), label="Model wealth index", color="#1f77b4", lw=2)
    ax2.plot(t, wealth(eq), label="50/50 wealth index", color="gray", ls="--", lw=1.4)
    ax2.axhline(1.0, color="k", lw=0.5, alpha=0.35)
    ax2.set_ylabel("Wealth index ($1 at start)")
    ax2.set_xlabel("Date")
    ax2.set_title("Same path as $1 compounded daily (projection horizon = 1 trading day)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    fig.suptitle(
        "Daily projection: portfolio built from sequential one-day returns",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_train_val_loss_single(
    out_path: Path,
    title_tag: str,
    hist: pd.DataFrame,
) -> None:
    """Train and validation loss vs epoch (log scale of absolute value; losses can be negative)."""
    need = {"epoch", "train_loss", "val_loss"}
    if not need.issubset(hist.columns):
        raise ValueError(f"Expected columns {need} in training history")
    fig, ax = plt.subplots(figsize=(10, 5))
    ep = hist["epoch"]
    tr = hist["train_loss"].astype(float).abs().clip(lower=1e-12)
    va = hist["val_loss"].astype(float).abs().clip(lower=1e-12)
    ax.semilogy(ep, tr, color="#1f77b4", lw=2, label="Training loss")
    ax.semilogy(ep, va, color="#ff7f0e", lw=2, ls="--", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|Loss| (log scale)")
    ax.set_title(f"{title_tag} — training vs validation loss")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_train_val_loss_combined(
    out_path: Path,
    gru_hist: pd.DataFrame | None,
    lstm_hist: pd.DataFrame | None,
) -> None:
    """One figure, two subplots: GRU and LSTM."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (tag, hist) in zip(
        axes.flat,
        [("GRU", gru_hist), ("LSTM", lstm_hist)],
    ):
        if hist is None or len(hist) == 0:
            ax.set_visible(False)
            continue
        ep = hist["epoch"]
        tr = hist["train_loss"].astype(float).abs().clip(lower=1e-12)
        va = hist["val_loss"].astype(float).abs().clip(lower=1e-12)
        ax.semilogy(ep, tr, color="#1f77b4", lw=2, label="Train")
        ax.semilogy(ep, va, color="#ff7f0e", lw=2, ls="--", label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_title(tag)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("|Loss| (log scale)")
    fig.suptitle("Training vs validation loss (GRU & LSTM)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--returns",
        type=Path,
        default=ROOT / "results" / "ipo_180day_mcap_returns.csv",
        help="CSV with SPY_Only, IPO_Only",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "figures" / "daily_projection",
        help="Output directory for PNGs",
    )
    p.add_argument(
        "--gru-weights",
        type=Path,
        default=ROOT / "results" / "ipo_optimizer_weights.csv",
    )
    p.add_argument(
        "--lstm-weights",
        type=Path,
        default=ROOT / "results" / "ipo_optimizer_weights_lstm.csv",
    )
    p.add_argument(
        "--gru-history",
        type=Path,
        default=ROOT / "results" / "training_history.csv",
        help="CSV with epoch, train_loss, val_loss (GRU)",
    )
    p.add_argument(
        "--lstm-history",
        type=Path,
        default=ROOT / "results" / "training_history_lstm.csv",
        help="CSV with epoch, train_loss, val_loss (LSTM)",
    )
    p.add_argument(
        "--skip-loss",
        action="store_true",
        help="Only plot daily projection, not training curves",
    )
    p.add_argument(
        "--skip-projection",
        action="store_true",
        help="Only plot training curves, not daily projection",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    n_ok = 0

    if not args.skip_projection:
        retp = args.returns
        if not retp.is_file():
            print(f"error: returns file not found: {retp}", file=sys.stderr)
            return 1
        for label, wpath, fname in [
            ("GRU", args.gru_weights, "daily_projection_gru.png"),
            ("LSTM", args.lstm_weights, "daily_projection_lstm.png"),
        ]:
            if not wpath.is_file():
                print(f"warning: skip {label} projection: missing {wpath}", file=sys.stderr)
                continue
            pr, eq, spy, ipo, _ = load_aligned(wpath, retp)
            if len(pr) < 5:
                print(
                    f"warning: skip {label} projection: too few aligned rows ({len(pr)})",
                    file=sys.stderr,
                )
                continue
            outp = out_dir / fname
            plot_daily_projection(outp, label, pr, eq, spy, ipo)
            print(f"Wrote {outp}")
            n_ok += 1

    if not args.skip_loss:
        gru_df: pd.DataFrame | None = None
        lstm_df: pd.DataFrame | None = None
        if args.gru_history.is_file():
            gru_df = pd.read_csv(args.gru_history)
            plot_train_val_loss_single(
                out_dir / "train_val_loss_gru.png",
                "GRU",
                gru_df,
            )
            print(f"Wrote {out_dir / 'train_val_loss_gru.png'}")
            n_ok += 1
        else:
            print(f"warning: missing GRU history: {args.gru_history}", file=sys.stderr)

        if args.lstm_history.is_file():
            lstm_df = pd.read_csv(args.lstm_history)
            plot_train_val_loss_single(
                out_dir / "train_val_loss_lstm.png",
                "LSTM",
                lstm_df,
            )
            print(f"Wrote {out_dir / 'train_val_loss_lstm.png'}")
            n_ok += 1
        else:
            print(f"warning: missing LSTM history: {args.lstm_history}", file=sys.stderr)

        if gru_df is not None or lstm_df is not None:
            plot_train_val_loss_combined(
                out_dir / "train_val_loss_gru_lstm.png",
                gru_df,
                lstm_df,
            )
            print(f"Wrote {out_dir / 'train_val_loss_gru_lstm.png'}")
            n_ok += 1

    if n_ok == 0:
        print("error: no figures written", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
