#!/usr/bin/env python3
"""Plot validation (per-epoch) and holdout (daily) metrics for GRU / LSTM / Transformer.

Reads training_history*.csv, ipo_optimizer_returns_{val,test}*.csv, weights, and
selection_metrics*.json from an artifacts directory (e.g. copied VM output).

Writes comparison figures under figures/ (default: figures/model_comparison_vm/), including
test/val cumulative return overlays vs the 50/50 benchmark (linear scale).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = ("gru", "lstm", "transformer")

# Training-history columns that are >0 in normal runs (safe for semilogy after clip).
SEMILOG_VAL_COLS = [
    "val_volatility",
    "val_turnover",
    "val_cvar",
    "val_vol_excess",
    "val_weight_path",
]
# Signed or mixed — linear y.
LINEAR_VAL_COLS = [
    "val_mean_return",
    "val_excess_vs_ew",
    "val_sel_tail_q_excess",
    "val_sel_mean_excess",
    "val_sel_max_drawdown",
]


def result_suffix(model: str) -> str:
    return "" if model == "gru" else f"_{model}"


def history_csv(results_dir: Path, model: str) -> Path:
    if model == "gru":
        return results_dir / "training_history.csv"
    return results_dir / f"training_history_{model}.csv"


def load_histories(results_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for m in MODELS:
        p = history_csv(results_dir, m)
        if not p.is_file():
            print(f"warning: missing {p}", file=sys.stderr)
            continue
        df = pd.read_csv(p)
        df["model"] = m
        out[m] = df
    return out


def _semilogy_safe(ax, x, y, label: str, **kwargs) -> None:
    y = np.asarray(y, dtype=float)
    y_plot = np.clip(np.abs(y), 1e-16, None)
    ax.semilogy(x, y_plot, label=label, **kwargs)


def plot_epoch_semilog_losses(histories: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [("train_loss", "Train loss"), ("val_loss", "Val loss"), ("selection_objective", "Selection objective")]
    for ax, (col, title) in zip(axes, metrics):
        for m, df in histories.items():
            if col not in df.columns:
                continue
            _semilogy_safe(ax, df["epoch"], df[col], m.upper())
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Training / validation loss and selection objective (log scale)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_epoch_semilog_lr(histories: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for m, df in histories.items():
        if "lr" in df.columns:
            _semilogy_safe(ax, df["epoch"], df["lr"], m.upper())
    ax.set_title("Learning rate")
    ax.set_xlabel("Epoch")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_epoch_semilog_val_components(histories: dict[str, pd.DataFrame], out_path: Path) -> None:
    cols = [c for c in SEMILOG_VAL_COLS if any(c in df.columns for df in histories.values())]
    if not cols:
        return
    n = len(cols)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        ax = axes[i]
        for m, df in histories.items():
            if col not in df.columns:
                continue
            _semilogy_safe(ax, df["epoch"], df[col], m.upper())
        ax.set_title(col)
        ax.set_xlabel("Epoch")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Per-epoch validation components (positive branch, log scale)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_epoch_linear_val_excess(histories: dict[str, pd.DataFrame], out_path: Path) -> None:
    cols = [c for c in LINEAR_VAL_COLS if any(c in df.columns for df in histories.values())]
    if not cols:
        return
    n = len(cols)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        ax = axes[i]
        for m, df in histories.items():
            if col not in df.columns:
                continue
            ax.plot(df["epoch"], df[col], label=m.upper())
        ax.set_title(col)
        ax.set_xlabel("Epoch")
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Per-epoch validation excess and selection metrics (linear scale)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def sharpe_from_ret(port_ret: np.ndarray) -> float:
    mu = float(np.mean(port_ret)) if len(port_ret) else 0.0
    vol = float(np.std(port_ret)) if len(port_ret) else 0.0
    return (mu / vol * np.sqrt(252.0)) if vol > 1e-12 else 0.0


def max_drawdown_from_ret(port_ret: np.ndarray) -> float:
    if len(port_ret) == 0:
        return 0.0
    wealth = np.cumprod(1.0 + port_ret)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / np.clip(peak, 1e-12, None) - 1.0
    return float(np.min(dd))


def load_merged_holdout(results_dir: Path, model: str, split: str) -> pd.DataFrame | None:
    """split is 'val' or 'test'."""
    suf = result_suffix(model)
    w_path = results_dir / f"ipo_optimizer_weights_{split}{suf}.csv"
    r_path = results_dir / f"ipo_optimizer_returns_{split}{suf}.csv"
    if not w_path.is_file() or not r_path.is_file():
        print(f"warning: missing weights or returns for {model} {split}", file=sys.stderr)
        return None
    w = pd.read_csv(w_path)
    r = pd.read_csv(r_path)
    m = r.merge(w, on="date", how="inner")
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values("date")
    m["model_return"] = m["weight_market"] * m["market_return"] + m["weight_IPO"] * m["ipo_return"]
    m["excess_vs_5050"] = m["model_return"] - m["equal_weight_return"]
    return m


def load_selection(results_dir: Path, model: str) -> dict | None:
    suf = result_suffix(model)
    path = results_dir / f"ipo_optimizer_selection_metrics{suf}.json"
    if not path.is_file():
        print(f"warning: missing {path}", file=sys.stderr)
        return None
    return json.loads(path.read_text())


def plot_holdout_cumwealth_semilogy(
    merged: dict[str, pd.DataFrame], split: str, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    first = next(iter(merged.values()))
    ew = np.cumprod(1.0 + first["equal_weight_return"].values)
    ax.semilogy(first["date"], ew, label="Equal 50/50", color="black", linestyle="--", linewidth=1.5, alpha=0.8)
    for model, df in merged.items():
        w = np.cumprod(1.0 + df["model_return"].values)
        ax.semilogy(df["date"], w, label=f"{model.upper()}", linewidth=2)
    ax.set_title(f"{split.capitalize()}: cumulative wealth (1+r) — log scale")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth index")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_holdout_cumulative_return_benchmark_linear(
    merged: dict[str, pd.DataFrame], split: str, out_path: Path
) -> None:
    """Cumulative total return Π(1+r)-1 for each model and the equal 50/50 benchmark (linear scale)."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    first = next(iter(merged.values()))
    bench = np.cumprod(1.0 + first["equal_weight_return"].values) - 1.0
    ax.plot(
        first["date"],
        bench,
        label="Equal 50/50",
        color="black",
        linestyle="--",
        linewidth=2.2,
        zorder=2,
    )
    for model, df in merged.items():
        cum_ret = np.cumprod(1.0 + df["model_return"].values) - 1.0
        ax.plot(df["date"], cum_ret, label=model.upper(), linewidth=2)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_title(f"{split.capitalize()}: cumulative return vs 50/50 benchmark")
    # y = ∏(1+r_day)−1 over the holdout calendar (not annualized; grows fast if mean daily r is large).
    ax.set_ylabel(r"Cumulative compound return, $\prod_t(1+r_t)-1$")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_holdout_cum_excess_linear(merged: dict[str, pd.DataFrame], split: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, df in merged.items():
        excess = np.cumprod(1.0 + df["excess_vs_5050"].values) - 1.0
        ax.plot(df["date"], excess, label=model.upper(), linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title(f"{split.capitalize()}: cumulative excess vs equal 50/50 (linear)")
    ax.set_ylabel("Cumulative excess return")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_holdout_rolling_excess(
    merged: dict[str, pd.DataFrame], split: str, window: int, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    min_p = max(5, window // 3)
    for model, df in merged.items():
        roll = df["excess_vs_5050"].rolling(window=window, min_periods=min_p).sum()
        ax.plot(df["date"], roll, label=model.upper(), linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title(f"{split.capitalize()}: rolling {window}d sum of excess vs 50/50")
    ax.set_ylabel("Rolling sum")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_holdout_rolling_vol_semilogy(
    merged: dict[str, pd.DataFrame], split: str, window: int, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    min_p = max(5, window // 3)
    for model, df in merged.items():
        vol = df["model_return"].rolling(window=window, min_periods=min_p).std()
        ax.semilogy(df["date"], np.clip(vol.values, 1e-8, None), label=model.upper(), linewidth=2)
    ax.set_title(f"{split.capitalize()}: rolling {window}d stdev of daily returns (log scale)")
    ax.set_ylabel("Rolling stdev")
    ax.set_xlabel("Date")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_scalar_bars(selection_by_model: dict[str, dict], out_path: Path) -> None:
    models = list(selection_by_model.keys())
    if not models:
        return
    val_obj = [selection_by_model[m]["validation"]["objective"] for m in models]
    test_obj = [selection_by_model[m]["test"]["objective"] for m in models]
    val_tq = [abs(selection_by_model[m]["validation"]["tail_q_excess"]) for m in models]
    test_tq = [abs(selection_by_model[m]["test"]["tail_q_excess"]) for m in models]

    x = np.arange(len(models))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, title, a, b in zip(
        axes,
        ["Selection objective (lower is better)", "|Tail-q excess| (lower is better)"],
        [(val_obj, test_obj), (val_tq, test_tq)],
        [("val", "test"), ("val", "test")],
    ):
        ax.bar(x - w / 2, a[0], w, label="Validation")
        ax.bar(x + w / 2, a[1], w, label="Test")
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models])
        ax.set_title(title)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Checkpoint selection metrics from JSON (final model)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_scalar_sharpe_dd(
    merged_test: dict[str, pd.DataFrame], out_path: Path
) -> None:
    models = list(merged_test.keys())
    sharpes = [sharpe_from_ret(merged_test[m]["model_return"].values) for m in models]
    dds = [max_drawdown_from_ret(merged_test[m]["model_return"].values) for m in models]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x, sharpes, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.upper() for m in models])
    axes[0].set_title("Test Sharpe (annualized, √252)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(x, dds, color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.upper() for m in models])
    axes[1].set_title("Test max drawdown")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Test-period summary from daily returns")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ROOT / "vm_artifacts_20260408_1146",
        help="Directory containing results/ with CSVs and JSON",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "figures" / "model_comparison_vm",
        help="Output directory for PNGs",
    )
    p.add_argument("--rolling-window", type=int, default=21)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.artifacts_dir / "results"
    if not results_dir.is_dir():
        print(f"error: {results_dir} is not a directory", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    histories = load_histories(results_dir)
    if histories:
        plot_epoch_semilog_losses(histories, args.out_dir / "epoch_semilog_losses.png")
        plot_epoch_semilog_lr(histories, args.out_dir / "epoch_semilog_lr.png")
        plot_epoch_semilog_val_components(histories, args.out_dir / "epoch_semilog_val_risk.png")
        plot_epoch_linear_val_excess(histories, args.out_dir / "epoch_linear_val_excess_selection.png")

    rolling = args.rolling_window
    for split in ("val", "test"):
        merged: dict[str, pd.DataFrame] = {}
        for m in MODELS:
            d = load_merged_holdout(results_dir, m, split)
            if d is not None:
                merged[m] = d
        if not merged:
            continue
        plot_holdout_cumwealth_semilogy(merged, split, args.out_dir / f"{split}_cumwealth_semilogy.png")
        plot_holdout_cumulative_return_benchmark_linear(
            merged, split, args.out_dir / f"{split}_cumulative_return_benchmark_linear.png"
        )
        plot_holdout_cum_excess_linear(merged, split, args.out_dir / f"{split}_cum_excess_vs_5050.png")
        plot_holdout_rolling_excess(merged, split, rolling, args.out_dir / f"{split}_rolling_excess_w{rolling}.png")
        plot_holdout_rolling_vol_semilogy(
            merged, split, rolling, args.out_dir / f"{split}_rolling_vol_semilogy_w{rolling}.png"
        )

    merged_test = {m: load_merged_holdout(results_dir, m, "test") for m in MODELS}
    merged_test = {k: v for k, v in merged_test.items() if v is not None}
    selection_by_model: dict[str, dict] = {}
    for m in MODELS:
        sel = load_selection(results_dir, m)
        if sel is not None:
            selection_by_model[m] = sel
    if selection_by_model:
        plot_scalar_bars(selection_by_model, args.out_dir / "scalar_selection_objective_tailq.png")
    if merged_test:
        plot_scalar_sharpe_dd(merged_test, args.out_dir / "scalar_test_sharpe_drawdown.png")

    # Summary table CSV
    rows = []
    for m in MODELS:
        sel = selection_by_model.get(m)
        te = merged_test.get(m)
        row = {"model": m}
        if sel:
            row.update(
                {
                    "val_objective": sel["validation"]["objective"],
                    "val_tail_q_excess": sel["validation"]["tail_q_excess"],
                    "test_objective": sel["test"]["objective"],
                    "test_tail_q_excess": sel["test"]["tail_q_excess"],
                }
            )
        if te is not None:
            row["test_sharpe"] = sharpe_from_ret(te["model_return"].values)
            row["test_max_drawdown"] = max_drawdown_from_ret(te["model_return"].values)
            row["test_mean_ipo_weight"] = float(te["weight_IPO"].mean())
        rows.append(row)
    if rows:
        summary = pd.DataFrame(rows)
        summary_path = args.out_dir / "model_comparison_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Wrote {summary_path}")

    print(f"Figures saved under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
