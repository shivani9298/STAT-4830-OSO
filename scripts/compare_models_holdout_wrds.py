#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "run_ipo_optimizer_wrds.py"


def suffix_for_model(model: str) -> str:
    return "" if model == "gru" else f"_{model}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and compare GRU/LSTM/Transformer on holdout WRDS split.")
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date", default="2025-12-31")
    p.add_argument("--val-start", default=None)
    p.add_argument("--test-start", default=None)
    p.add_argument("--max-history", action="store_true")
    p.add_argument("--rolling-window", type=int, default=21)
    p.add_argument("--rolling-tail-quantile", type=float, default=0.10)
    p.add_argument("--selection-drawdown-penalty", type=float, default=0.0)
    p.add_argument("--ipo-index-method", default="fast", choices=["fast", "legacy"])
    p.add_argument("--use-cache", action="store_true")
    p.add_argument("--cache-dir", default=str(ROOT / "results" / "cache_wrds"))
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--models", nargs="+", default=["gru", "lstm", "transformer"])
    return p.parse_args()


def run_model(args: argparse.Namespace, model: str) -> None:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--model",
        model,
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--selection-metric",
        "rolling_tail_excess",
        "--rolling-window",
        str(args.rolling_window),
        "--rolling-tail-quantile",
        str(args.rolling_tail_quantile),
        "--selection-drawdown-penalty",
        str(args.selection_drawdown_penalty),
        "--ipo-index-method",
        args.ipo_index_method,
    ]
    if args.max_history:
        cmd.append("--max-history")
    if args.use_cache:
        cmd.append("--use-cache")
        cmd.extend(["--cache-dir", args.cache_dir])
    if args.val_start:
        cmd.extend(["--val-start", args.val_start])
    if args.test_start:
        cmd.extend(["--test-start", args.test_start])
    print(f"[run] {model}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


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


def _load_outputs(model: str) -> dict:
    suffix = suffix_for_model(model)
    res_dir = ROOT / "results"

    sel = json.loads((res_dir / f"ipo_optimizer_selection_metrics{suffix}.json").read_text())
    wv = pd.read_csv(res_dir / f"ipo_optimizer_weights_val{suffix}.csv")
    wt = pd.read_csv(res_dir / f"ipo_optimizer_weights_test{suffix}.csv")
    rv = pd.read_csv(res_dir / f"ipo_optimizer_returns_val{suffix}.csv")
    rt = pd.read_csv(res_dir / f"ipo_optimizer_returns_test{suffix}.csv")

    merged_val = rv.merge(wv, on="date", how="inner")
    merged_test = rt.merge(wt, on="date", how="inner")

    merged_val["model_return"] = (
        merged_val["weight_market"] * merged_val["market_return"]
        + merged_val["weight_IPO"] * merged_val["ipo_return"]
    )
    merged_test["model_return"] = (
        merged_test["weight_market"] * merged_test["market_return"]
        + merged_test["weight_IPO"] * merged_test["ipo_return"]
    )
    merged_val["excess_vs_5050"] = merged_val["model_return"] - merged_val["equal_weight_return"]
    merged_test["excess_vs_5050"] = merged_test["model_return"] - merged_test["equal_weight_return"]

    return {
        "selection": sel,
        "val": merged_val,
        "test": merged_test,
    }


def _plot_overlays(outputs: dict[str, dict], rolling_window: int) -> None:
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Test cumulative excess vs 50/50 overlay.
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, out in outputs.items():
        df = out["test"].copy()
        cum_excess = np.cumprod(1.0 + df["excess_vs_5050"].values) - 1.0
        ax.plot(pd.to_datetime(df["date"]), cum_excess, label=model.upper(), linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title("Test Cumulative Excess Return vs Equal 50/50")
    ax.set_ylabel("Cumulative Excess Return")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "model_comparison_test_cum_excess_vs_5050.png", dpi=160)
    plt.close(fig)

    # Validation rolling excess overlay.
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, out in outputs.items():
        df = out["val"].copy()
        roll = (
            df["excess_vs_5050"]
            .rolling(window=rolling_window, min_periods=max(5, rolling_window // 3))
            .sum()
        )
        ax.plot(pd.to_datetime(df["date"]), roll, label=model.upper(), linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title(f"Validation Rolling {rolling_window}d Excess vs Equal 50/50")
    ax.set_ylabel("Rolling Excess Return Sum")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "model_comparison_val_rolling_excess_vs_5050.png", dpi=160)
    plt.close(fig)


def _build_table(outputs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model, out in outputs.items():
        val_df = out["val"]
        test_df = out["test"]
        sel = out["selection"]
        rows.append(
            {
                "model": model,
                "val_tail_q_excess": sel["validation"]["tail_q_excess"],
                "val_selection_objective": sel["validation"]["objective"],
                "val_sharpe": sharpe_from_ret(val_df["model_return"].values),
                "val_max_drawdown": max_drawdown_from_ret(val_df["model_return"].values),
                "test_tail_q_excess": sel["test"]["tail_q_excess"],
                "test_selection_objective": sel["test"]["objective"],
                "test_sharpe": sharpe_from_ret(test_df["model_return"].values),
                "test_max_drawdown": max_drawdown_from_ret(test_df["model_return"].values),
                "test_avg_ipo_weight": float(test_df["weight_IPO"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    table = table.sort_values(
        by=["test_tail_q_excess", "test_sharpe"],
        ascending=[False, False],
    ).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    cols = [
        "rank",
        "model",
        "val_tail_q_excess",
        "val_selection_objective",
        "val_sharpe",
        "val_max_drawdown",
        "test_tail_q_excess",
        "test_selection_objective",
        "test_sharpe",
        "test_max_drawdown",
        "test_avg_ipo_weight",
    ]
    return table[cols]


def main() -> int:
    args = parse_args()
    models = [m.lower() for m in args.models]

    if not args.skip_train:
        for model in models:
            run_model(args, model)

    outputs = {m: _load_outputs(m) for m in models}
    _plot_overlays(outputs, rolling_window=args.rolling_window)
    table = _build_table(outputs)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    table_csv = out_dir / "model_comparison_holdout_table.csv"
    table_md = out_dir / "model_comparison_holdout_table.md"
    table.to_csv(table_csv, index=False)
    with open(table_md, "w") as f:
        f.write(table.to_string(index=False))

    print("\nModel ranking (primary: test_tail_q_excess):")
    print(table.to_string(index=False))
    print(f"\nSaved table to {table_csv}")
    print(f"Saved markdown to {table_md}")
    print("Saved plots to figures/model_comparison_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
