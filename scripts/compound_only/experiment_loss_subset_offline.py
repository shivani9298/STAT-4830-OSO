#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_layer import add_optional_features, build_rolling_windows, train_val_split
from src.export import export_summary, export_weights_csv, portfolio_stats, predict_weights
from src.plot_loss import plot_cumulative_returns_vs_equal_weight, plot_training_loss
from src.train import run_training


def load_cached_returns() -> pd.DataFrame:
    p = ROOT / "results" / "recent" / "ipo_180day_mcap_returns.csv"
    raw = pd.read_csv(p, parse_dates=[0]).rename(columns={"Unnamed: 0": "date"})
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"]),
            "market_return": raw["SPY_Only"].astype(float).clip(-0.10, 0.10),
            "ipo_return": raw["IPO_Only"].astype(float).clip(-0.50, 0.50),
        }
    ).dropna()
    return df


def cumulative_pct(r: np.ndarray) -> np.ndarray:
    return (np.cumprod(1.0 + np.asarray(r, dtype=float)) - 1.0) * 100.0


def run_one(
    name: str,
    data: dict,
    device: torch.device,
    out_results: Path,
    out_figs: Path,
    lambda_diversify: float,
) -> dict:
    model, history = run_training(
        data,
        device=device,
        epochs=80,
        lr=3e-4,
        lr_decay=0.1,
        batch_size=256,
        patience=20,
        lambda_vol=0.0,  # remove variance term
        lambda_cvar=0.5,
        lambda_turnover=1e-4,
        lambda_path=0.0,  # remove weight_path term
        lambda_diversify=lambda_diversify,
        min_weight=0.1,
        lambda_vol_excess=1.0,
        target_vol_annual=0.25,
        hidden_size=64,
        model_type="gru",
        mean_return_weight=0.0,
        log_growth_weight=1.0,
        log_discount_gamma=0.95,
        verbose=True,
        log_every=10,
    )

    hist_df = pd.DataFrame(history)
    hist_path = out_results / f"training_history_loss_subset_rllog_{name}.csv"
    hist_df.to_csv(hist_path, index=False)

    w = predict_weights(model, data["X_val"], device)
    stats = portfolio_stats(w, data["R_val"])
    w_path = out_results / f"ipo_optimizer_weights_loss_subset_rllog_{name}.csv"
    s_path = out_results / f"ipo_optimizer_summary_loss_subset_rllog_{name}.txt"
    export_weights_csv(data["dates_val"], w, w_path)
    export_summary(stats, w, s_path, R=data["R_val"])

    plot_training_loss(
        history,
        out_figs / f"loss_subset_rllog_train_val_linear_{name}.png",
        title=f"Loss subset + discounted log-return ({name}): train vs val loss",
        semilogy=False,
        rolling_epochs=3,
    )
    plot_training_loss(
        history,
        out_figs / f"loss_subset_rllog_train_val_semilogy_{name}.png",
        title=f"Loss subset + discounted log-return ({name}): train vs val loss (semilogy)",
        semilogy=True,
        rolling_epochs=3,
    )
    plot_cumulative_returns_vs_equal_weight(
        w,
        data["R_val"],
        data["dates_val"],
        out_figs / f"loss_subset_rllog_returns_vs_5050_{name}.png",
        title=f"Validation returns with discounted log-return ({name}) vs 50/50",
    )

    model_ret = (w * data["R_val"]).sum(axis=1)
    return {
        "name": name,
        "lambda_diversify": float(lambda_diversify),
        "stats": stats,
        "model_ret": model_ret,
        "dates_val": np.asarray(data["dates_val"]),
        "R_val": np.asarray(data["R_val"]),
        "weights": w,
        "history": history,
        "hist_csv": str(hist_path),
        "weights_csv": str(w_path),
        "summary_txt": str(s_path),
    }


def plot_returns_comparison(results: list[dict], out_path: Path) -> None:
    ref = results[0]
    dates = pd.to_datetime(ref["dates_val"])
    R = ref["R_val"]
    mkt = R[:, 0]
    ipo = R[:, 1]
    eq = 0.5 * mkt + 0.5 * ipo

    fig, ax = plt.subplots(figsize=(10, 5.3))
    for r in results:
        ax.plot(dates, cumulative_pct(r["model_ret"]), linewidth=2.0, label=f"Model ({r['name']})")
    ax.plot(dates, cumulative_pct(eq), "--", linewidth=1.8, label="50/50")
    ax.plot(dates, cumulative_pct(mkt), "-.", linewidth=1.6, label="Market only")
    ax.plot(dates, cumulative_pct(ipo), ":", linewidth=1.8, label="IPO only")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title("Loss-subset + discounted log-return: cumulative validation returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_cached_returns().set_index("date")
    df = add_optional_features(df, include_vix=False)
    feature_cols = list(df.columns)
    X, R, dates = build_rolling_windows(df, window_len=126, feature_cols=feature_cols)
    X_tr, R_tr, d_tr, X_va, R_va, d_va = train_val_split(X, R, dates, val_frac=0.2)
    print(f"Train windows={len(X_tr)}, val windows={len(X_va)}")

    data = {
        "X_train": X_tr,
        "R_train": R_tr,
        "dates_train": d_tr,
        "X_val": X_va,
        "R_val": R_va,
        "dates_val": d_va,
        "feature_cols": feature_cols,
        "df": df,
        "n_assets": 2,
        "window_len": 126,
    }

    out_results = ROOT / "results"
    out_figs = ROOT / "figures" / "experiments" / "loss_subset"
    out_results.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    variants = [
        ("no_diversify", 0.0),
        ("with_diversify", 0.1),
    ]
    res = []
    for name, ld in variants:
        print(f"\n=== Running variant: {name} (lambda_diversify={ld}) ===")
        res.append(run_one(name, data, device, out_results, out_figs, lambda_diversify=ld))

    cmp_png = out_figs / "loss_subset_rllog_returns_comparison.png"
    plot_returns_comparison(res, cmp_png)

    rows = []
    for r in res:
        s = r["stats"]
        rows.append(
            {
                "variant": r["name"],
                "lambda_diversify": r["lambda_diversify"],
                "total_return": s["total_return"],
                "ann_return": s["return_annualized"],
                "ann_vol": s["volatility_annualized"],
                "sharpe": s["sharpe_annualized"],
                "max_drawdown": s["max_drawdown"],
                "avg_ipo_weight": float(np.mean(r["weights"][:, 1])),
                "summary_path": r["summary_txt"],
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_csv = out_results / "loss_subset_rllog_experiment_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nWrote summary: {summary_csv}")
    print(f"Wrote comparison returns figure: {cmp_png}")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
