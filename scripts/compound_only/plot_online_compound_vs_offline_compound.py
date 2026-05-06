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

from run_ipo_optimizer_wrds import apply_env_overrides, load_best_config, prepare_data
from src.data_layer import build_online_schedule, build_rolling_windows, slice_windows_by_index
from src.export import predict_weights
from src.train import run_training
from src.wrds_data import get_connection


def cum_pct(x: np.ndarray) -> np.ndarray:
    return (np.cumprod(1.0 + np.asarray(x, dtype=float)) - 1.0) * 100.0


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    online_path = ROOT / "results" / "ipo_optimizer_online_path.csv"
    if not online_path.exists():
        raise FileNotFoundError(f"Missing {online_path}")
    online_df = pd.read_csv(online_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    # Online run under comparison.
    online_ret = online_df["net_ret"].astype(float).to_numpy()
    market_ret = online_df["realized_market"].astype(float).to_numpy()
    ipo_ret = online_df["realized_ipo"].astype(float).to_numpy()
    eq_ret = 0.5 * market_ret + 0.5 * ipo_ret
    w_online_ipo = online_df["weight_ipo"].astype(float).to_numpy()
    dates_online = pd.to_datetime(online_df["date"])

    # Rebuild WRDS window panel so we can evaluate a static offline compound-only model on the same dates.
    print("Loading WRDS data for offline same-date comparator...")
    conn = get_connection()
    prep = prepare_data(conn)
    df = prep["df"]
    feature_cols = prep["feature_cols"]
    cfg = apply_env_overrides(load_best_config())
    cfg["compound_only_loss"] = True
    cfg["mean_return_weight"] = 0.0
    cfg["log_growth_weight"] = 1.0
    cfg["lambda_cvar"] = 0.0
    cfg["lambda_vol"] = 0.0
    cfg["lambda_turnover"] = 0.0
    cfg["lambda_path"] = 0.0
    cfg["lambda_vol_excess"] = 0.0
    cfg["lambda_diversify"] = 0.0

    X, R, dates = build_rolling_windows(df, window_len=cfg["window_len"], feature_cols=feature_cols)
    schedule = build_online_schedule(
        dates,
        warmup_windows=int(cfg.get("warmup_windows", 252)),
        update_freq=str(cfg.get("update_freq", "W")),
        step=1,
        decision_lag=int(cfg.get("decision_lag", 0)),
    )
    if not schedule:
        raise RuntimeError("Empty online schedule; cannot build comparator.")
    first_train_end = int(schedule[0]["train_end_idx"])

    # Train a static offline model on initial history only, then evaluate on exact online eval dates.
    X_hist0, R_hist0, _ = slice_windows_by_index(X, R, dates, end_idx=first_train_end)
    n0 = X_hist0.shape[0]
    if n0 < 2:
        raise RuntimeError("Not enough initial history windows for offline comparator.")
    n0_val = max(1, int(n0 * float(cfg.get("val_frac", 0.2))))
    split = max(1, n0 - n0_val)
    if split >= n0:
        split = n0 - 1
    offline_data = {
        "X_train": X_hist0[:split],
        "R_train": R_hist0[:split],
        "X_val": X_hist0[split:],
        "R_val": R_hist0[split:],
        "n_assets": 2,
        "window_len": cfg["window_len"],
    }
    print("Training offline compound-only comparator model...")
    offline_model, _ = run_training(
        offline_data,
        device=device,
        epochs=int(cfg["epochs"]),
        lr=float(cfg["lr"]),
        lr_decay=float(cfg.get("lr_decay", 0.1)),
        batch_size=int(cfg["batch_size"]),
        patience=int(cfg["patience"]),
        lambda_vol=0.0,
        lambda_cvar=0.0,
        lambda_turnover=0.0,
        lambda_path=0.0,
        lambda_diversify=0.0,
        min_weight=float(cfg.get("min_weight", 0.1)),
        lambda_vol_excess=0.0,
        target_vol_annual=float(cfg.get("target_vol_annual", 0.25)),
        hidden_size=int(cfg["hidden_size"]),
        model_type=str(cfg.get("model_type", "gru")),
        mean_return_weight=0.0,
        log_growth_weight=1.0,
        verbose=False,
        log_every=0,
    )

    # Map online dates -> eval window indices
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(pd.to_datetime(dates))}
    eval_indices = [date_to_idx[pd.Timestamp(d)] for d in dates_online if pd.Timestamp(d) in date_to_idx]
    if len(eval_indices) != len(dates_online):
        raise RuntimeError("Failed to map all online dates to rolling-window indices.")
    eval_indices_arr = np.asarray(eval_indices, dtype=int)
    w_off = predict_weights(offline_model, X[eval_indices_arr], device)
    r_off = (w_off * R[eval_indices_arr]).sum(axis=1)
    w_off_ipo = w_off[:, 1]

    out_results = ROOT / "results"
    out_figs = ROOT / "figures" / "online_evaluation"
    out_results.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    cmp_csv = out_results / "online_compound_vs_offline_compound_same_dates.csv"
    pd.DataFrame(
        {
            "date": dates_online,
            "ret_online_compound": online_ret,
            "ret_offline_compound_static": r_off,
            "ret_5050": eq_ret,
            "ret_market_only": market_ret,
            "ret_ipo_only": ipo_ret,
            "weight_ipo_online_compound": w_online_ipo,
            "weight_ipo_offline_compound_static": w_off_ipo,
        }
    ).to_csv(cmp_csv, index=False)

    # Returns trajectory plot
    ret_png = out_figs / "online_compound_returns_vs_benchmarks_and_offline_compound.png"
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    ax.plot(dates_online, cum_pct(online_ret), linewidth=2.2, label="Online compound-only")
    ax.plot(dates_online, cum_pct(r_off), linewidth=2.2, label="Offline compound-only (static, same dates)")
    ax.plot(dates_online, cum_pct(eq_ret), "--", linewidth=1.8, label="50/50")
    ax.plot(dates_online, cum_pct(market_ret), "-.", linewidth=1.6, label="Market-only")
    ax.plot(dates_online, cum_pct(ipo_ret), ":", linewidth=1.8, label="IPO-only")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title("Online compound-only: returns vs benchmarks + offline compound comparator")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(ret_png, dpi=170)
    plt.close(fig)

    # IPO weight trajectory plot
    w_png = out_figs / "online_compound_ipo_weight_trajectory.png"
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.plot(dates_online, w_online_ipo, linewidth=1.9, label="Online compound-only IPO weight")
    ax.plot(dates_online, w_off_ipo, linewidth=1.6, linestyle="--", label="Offline compound-only static IPO weight")
    ax.set_title("IPO weight trajectory (exact online compound-only run)")
    ax.set_xlabel("Date")
    ax.set_ylabel("IPO allocation weight")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(w_png, dpi=170)
    plt.close(fig)

    print(f"Wrote {cmp_csv}")
    print(f"Wrote {ret_png}")
    print(f"Wrote {w_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
