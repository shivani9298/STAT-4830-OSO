#!/usr/bin/env python3
from __future__ import annotations

import json
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
from src.export import predict_weights
from src.train import run_training


def load_cached_returns() -> pd.DataFrame:
    candidates = [
        ROOT / "results" / "recent" / "ipo_180day_mcap_returns.csv",
        ROOT / "results" / "older" / "ipo_180day_mcap_returns.csv",
        ROOT / "results" / "recent" / "ipo_optimizer_returns_val.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        raw = pd.read_csv(p, parse_dates=[0]).rename(columns={"Unnamed: 0": "date"})
        if {"SPY_Only", "IPO_Only"}.issubset(raw.columns):
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(raw["date"]),
                    "market_return": raw["SPY_Only"].astype(float).clip(-0.10, 0.10),
                    "ipo_return": raw["IPO_Only"].astype(float).clip(-0.50, 0.50),
                }
            ).dropna()
        if {"market_return", "ipo_return"}.issubset(raw.columns):
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(raw["date"]),
                    "market_return": raw["market_return"].astype(float).clip(-0.10, 0.10),
                    "ipo_return": raw["ipo_return"].astype(float).clip(-0.50, 0.50),
                }
            ).dropna()
    raise FileNotFoundError("No suitable cached IPO/market returns source found in results/")


def summary_delta(base_w: np.ndarray, pert_w: np.ndarray) -> dict[str, float]:
    d = pert_w[:, 1] - base_w[:, 1]  # IPO weight delta
    ad = np.abs(d)
    return {
        "mean_abs_delta_ipo_weight": float(ad.mean()),
        "max_abs_delta_ipo_weight": float(ad.max()),
        "std_delta_ipo_weight": float(d.std()),
        "mean_signed_delta_ipo_weight": float(d.mean()),
    }


def main() -> int:
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build the same offline static setup (full objective defaults).
    df = load_cached_returns().set_index("date")
    df = add_optional_features(df, include_vix=False)
    feature_cols = list(df.columns)
    X, R, dates = build_rolling_windows(df, window_len=126, feature_cols=feature_cols)
    X_tr, R_tr, d_tr, X_va, R_va, d_va = train_val_split(X, R, dates, val_frac=0.2)
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
    print(f"Train windows={len(X_tr)}, val windows={len(X_va)}")

    model, history = run_training(
        data,
        device=device,
        epochs=80,
        lr=3e-4,
        lr_decay=0.1,
        batch_size=256,
        patience=20,
        lambda_vol=0.5,
        lambda_cvar=0.5,
        lambda_turnover=1e-4,
        lambda_path=1e-4,
        lambda_diversify=0.0,
        min_weight=0.1,
        lambda_vol_excess=1.0,
        target_vol_annual=0.25,
        hidden_size=64,
        model_type="gru",
        mean_return_weight=1.0,
        log_growth_weight=0.0,
        verbose=True,
        log_every=20,
    )
    base_w = predict_weights(model, X_va, device)
    print("Baseline IPO weight range:", float(base_w[:, 1].min()), float(base_w[:, 1].max()))

    out_dir = ROOT / "results"
    fig_dir = ROOT / "figures" / "experiments" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Feature-wise std from validation panel.
    val_flat = X_va.reshape(-1, X_va.shape[-1])
    feat_std = np.std(val_flat, axis=0) + 1e-12

    tests: list[tuple[str, np.ndarray]] = []

    # 1) Global Gaussian perturbation.
    sigma = 0.10 * feat_std.reshape(1, 1, -1)
    noise = np.random.normal(0.0, 1.0, size=X_va.shape) * sigma
    tests.append(("global_noise_0.1std", X_va + noise))

    # 2) Shuffle time order inside each window (destroys temporal patterns).
    X_shuffle = X_va.copy()
    for i in range(X_shuffle.shape[0]):
        perm = np.random.permutation(X_shuffle.shape[1])
        X_shuffle[i] = X_shuffle[i, perm, :]
    tests.append(("time_shuffle_within_window", X_shuffle))

    # 3) Zero all features.
    tests.append(("all_features_zeroed", np.zeros_like(X_va)))

    # 4+) Feature shocks (+1 std on all timesteps) one-by-one.
    for j, f in enumerate(feature_cols):
        Xp = X_va.copy()
        Xp[:, :, j] += feat_std[j]
        tests.append((f"feature_plus_1std::{f}", Xp))

    rows: list[dict[str, float | str]] = []
    for name, Xp in tests:
        wp = predict_weights(model, Xp, device)
        stats = summary_delta(base_w, wp)
        row: dict[str, float | str] = {"test": name}
        row.update(stats)
        rows.append(row)

    res_df = pd.DataFrame(rows).sort_values("mean_abs_delta_ipo_weight", ascending=False)
    csv_path = out_dir / "sensitivity_test_full_objective.csv"
    res_df.to_csv(csv_path, index=False)

    # Save compact summary JSON for quick read.
    top = res_df.head(10).to_dict(orient="records")
    summary = {
        "n_tests": int(len(res_df)),
        "top_by_mean_abs_delta_ipo_weight": top,
        "baseline_ipo_weight_mean": float(base_w[:, 1].mean()),
        "baseline_ipo_weight_min": float(base_w[:, 1].min()),
        "baseline_ipo_weight_max": float(base_w[:, 1].max()),
        "epochs_trained": int(len(history)),
    }
    json_path = out_dir / "sensitivity_test_full_objective_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # Plot top sensitivity bars.
    topn = res_df.head(12).copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.barh(topn["test"][::-1], topn["mean_abs_delta_ipo_weight"][::-1], color="C0")
    ax.set_title("Sensitivity test (full objective): mean |delta IPO weight|")
    ax.set_xlabel("Mean absolute IPO-weight change")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    bar_path = fig_dir / "sensitivity_top_mean_abs_delta_ipo_weight.png"
    fig.savefig(bar_path, dpi=170)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {bar_path}")
    print(res_df.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
