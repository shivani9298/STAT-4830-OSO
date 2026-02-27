#!/usr/bin/env python3
"""
Tune hyperparameters for IPO Portfolio Optimizer.

Grid search over key hyperparameters; optimize validation Sharpe.
Saves best config to results/ipo_optimizer_best_config.json.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from run_ipo_optimizer_wrds import get_connection, prepare_data
from src.data_layer import build_rolling_windows, train_val_split
from src.train import run_training
from src.export import predict_weights, portfolio_stats

# Hyperparameter grid (all configurations; 16 configs ~1.5 hr)
GRID = {
    "window_len": [84, 126],
    "val_frac": [0.2],
    "lr": [1e-3],
    "batch_size": [32],
    "lambda_vol": [0.5, 1.0],
    "lambda_cvar": [0.5, 1.0],
    "lambda_vol_excess": [0.5, 1.0],
    "target_vol_annual": [0.20, 0.25],
    "hidden_size": [64],
    "patience": [10],
    "epochs": [50],
}


def run_config(data_prep, config, device):
    """Train with given config, return validation Sharpe and stats."""
    df = data_prep["df"]
    feature_cols = data_prep["feature_cols"]
    window_len = config["window_len"]
    val_frac = config["val_frac"]

    X, R, dates = build_rolling_windows(df, window_len=window_len, feature_cols=feature_cols)
    if X.shape[0] < 50:
        return float("-inf"), {}

    X_train, R_train, d_train, X_val, R_val, d_val = train_val_split(X, R, dates, val_frac=val_frac)
    if X_val.shape[0] < 10:
        return float("-inf"), {}

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
        "window_len": window_len,
    }

    model, _ = run_training(
        data,
        device=device,
        epochs=config["epochs"],
        lr=config["lr"],
        batch_size=config["batch_size"],
        patience=config["patience"],
        lambda_cvar=config["lambda_cvar"],
        lambda_turnover=0.01,
        lambda_vol=config["lambda_vol"],
        lambda_path=0.01,
        lambda_vol_excess=config.get("lambda_vol_excess", 1.0),
        target_vol_annual=config.get("target_vol_annual", 0.25),
        lambda_diversify=0.0,
        min_weight=0.1,
        hidden_size=config["hidden_size"],
        num_layers=1,
        model_type="gru",
    )
    weights = predict_weights(model, data["X_val"], device)
    stats = portfolio_stats(weights, data["R_val"])
    return stats["sharpe_annualized"], stats


def main():
    print("Connecting to WRDS...", flush=True)
    conn = get_connection()
    print("Preparing data...", flush=True)
    data_prep = prepare_data(conn)

    keys = list(GRID.keys())
    configs = [
        dict(zip(keys, vals))
        for vals in product(*(GRID[k] for k in keys))
    ]
    print(f"Tuning over {len(configs)} configurations...", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_sharpe = float("-inf")
    best_config = None
    best_stats = {}
    results = []
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    config_path = out_dir / "ipo_optimizer_best_config.json"

    def _save_best():
        """Save best config so far (enables use of partial results if stopped early)."""
        if best_config is None:
            return
        out = {
            "best_config": best_config,
            "best_sharpe": best_sharpe,
            "best_stats": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in best_stats.items()},
            "all_results": [
                {"config": r["config"], "sharpe": r["sharpe"], "stats": r.get("stats", {})}
                for r in results
            ],
        }
        with open(config_path, "w") as f:
            json.dump(out, f, indent=2)

    for i, config in enumerate(configs):
        try:
            sharpe, stats = run_config(data_prep, config, device)
            results.append({"config": config, "sharpe": sharpe, "stats": stats})
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_config = config.copy()
                best_stats = stats.copy()
                _save_best()
                print(f"  [{i+1}/{len(configs)}] New best: Sharpe={sharpe:.2f} | {config}", flush=True)
            elif (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(configs)}] Sharpe={sharpe:.2f}", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(configs)}] FAILED: {e}", flush=True)
            results.append({"config": config, "sharpe": float("-inf"), "error": str(e)})

    best_out = {
        "best_config": best_config,
        "best_sharpe": best_sharpe,
        "best_stats": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in best_stats.items()},
        "all_results": [
            {
                "config": r["config"],
                "sharpe": r["sharpe"],
                "stats": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r.get("stats", {}).items()},
            }
            for r in results
        ],
    }
    config_path = out_dir / "ipo_optimizer_best_config.json"
    with open(config_path, "w") as f:
        json.dump(best_out, f, indent=2)

    print("\n" + "=" * 50)
    print("Best config:", json.dumps(best_config, indent=2))
    print(f"Validation Sharpe: {best_sharpe:.2f}")
    print(f"Saved to {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
