#!/usr/bin/env python3
"""
Systematic diagnostics for near-constant portfolio weights.

What it does:
1) Inspect existing exported weights/returns CSVs (if present).
2) Run a deterministic synthetic sanity check to verify the GRU can learn
   time-varying allocations when predictive signal exists.
3) Run a small ablation over loss settings to identify terms that can freeze weights.
4) Write a JSON report and a recommended config override JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.export import predict_weights
from src.train import run_training


def analyze_existing_outputs(weights_path: Path, returns_path: Path) -> dict:
    out: dict[str, object] = {}
    if weights_path.exists():
        w = pd.read_csv(weights_path)
        wm = w["weight_market"].to_numpy(dtype=float)
        wi = w["weight_IPO"].to_numpy(dtype=float)
        out["weights"] = {
            "n_rows": int(len(w)),
            "market_mean": float(np.mean(wm)),
            "ipo_mean": float(np.mean(wi)),
            "market_std": float(np.std(wm)),
            "ipo_std": float(np.std(wi)),
            "avg_turnover": float(np.mean(np.abs(np.diff(wi)))) if len(wi) > 1 else 0.0,
            "max_daily_delta": float(np.max(np.abs(np.diff(wi)))) if len(wi) > 1 else 0.0,
        }
    else:
        out["weights"] = {"missing": str(weights_path)}

    if returns_path.exists():
        r = pd.read_csv(returns_path)
        m = r["market_return"].to_numpy(dtype=float)
        i = r["ipo_return"].to_numpy(dtype=float)
        ws = np.linspace(0.0, 1.0, 1001)
        port = np.outer(1.0 - ws, m) + np.outer(ws, i)
        mean = np.mean(port, axis=1)
        vol = np.std(port, axis=1)
        sharpe = np.where(vol > 1e-12, mean / vol * np.sqrt(252.0), 0.0)
        total = np.prod(1.0 + port, axis=1) - 1.0
        out["returns"] = {
            "n_rows": int(len(r)),
            "market_mean": float(np.mean(m)),
            "ipo_mean": float(np.mean(i)),
            "market_std": float(np.std(m)),
            "ipo_std": float(np.std(i)),
            "corr_market_ipo": float(np.corrcoef(m, i)[0, 1]),
            "best_static_ipo_weight_by_sharpe": float(ws[int(np.argmax(sharpe))]),
            "best_static_ipo_weight_by_total_return": float(ws[int(np.argmax(total))]),
            "best_static_ipo_weight_by_mean_return": float(ws[int(np.argmax(mean))]),
        }
    else:
        out["returns"] = {"missing": str(returns_path)}
    return out


def make_predictive_dataset(
    *,
    seed: int = 7,
    n_windows: int = 1200,
    window_len: int = 20,
) -> tuple[dict, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = n_windows + window_len + 2

    state = np.zeros(total, dtype=np.float64)
    for t in range(1, total):
        state[t] = 0.87 * state[t - 1] + 0.55 * rng.standard_normal()

    signal = np.tanh(state)
    market = 0.00035 + 0.006 * rng.standard_normal(total)
    excess = np.zeros(total, dtype=np.float64)
    excess[1:] = 0.0028 * signal[:-1] + 0.0022 * rng.standard_normal(total - 1)
    ipo = market + excess

    feat = np.column_stack(
        [
            market,
            ipo,
            signal,
            np.r_[0.0, excess[:-1]],
            np.r_[0.0, market[:-1]],
        ]
    ).astype(np.float32)

    X, R, sig = [], [], []
    for t in range(window_len, window_len + n_windows):
        X.append(feat[t - window_len : t])
        R.append([market[t], ipo[t]])
        sig.append(signal[t - 1])
    X = np.asarray(X, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    sig = np.asarray(sig, dtype=np.float32)

    cut = int(0.7 * n_windows)
    data = {
        "X_train": X[:cut],
        "R_train": R[:cut],
        "dates_train": np.arange(cut),
        "X_val": X[cut:],
        "R_val": R[cut:],
        "dates_val": np.arange(n_windows - cut),
        "feature_cols": [],
        "df": None,
        "n_assets": 2,
        "window_len": window_len,
    }
    return data, sig[cut:], R[cut:]


def run_synthetic_experiment(
    data: dict,
    signal_val: np.ndarray,
    returns_val: np.ndarray,
    *,
    name: str,
    epochs: int,
    seed: int,
    loss_kw: dict,
) -> dict:
    torch.manual_seed(seed)
    model, history = run_training(
        data,
        device=torch.device("cpu"),
        epochs=epochs,
        lr=2e-3,
        batch_size=64,
        patience=max(4, epochs // 2),
        model_type="gru",
        hidden_size=32,
        verbose=False,
        log_every=0,
        weight_decay=1e-5,
        dropout=0.0,
        **loss_kw,
    )
    w = predict_weights(model, data["X_val"], torch.device("cpu"))
    ipo_w = w[:, 1]
    model_ret = np.sum(w * returns_val, axis=1)
    eq_ret = 0.5 * returns_val[:, 0] + 0.5 * returns_val[:, 1]
    return {
        "name": name,
        "epochs_ran": int(len(history)),
        "ipo_weight_mean": float(np.mean(ipo_w)),
        "ipo_weight_std": float(np.std(ipo_w)),
        "corr_weight_signal": float(np.corrcoef(ipo_w, signal_val)[0, 1]),
        "mean_return_model": float(np.mean(model_ret)),
        "mean_return_equal_weight": float(np.mean(eq_ret)),
        "total_return_model": float(np.prod(1.0 + model_ret) - 1.0),
        "total_return_equal_weight": float(np.prod(1.0 + eq_ret) - 1.0),
        "config": {k: float(v) for k, v in loss_kw.items() if isinstance(v, (int, float))},
    }


def choose_recommended_config(experiments: list[dict]) -> dict:
    ranked = sorted(
        experiments,
        key=lambda e: (
            float(e["mean_return_model"]),
            float(e["ipo_weight_std"]),
            float(e["corr_weight_signal"]),
        ),
        reverse=True,
    )
    best = ranked[0]
    return {
        "model_type": "gru",
        "lambda_cvar": best["config"].get("lambda_cvar", 0.5),
        "lambda_vol": best["config"].get("lambda_vol", 0.5),
        "lambda_turnover": best["config"].get("lambda_turnover", 0.0),
        "lambda_path": best["config"].get("lambda_path", 0.0),
        "lambda_vol_excess": best["config"].get("lambda_vol_excess", 1.0),
        "target_vol_annual": best["config"].get("target_vol_annual", 0.25),
        "mean_return_weight": best["config"].get("mean_return_weight", 1.0),
        "log_growth_weight": best["config"].get("log_growth_weight", 0.0),
        "_chosen_from_experiment": best["name"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose near-equal model weights and suggest tuning.")
    p.add_argument(
        "--weights",
        default=ROOT / "results" / "ipo_optimizer_weights.csv",
        type=Path,
        help="Path to exported model weights CSV.",
    )
    p.add_argument(
        "--returns",
        default=ROOT / "results" / "ipo_optimizer_returns_val.csv",
        type=Path,
        help="Path to exported validation returns CSV.",
    )
    p.add_argument(
        "--out-json",
        default=ROOT / "results" / "weight_diagnostic_report.json",
        type=Path,
        help="Where to write the full diagnostic report JSON.",
    )
    p.add_argument(
        "--out-config",
        default=ROOT / "results" / "ipo_optimizer_recommended_config.json",
        type=Path,
        help="Where to write a recommended config override JSON.",
    )
    p.add_argument("--epochs", type=int, default=12, help="Synthetic ablation training epochs.")
    p.add_argument("--n-windows", type=int, default=1200, help="Synthetic sample size.")
    args = p.parse_args()

    existing = analyze_existing_outputs(args.weights, args.returns)
    data, signal_val, returns_val = make_predictive_dataset(
        seed=7, n_windows=int(args.n_windows), window_len=20
    )

    configs = [
        (
            "baseline_defaults",
            dict(
                lambda_cvar=0.5,
                lambda_vol=0.5,
                lambda_turnover=0.01,
                lambda_path=0.01,
                lambda_diversify=0.0,
                lambda_vol_excess=1.0,
                target_vol_annual=0.25,
                mean_return_weight=1.0,
                log_growth_weight=0.0,
            ),
        ),
        (
            "no_turnover_path",
            dict(
                lambda_cvar=0.5,
                lambda_vol=0.5,
                lambda_turnover=0.0,
                lambda_path=0.0,
                lambda_diversify=0.0,
                lambda_vol_excess=1.0,
                target_vol_annual=0.25,
                mean_return_weight=1.0,
                log_growth_weight=0.0,
            ),
        ),
        (
            "return_focused",
            dict(
                lambda_cvar=0.3,
                lambda_vol=0.3,
                lambda_turnover=0.0,
                lambda_path=0.0,
                lambda_diversify=0.0,
                lambda_vol_excess=0.5,
                target_vol_annual=0.25,
                mean_return_weight=2.5,
                log_growth_weight=0.5,
            ),
        ),
    ]

    experiments = [
        run_synthetic_experiment(
            data,
            signal_val,
            returns_val,
            name=name,
            epochs=int(args.epochs),
            seed=7,
            loss_kw=kw,
        )
        for name, kw in configs
    ]

    report = {
        "existing_output_analysis": existing,
        "synthetic_ablation": experiments,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    recommended = choose_recommended_config(experiments)
    with open(args.out_config, "w", encoding="utf-8") as f:
        json.dump(recommended, f, indent=2)

    print(f"Wrote diagnostic report: {args.out_json}")
    print(f"Wrote recommended config: {args.out_config}")
    for ex in experiments:
        print(
            f"[{ex['name']}] ipo_std={ex['ipo_weight_std']:.6f}  "
            f"corr={ex['corr_weight_signal']:.3f}  "
            f"mean_ret={ex['mean_return_model']:.6e}  "
            f"eq_mean={ex['mean_return_equal_weight']:.6e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
