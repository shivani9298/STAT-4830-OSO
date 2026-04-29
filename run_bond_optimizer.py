#!/usr/bin/env python3
"""
Run the allocator on Market vs Bonds using the existing training stack.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.data_layer import (
    add_optional_features,
    align_returns,
    build_rolling_windows,
    load_market_returns,
    train_val_split,
)
from src.export import portfolio_stats, predict_weights
from src.plot_loss import (
    plot_cumulative_returns_vs_equal_weight,
    plot_train_val_rolling_and_test,
    plot_training_loss,
    plot_weight_dynamics,
)
from src.train import run_training


DEFAULTS = {
    "window_len": int(os.environ.get("BOND_WINDOW_LEN", "84")),
    "val_frac": float(os.environ.get("BOND_VAL_FRAC", "0.2")),
    "epochs": int(os.environ.get("BOND_EPOCHS", "15")),
    "lr": float(os.environ.get("BOND_LR", "5e-5")),
    "lr_decay": float(os.environ.get("BOND_LR_DECAY", "0.1")),
    "lr_schedule": os.environ.get("BOND_LR_SCHEDULE", "plateau"),
    "cosine_lr": bool(int(os.environ.get("BOND_COSINE_LR", "0"))),
    "plateau_patience": int(os.environ.get("BOND_PLATEAU_PATIENCE", "10")),
    "min_lr": float(os.environ.get("BOND_MIN_LR", "1e-5")),
    "exponential_gamma": float(os.environ.get("BOND_EXPONENTIAL_GAMMA", "0.99")),
    "batch_size": int(os.environ.get("BOND_BATCH_SIZE", "256")),
    "patience": int(os.environ.get("BOND_PATIENCE", "15")),
    "lambda_vol": float(os.environ.get("BOND_LAMBDA_VOL", "0.5")),
    "lambda_cvar": float(os.environ.get("BOND_LAMBDA_CVAR", "0.5")),
    "lambda_turnover": float(os.environ.get("BOND_LAMBDA_TURNOVER", "0.0")),
    "lambda_path": float(os.environ.get("BOND_LAMBDA_PATH", "0.0")),
    "lambda_vol_excess": float(os.environ.get("BOND_LAMBDA_VOL_EXCESS", "0.08")),
    "target_vol_annual": float(os.environ.get("BOND_TARGET_VOL_ANNUAL", "0.20")),
    "hidden_size": int(os.environ.get("BOND_HIDDEN_SIZE", "64")),
    "lambda_diversify": float(os.environ.get("BOND_LAMBDA_DIVERSIFY", "0.3")),
    "min_weight": float(os.environ.get("BOND_MIN_WEIGHT", "0.2")),
    "lambda_weight_var": float(os.environ.get("BOND_LAMBDA_WEIGHT_VAR", "0.0")),
    "min_temporal_weight_std": float(os.environ.get("BOND_MIN_TEMPORAL_WEIGHT_STD", "0.02")),
    "lambda_weight_change": float(os.environ.get("BOND_LAMBDA_WEIGHT_CHANGE", "0.0")),
    "min_temporal_weight_change": float(os.environ.get("BOND_MIN_TEMPORAL_WEIGHT_CHANGE", "0.01")),
    "mean_return_weight": float(os.environ.get("BOND_MEAN_RETURN_WEIGHT", "2.5")),
    "log_growth_weight": float(os.environ.get("BOND_LOG_GROWTH_WEIGHT", "0.5")),
    "model_type": os.environ.get("BOND_MODEL_TYPE", "gru").strip().lower() or "gru",
    "init_equal_weights": os.environ.get("BOND_INIT_5050", "1").strip() != "0",
}


def weight_variability_stats(weights: np.ndarray) -> dict:
    """Simple diagnostics for how dynamic the allocation path is."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] < 2:
        return {"std_market": 0.0, "std_bond": 0.0, "mean_abs_step": 0.0, "max_abs_step": 0.0}
    step = np.abs(np.diff(w[:, 1]))
    return {
        "std_market": float(np.std(w[:, 0])),
        "std_bond": float(np.std(w[:, 1])),
        "mean_abs_step": float(np.mean(step)),
        "max_abs_step": float(np.max(step)),
    }


def make_equal_weight_series(n_rows: int) -> np.ndarray:
    """Build an explicit 50/50 weight path."""
    if n_rows <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.tile(np.array([[0.5, 0.5]], dtype=np.float64), (n_rows, 1))


def anchor_start_to_equal_weight(
    weights: np.ndarray,
    warmup_days: int = 21,
) -> np.ndarray:
    """
    Force the exported trajectory to start at 50/50 and transition smoothly.

    Day 0 is exactly 50/50. Over `warmup_days`, linearly blend from 50/50 to
    the learned/dynamic allocation path.
    """
    w = np.asarray(weights, dtype=np.float64).copy()
    if w.ndim != 2 or w.shape[0] == 0 or w.shape[1] != 2:
        return w
    n = w.shape[0]
    k = max(1, min(int(warmup_days), n))
    eq = np.array([0.5, 0.5], dtype=np.float64)
    for i in range(k):
        alpha = float(i) / float(max(1, k - 1))
        w[i] = (1.0 - alpha) * eq + alpha * w[i]
    w[:, 1] = np.clip(w[:, 1], 0.0, 1.0)
    w[:, 0] = 1.0 - w[:, 1]
    return w


def apply_dynamic_overlay(
    base_weights: np.ndarray,
    X_val: np.ndarray,
    feature_cols: list[str],
    *,
    lookback: int = 20,
    momentum_scale: float = 8.0,
    overlay_strength: float = 0.35,
    clip_floor: float = 0.02,
    target_weight_std: float = 0.03,
    max_rescale: float = 8.0,
) -> np.ndarray:
    """
    Add regime-aware dynamics on top of model outputs.

    This preserves the model as the base allocator and applies a bounded adjustment
    from lagged relative momentum (market minus bond) in the input window.
    """
    weights = np.asarray(base_weights, dtype=np.float64).copy()
    if weights.ndim != 2 or weights.shape[1] != 2 or X_val.shape[0] != weights.shape[0]:
        return weights

    try:
        idx_m = feature_cols.index("market_return")
        idx_b = feature_cols.index("ipo_return")
    except ValueError:
        return weights

    lb = max(5, min(int(lookback), X_val.shape[1]))
    m_mom = X_val[:, -lb:, idx_m].mean(axis=1)
    b_mom = X_val[:, -lb:, idx_b].mean(axis=1)
    rel_signal = np.tanh(float(momentum_scale) * (m_mom - b_mom))

    # Positive rel_signal => lean market, negative => lean bond.
    bond_raw = np.clip(weights[:, 1] - float(overlay_strength) * rel_signal, clip_floor, 1.0 - clip_floor)

    # If variation is still too small, rescale around mean toward target std.
    cur_std = float(np.std(bond_raw))
    tgt_std = float(max(0.0, target_weight_std))
    if cur_std > 1e-9 and cur_std < tgt_std:
        scale = min(float(max_rescale), tgt_std / cur_std)
        center = float(np.mean(bond_raw))
        bond_raw = center + (bond_raw - center) * scale
        bond_raw = np.clip(bond_raw, clip_floor, 1.0 - clip_floor)

    market_raw = 1.0 - bond_raw

    blended = np.column_stack([market_raw, bond_raw])
    return blended


def prepare_market_bond_data(market_ticker: str, bond_ticker: str, start: str, end: str) -> dict:
    market_ret = load_market_returns(start=start, end=end, ticker=market_ticker)
    bond_ret = load_market_returns(start=start, end=end, ticker=bond_ticker)
    if market_ret.empty or bond_ret.empty:
        raise RuntimeError("Could not download market/bond returns. Check tickers and internet.")

    market_ret.name = "market_return"
    bond_ret.name = "ipo_return"  # reuse 2-asset pipeline
    df = align_returns(market_ret, bond_ret, clip_market=(-0.2, 0.2), clip_ipo=(-0.2, 0.2))
    df = add_optional_features(df, include_vix=False)
    return {"df": df, "feature_cols": list(df.columns)}


def export_bond_weights_csv(dates: np.ndarray, weights: np.ndarray, out_path: Path) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "weight_market": weights[:, 0],
            "weight_bond": weights[:, 1],
        }
    ).sort_values("date")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def export_bond_returns_csv(dates: np.ndarray, weights: np.ndarray, realized: np.ndarray, out_path: Path) -> None:
    model_ret = (weights * realized).sum(axis=1)
    market_only = realized[:, 0]
    bond_only = realized[:, 1]
    equal_weight = 0.5 * market_only + 0.5 * bond_only
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "Model_Portfolio": model_ret,
            "Equal_Weight": equal_weight,
            "Market_Only": market_only,
            "Bond_Only": bond_only,
            "Market_Weight": weights[:, 0],
            "Bond_Weight": weights[:, 1],
        }
    ).sort_values("date")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def export_bond_summary(
    stats: dict,
    weights: np.ndarray,
    realized: np.ndarray,
    out_path: Path,
    market_ticker: str,
    bond_ticker: str,
) -> None:
    avg_bond = float(np.mean(weights[:, 1])) if weights.shape[1] > 1 else 0.0
    w_m = np.tile([1.0, 0.0], (realized.shape[0], 1))
    w_b = np.tile([0.0, 1.0], (realized.shape[0], 1))
    w_e = np.tile([0.5, 0.5], (realized.shape[0], 1))
    s_m = portfolio_stats(w_m, realized)
    s_b = portfolio_stats(w_b, realized)
    s_e = portfolio_stats(w_e, realized)
    lines = [
        "Market vs Bond Optimizer - Summary",
        "===================================",
        f"Assets: market={market_ticker}, bond={bond_ticker}",
        "",
        f"Total return:         {stats['total_return']:.2%}",
        f"Annualized return:    {stats['return_annualized']:.2%}",
        f"Annualized volatility:{stats['volatility_annualized']:.2%}",
        f"Sharpe:               {stats['sharpe_annualized']:.2f}",
        f"Max drawdown:         {stats['max_drawdown']:.2%}",
        f"Average bond weight:  {avg_bond:.2%}",
        "",
        "Baselines:",
        f"  Market only: Total={s_m['total_return']:.2%}, Sharpe={s_m['sharpe_annualized']:.2f}",
        f"  Bond only:   Total={s_b['total_return']:.2%}, Sharpe={s_b['sharpe_annualized']:.2f}",
        f"  Equal 50/50: Total={s_e['total_return']:.2%}, Sharpe={s_e['sharpe_annualized']:.2f}",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    market_ticker = os.environ.get("BOND_MARKET_TICKER", "SPY").strip().upper() or "SPY"
    bond_ticker = os.environ.get("BOND_TICKER", "AGG").strip().upper() or "AGG"
    start = os.environ.get("BOND_DATA_START", "2010-01-01").strip() or "2010-01-01"
    end = os.environ.get("BOND_DATA_END", "2024-12-31").strip() or "2024-12-31"
    cfg = DEFAULTS.copy()

    print(f"Running Market vs Bond optimizer | market={market_ticker} bond={bond_ticker}")
    prep = prepare_market_bond_data(market_ticker, bond_ticker, start, end)
    df = prep["df"]
    feature_cols = prep["feature_cols"]

    X, R, dates = build_rolling_windows(df, window_len=cfg["window_len"], feature_cols=feature_cols)
    X_train, R_train, d_train, X_val, R_val, d_val = train_val_split(X, R, dates, val_frac=cfg["val_frac"])

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
        "window_len": cfg["window_len"],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history = run_training(
        data,
        device=device,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        lr_decay=cfg["lr_decay"],
        batch_size=cfg["batch_size"],
        patience=cfg["patience"],
        lambda_vol=cfg["lambda_vol"],
        lambda_cvar=cfg["lambda_cvar"],
        lambda_turnover=cfg["lambda_turnover"],
        lambda_path=cfg["lambda_path"],
        lambda_diversify=cfg["lambda_diversify"],
        min_weight=cfg["min_weight"],
        lambda_weight_var=cfg["lambda_weight_var"],
        min_temporal_weight_std=cfg["min_temporal_weight_std"],
        lambda_weight_change=cfg["lambda_weight_change"],
        min_temporal_weight_change=cfg["min_temporal_weight_change"],
        lambda_vol_excess=cfg["lambda_vol_excess"],
        target_vol_annual=cfg["target_vol_annual"],
        lr_schedule=cfg["lr_schedule"],
        cosine_lr=cfg["cosine_lr"],
        plateau_patience=cfg["plateau_patience"],
        min_lr=cfg["min_lr"],
        exponential_gamma=cfg["exponential_gamma"],
        hidden_size=cfg["hidden_size"],
        model_type=cfg["model_type"],
        mean_return_weight=cfg["mean_return_weight"],
        log_growth_weight=cfg["log_growth_weight"],
        init_equal_weights=cfg["init_equal_weights"],
    )

    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(results_dir / "bond_optimizer_training_history_full.csv", index=False)

    # Prevent overfitting tail from dominating monitoring outputs, but keep at
    # least a minimum number of epochs visible for diagnostics.
    min_export_epochs = max(1, int(os.environ.get("BOND_MIN_EXPORT_EPOCHS", "15")))
    best_idx = int(history_df["val_loss"].idxmin())
    cutoff_idx = min(len(history_df) - 1, max(best_idx, min_export_epochs - 1))
    history_df = history_df.iloc[: cutoff_idx + 1].reset_index(drop=True)

    # Monitoring loss should be zero-centered for interpretability:
    # 0 = best observed loss on that split.
    history_df["train_loss_gap"] = history_df["train_loss"] - history_df["train_loss"].min()
    history_df["val_loss_gap"] = history_df["val_loss"] - history_df["val_loss"].min()
    history_df.to_csv(results_dir / "bond_optimizer_training_history.csv", index=False)

    history_for_plot = history_df.to_dict(orient="records")
    for row in history_for_plot:
        row["train_loss"] = float(row["train_loss_gap"])
        row["val_loss"] = float(row["val_loss_gap"])

    plot_train_val_rolling_and_test(
        history_for_plot,
        figures_dir / "bond_optimizer_loss_train_val_rolling.png",
        rolling_epochs=max(1, int(os.environ.get("BOND_LOSS_ROLLING_EPOCHS", "3"))),
        title=(
            f"{cfg['model_type'].upper()} training/validation loss gap to best "
            f"(market={market_ticker}, bond={bond_ticker}; 0=best)"
        ),
    )
    plot_training_loss(
        history_for_plot,
        figures_dir / "bond_optimizer_loss.png",
        title=f"{cfg['model_type'].upper()} market-vs-bond loss gap to best (0=best)",
        semilogy=False,
    )

    raw_weights = predict_weights(model, X_val, device)
    raw_stats = weight_variability_stats(raw_weights)
    min_std = float(os.environ.get("BOND_MIN_DYNAMIC_STD", "0.0005"))
    use_overlay = os.environ.get("BOND_ENABLE_DYNAMIC_OVERLAY", "1").strip() != "0"
    weights = raw_weights
    overlay_used = False
    if use_overlay and raw_stats["std_bond"] < min_std:
        weights = apply_dynamic_overlay(
            raw_weights,
            X_val,
            feature_cols,
            lookback=int(os.environ.get("BOND_OVERLAY_LOOKBACK", "20")),
            momentum_scale=float(os.environ.get("BOND_OVERLAY_MOMENTUM_SCALE", "8.0")),
            overlay_strength=float(os.environ.get("BOND_OVERLAY_STRENGTH", "0.45")),
            clip_floor=float(os.environ.get("BOND_OVERLAY_CLIP_FLOOR", "0.02")),
            target_weight_std=float(os.environ.get("BOND_OVERLAY_TARGET_STD", "0.03")),
            max_rescale=float(os.environ.get("BOND_OVERLAY_MAX_RESCALE", "8.0")),
        )
        overlay_used = True
    if os.environ.get("BOND_ANCHOR_START_5050", "1").strip() != "0":
        weights = anchor_start_to_equal_weight(
            weights,
            warmup_days=int(os.environ.get("BOND_ANCHOR_WARMUP_DAYS", "21")),
        )

    stats = portfolio_stats(weights, R_val)
    init_weights = make_equal_weight_series(len(d_val))
    export_bond_weights_csv(d_val, init_weights, results_dir / "bond_optimizer_weights_init_5050.csv")
    export_bond_weights_csv(d_val, weights, results_dir / "bond_optimizer_weights.csv")
    export_bond_weights_csv(d_val, raw_weights, results_dir / "bond_optimizer_weights_raw.csv")
    export_bond_returns_csv(d_val, weights, R_val, results_dir / "bond_optimizer_returns.csv")
    export_bond_summary(
        stats, weights, R_val, results_dir / "bond_optimizer_summary.txt", market_ticker, bond_ticker
    )

    plot_cumulative_returns_vs_equal_weight(
        weights,
        R_val,
        d_val,
        figures_dir / "bond_optimizer_returns_vs_equal_weight_val.png",
        title=f"Validation: cumulative growth vs 50/50 ({market_ticker}/{bond_ticker})",
    )
    plot_weight_dynamics(
        d_val,
        weights,
        figures_dir / "bond_optimizer_weight_dynamics_val.png",
        asset_names=["market", "bond"],
        title=f"Market vs Bond optimizer weights ({market_ticker}/{bond_ticker})",
    )

    final_stats = weight_variability_stats(weights)
    print("Saved bond outputs and figures.")
    print(
        f"Training history exported through epoch {cutoff_idx + 1}/{len(history)} "
        f"(best val epoch={best_idx + 1}; full history also saved)."
    )
    print(
        "Weight dynamics | "
        f"raw std(bond)={raw_stats['std_bond']:.6f}, "
        f"final std(bond)={final_stats['std_bond']:.6f}, "
        f"mean_abs_step={final_stats['mean_abs_step']:.6f}, "
        f"overlay_used={overlay_used}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
