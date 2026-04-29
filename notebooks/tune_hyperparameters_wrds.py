#!/usr/bin/env python3
"""
Tune hyperparameters for IPO Portfolio Optimizer.

Uses **sector heads** when ``IPO_SECTOR_PORTFOLIOS`` / ``SECTOR_PORTFOLIOS`` enables them (same as
``run_ipo_optimizer_wrds.py``): shared encoder, one market-vs-sector-IPO softmax per sector.

Data: WRDS IPO + market panel from DATA_START through DATA_END (inclusive).
Rolling-window rows use train_val_test_split with an embargo of ``window_len`` trading
days so train / validation / test sets have **no overlapping calendar input spans**
across split boundaries (see src.data_layer.train_val_test_split).

Grid search over key hyperparameters; **select config by minimum validation loss** (same combined loss
and definition as ``run_training`` / checkpointing). Test-set Sharpe is reported only, not used for selection.
Uses ``lr_schedule=plateau`` by default (``ReduceLROnPlateau``): learning rate is reduced by factor
``lr_decay`` when validation loss does not improve for ``plateau_patience`` epochs.
Saves best config to ``results/recent/ipo_optimizer_best_config.json``, rolling train/val/test loss plot to
``results/recent/ipo_optimizer_tune_loss_train_val_test.png`` (and a copy under
``figures/recent/old_diagrams/``), and ``results/recent/ipo_optimizer_tune_best_history.json``.

**Environment (optional)**

- ``IPO_TUNE_DATA_START``, ``IPO_TUNE_DATA_END`` — sample range (default: from ``run_ipo_optimizer_wrds``).
- ``IPO_TUNE_VAL_START``, ``IPO_TUNE_TEST_START`` — split anchors for ``train_val_test_split``.
- ``IPO_TUNE_QUICK=1`` — only the **first** grid configuration (for fast smoke runs).
- ``IPO_TUNE_SELECTION`` — ``val_loss`` (default) or ``balanced``. **balanced** picks the config that
  minimizes ``val_loss + IPO_TUNE_BALANCE_WEIGHT * imbalance_cv``, where ``imbalance_cv`` is the
  coefficient of variation of **absolute λ-weighted** loss terms at the best-val epoch (penalizes one
  term dominating the total loss).
- ``IPO_TUNE_BALANCE_WEIGHT`` — non-negative; only used when ``IPO_TUNE_SELECTION=balanced`` (try ``0.05``–``0.2``).
"""
from __future__ import annotations

import os
import shutil
import sys
import json
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
RESULTS_DIR = ROOT / "results" / "recent"
FIGURES_DIR = ROOT / "figures" / "recent"

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import numpy as np
import torch

from run_ipo_optimizer_wrds import (
    get_connection,
    prepare_data,
    sector_portfolios_effective,
    START_DATE as DATA_START,
    END_DATE as DATA_END,
    VAL_START,
    TEST_START,
)
from src.data_layer import build_rolling_windows, build_rolling_windows_sector_heads, train_val_test_split
from src.train import run_training, run_training_sector_heads, validate, validate_sector_heads
from src.export import predict_weights, predict_sector_head_weights, portfolio_stats
from src.plot_loss import plot_train_val_rolling_and_test, slim_history_for_json
from src.loss_balance import balance_metrics_for_config, composite_tune_score
from src.wrds_data import close_wrds_connection

# VAL_START / TEST_START: run_ipo_optimizer_wrds (embargo uses window_len from each config).

# Hyperparameter grid (all configurations; 2×2×4×4×2×2 = 256 configs)
GRID = {
    # Training mechanics fixed — already validated by manual runs
    "window_len": [84],
    "lr": [3e-4],
    # Dynamic step sizes: plateau reduces LR when val loss stalls (lr_decay = new_lr / old_lr)
    "lr_schedule": ["plateau"],
    "lr_decay": [0.1],
    "plateau_patience": [4],
    "batch_size": [256],
    "hidden_size": [64],
    "patience": [50],
    "epochs": [50],
    # Loss hyperparameters — the actual search space
    "lambda_vol": [0.5, 1.0],
    "lambda_cvar": [0.5, 1.0],
    "lambda_turnover": [0.0, 0.0001, 0.0005, 0.001],
    "lambda_path": [0.0, 0.0001, 0.0005, 0.001],
    "lambda_vol_excess": [0.5, 1.0],
    "target_vol_annual": [0.20, 0.25],
}


def _json_safe(obj):
    """Recursively convert numpy scalars for JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def _stats_sector_heads(weights: np.ndarray, R: np.ndarray, sector_labels: list[str]) -> dict:
    """
    Per-sector ``portfolio_stats``; ``sharpe_annualized`` is the **mean Sharpe** across sector
    sleeves (comparable scalar to the 2-asset run for logging).
    """
    g = weights.shape[1]
    per_sector = []
    sharps = []
    for idx in range(g):
        s = portfolio_stats(weights[:, idx, :], R[:, idx, :])
        label = sector_labels[idx] if idx < len(sector_labels) else str(idx)
        per_sector.append({"sector": label, **s})
        sharps.append(s["sharpe_annualized"])
    mean_sh = float(np.mean(sharps)) if sharps else 0.0
    return {
        "sharpe_annualized": mean_sh,
        "mean_sharpe_annualized": mean_sh,
        "per_sector": per_sector,
    }


def run_config(data_prep, config, device, *, val_start: str, test_start: str):
    """Train with given config; return (best_val_loss, stats, history). Lower loss is better."""
    df = data_prep["df"]
    feature_cols = data_prep["feature_cols"]
    window_len = config["window_len"]
    use_sectors = bool(data_prep.get("sector_portfolios"))
    sector_ret_cols = data_prep.get("sector_ret_cols") or []
    sector_labels = data_prep.get("sector_labels") or []

    if use_sectors:
        X, R, dates = build_rolling_windows_sector_heads(
            df,
            window_len=window_len,
            feature_cols=feature_cols,
            sector_ret_cols=sector_ret_cols,
        )
    else:
        X, R, dates = build_rolling_windows(df, window_len=window_len, feature_cols=feature_cols)
    if X.shape[0] < 50:
        return float("inf"), {}, []

    X_train, R_train, d_train, X_val, R_val, d_val, X_test, R_test, d_test = train_val_test_split(
        X,
        R,
        dates,
        val_start=val_start,
        test_start=test_start,
        df_index=df.index,
        window_len=window_len,
    )
    if X_train.shape[0] < 50 or X_val.shape[0] < 10 or X_test.shape[0] < 10:
        return float("inf"), {}, []

    data = {
        "X_train": X_train,
        "R_train": R_train,
        "dates_train": d_train,
        "X_val": X_val,
        "R_val": R_val,
        "dates_val": d_val,
        "feature_cols": feature_cols,
        "df": df,
        "window_len": window_len,
    }
    if use_sectors:
        data["n_sectors"] = len(sector_labels)
    else:
        data["n_assets"] = 2

    train_kw = dict(
        device=device,
        epochs=config["epochs"],
        lr=config["lr"],
        batch_size=config["batch_size"],
        patience=config["patience"],
        lambda_cvar=config["lambda_cvar"],
        lambda_turnover=config.get("lambda_turnover", 0.01),
        lambda_vol=config["lambda_vol"],
        lambda_path=config.get("lambda_path", 0.01),
        lambda_vol_excess=config.get("lambda_vol_excess", 1.0),
        target_vol_annual=config.get("target_vol_annual", 0.25),
        lambda_diversify=0.0,
        min_weight=0.1,
        hidden_size=config["hidden_size"],
        num_layers=1,
        model_type="gru",
        verbose=True,
        log_every=5,
        lr_schedule=config.get("lr_schedule", "constant"),
        lr_decay=config.get("lr_decay", 0.5),
        plateau_patience=config.get("plateau_patience", 4),
        min_lr=config.get("min_lr", 1e-6),
        exponential_gamma=config.get("exponential_gamma", 0.99),
        mean_return_weight=config.get("mean_return_weight", 1.0),
        log_growth_weight=config.get("log_growth_weight", 0.0),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        grad_norm_mode=str(config.get("grad_norm_mode", "clip")),
    )
    if use_sectors:
        model, history = run_training_sector_heads(data, **train_kw)
    else:
        model, history = run_training(data, **train_kw)

    best_val_loss = min((row["val_loss"] for row in history), default=float("inf"))

    val_kw = dict(
        batch_size=config["batch_size"],
        lambda_cvar=config["lambda_cvar"],
        lambda_turnover=config.get("lambda_turnover", 0.01),
        lambda_vol=config["lambda_vol"],
        lambda_path=config.get("lambda_path", 0.01),
        lambda_diversify=0.0,
        min_weight=0.1,
        lambda_vol_excess=config.get("lambda_vol_excess", 1.0),
        target_vol_annual=config.get("target_vol_annual", 0.25),
        mean_return_weight=config.get("mean_return_weight", 1.0),
        log_growth_weight=config.get("log_growth_weight", 0.0),
    )
    if use_sectors:
        test_loss_eval, _ = validate_sector_heads(model, X_test, R_test, device, **val_kw)
    else:
        test_loss_eval, _ = validate(model, X_test, R_test, device, **val_kw)
    if use_sectors:
        weights_val = predict_sector_head_weights(model, data["X_val"], device)
        stats_val = _stats_sector_heads(weights_val, data["R_val"], sector_labels)
        weights_test = predict_sector_head_weights(model, X_test, device)
        stats_test = _stats_sector_heads(weights_test, R_test, sector_labels)
    else:
        weights_val = predict_weights(model, data["X_val"], device)
        stats_val = portfolio_stats(weights_val, data["R_val"])
        weights_test = predict_weights(model, X_test, device)
        stats_test = portfolio_stats(weights_test, R_test)
    stats = {
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss_eval),
        "sector_heads": use_sectors,
        "n_sectors": int(len(sector_labels)) if use_sectors else 0,
        "val": stats_val,
        "test": stats_test,
        "n_train_windows": int(X_train.shape[0]),
        "n_val_windows": int(X_val.shape[0]),
        "n_test_windows": int(X_test.shape[0]),
    }
    if history:
        stats["loss_balance"] = balance_metrics_for_config(config, history)
    else:
        stats["loss_balance"] = {}
    return best_val_loss, stats, history


def main():
    data_start = os.environ.get("IPO_TUNE_DATA_START", DATA_START)
    data_end = os.environ.get("IPO_TUNE_DATA_END", DATA_END)
    val_start = os.environ.get("IPO_TUNE_VAL_START", VAL_START)
    test_start = os.environ.get("IPO_TUNE_TEST_START", TEST_START)

    print("Connecting to WRDS...", flush=True)
    conn = get_connection()
    print("Preparing data...", flush=True)
    try:
        sp = sector_portfolios_effective()
        data_prep = prepare_data(conn, start=data_start, end=data_end, sector_portfolios=sp)
    finally:
        close_wrds_connection(conn)
    print(
        f"sector_portfolios={sp}  |  Data range {data_start}–{data_end}; splits use "
        f"window_len={GRID['window_len'][0]}-day embargo (train/val/test label windows do not share "
        f"input rows across boundaries; see train_val_test_split). "
        f"Anchors: val_start={val_start}, test_start={test_start}.",
        flush=True,
    )

    keys = list(GRID.keys())
    full_configs = [
        dict(zip(keys, vals))
        for vals in product(*(GRID[k] for k in keys))
    ]
    quick = os.environ.get("IPO_TUNE_QUICK", "").strip().lower() in ("1", "true", "yes", "on")
    if quick:
        configs = [full_configs[0]]
        print(f"IPO_TUNE_QUICK=1: using 1 config (of {len(full_configs)}).", flush=True)
    else:
        configs = full_configs
    print(f"Tuning over {len(configs)} configurations...", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    balance_w = float(os.environ.get("IPO_TUNE_BALANCE_WEIGHT", "0").strip() or "0")
    selection = os.environ.get("IPO_TUNE_SELECTION", "val_loss").strip().lower()
    if selection not in ("val_loss", "balanced"):
        raise ValueError("IPO_TUNE_SELECTION must be val_loss or balanced")

    best_val_loss = float("inf")
    best_score = float("inf")
    best_config = None
    best_stats = {}
    best_history: list = []
    results = []
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "ipo_optimizer_best_config.json"

    def _save_best():
        """Save best config so far (enables use of partial results if stopped early)."""
        if best_config is None:
            return
        out = {
            "data_split": {
                "data_start": data_start,
                "data_end": data_end,
                "val_start": val_start,
                "test_start": test_start,
            },
            "selection": selection,
            "balance_weight": balance_w,
            "best_val_loss": best_val_loss,
            "best_tune_score": best_score,
            "best_config": best_config,
            "best_stats": _json_safe(best_stats),
            "best_history": slim_history_for_json(best_history) if best_history else [],
            "all_results": [
                {
                    "config": r["config"],
                    "val_loss": r["val_loss"],
                    "tune_score": r.get("tune_score"),
                    "stats": _json_safe(r.get("stats", {})),
                }
                for r in results
            ],
        }
        with open(config_path, "w") as f:
            json.dump(out, f, indent=2)

    for i, config in enumerate(configs):
        try:
            val_loss, stats, history = run_config(
                data_prep, config, device, val_start=val_start, test_start=test_start
            )
            lb = stats.get("loss_balance") or {}
            imb = float(lb.get("imbalance_cv", 0.0))
            tune_score = composite_tune_score(val_loss, imb, balance_weight=balance_w)
            stats["tune_score"] = tune_score
            compare = tune_score if selection == "balanced" else val_loss
            results.append({"config": config, "val_loss": val_loss, "tune_score": tune_score, "stats": stats})
            if compare < best_score:
                best_score = compare
                best_val_loss = val_loss
                best_config = config.copy()
                best_stats = stats.copy()
                best_history = history
                _save_best()
                if selection == "balanced":
                    print(
                        f"  [{i+1}/{len(configs)}] New best: tune_score={tune_score:.6f} "
                        f"(val_loss={val_loss:.6f}, imbalance_cv={imb:.4f}, w={balance_w}) | "
                        f"val_Sharpe={stats['val']['sharpe_annualized']:.2f} | {config}",
                        flush=True,
                    )
                else:
                    print(
                        f"  [{i+1}/{len(configs)}] New best: val_loss={val_loss:.6f} | "
                        f"val_Sharpe={stats['val']['sharpe_annualized']:.2f} | {config}",
                        flush=True,
                    )
            elif (i + 1) % 10 == 0:
                print(
                    f"  [{i+1}/{len(configs)}] val_loss={val_loss:.6f}  "
                    f"val_Sharpe={stats['val']['sharpe_annualized']:.2f}",
                    flush=True,
                )
        except Exception as e:
            print(f"  [{i+1}/{len(configs)}] FAILED: {e}", flush=True)
            results.append(
                {"config": config, "val_loss": float("inf"), "tune_score": float("inf"), "error": str(e)}
            )

    fig_path = out_dir / "ipo_optimizer_tune_loss_train_val_test.png"
    if best_history:
        tl = best_stats.get("test_loss")
        plot_train_val_rolling_and_test(
            best_history,
            fig_path,
            test_loss=float(tl) if tl is not None else None,
            rolling_epochs=3,
            title=f"Train / val loss (rolling) + test — {data_start} to {data_end}",
        )
        print(f"Saved loss figure to {fig_path}", flush=True)
        fig_dir = FIGURES_DIR / "old_diagrams"
        fig_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(fig_path, fig_dir / fig_path.name)
        print(f"Copied to {fig_dir / fig_path.name}", flush=True)
        hist_json = out_dir / "ipo_optimizer_tune_best_history.json"
        with open(hist_json, "w", encoding="utf-8") as f:
            json.dump(slim_history_for_json(best_history), f, indent=2)
        print(f"Saved best run history to {hist_json}", flush=True)

    best_out = {
        "selection": selection,
        "balance_weight": balance_w,
        "primary_metric": "tune_score" if selection == "balanced" else "val_loss",
        "data_split": {
            "data_start": data_start,
            "data_end": data_end,
            "val_start": val_start,
            "test_start": test_start,
        },
        "best_config": best_config,
        "best_val_loss": best_val_loss,
        "best_tune_score": best_score,
        "best_history": slim_history_for_json(best_history) if best_history else [],
        "best_stats": _json_safe(best_stats),
        "all_results": [
            {
                "config": r["config"],
                "val_loss": r["val_loss"],
                "tune_score": r.get("tune_score"),
                "stats": _json_safe(r.get("stats", {})),
            }
            for r in results
        ],
    }
    config_path = out_dir / "ipo_optimizer_best_config.json"
    with open(config_path, "w") as f:
        json.dump(best_out, f, indent=2)

    print("\n" + "=" * 50)
    print("Best config:", json.dumps(best_config, indent=2))
    print(f"Best validation loss: {best_val_loss:.6f}")
    if selection == "balanced":
        print(f"Best tune_score (val_loss + {balance_w} * imbalance_cv): {best_score:.6f}")
        if best_stats.get("loss_balance", {}).get("abs_contributions"):
            print("  Abs λ-weighted contributions at best-val epoch:", best_stats["loss_balance"]["abs_contributions"])
    if best_stats.get("val"):
        vs = best_stats["val"].get("sharpe_annualized", float("nan"))
        print(f"Validation Sharpe (reporting only): {vs:.2f}")
    if best_stats.get("test"):
        ts = best_stats["test"].get("sharpe_annualized", float("nan"))
        print(f"Test Sharpe (held-out, reporting only): {ts:.2f}")
    print(f"Saved to {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
