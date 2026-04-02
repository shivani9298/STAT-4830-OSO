#!/usr/bin/env python3
"""
One-at-a-time Transformer ablations on a short WRDS date range (default 2020–2024).

Uses ``train_val_test_split`` with **embargo** so train / validation / test rolling
windows have **non-overlapping calendar input spans** at split boundaries (same logic
as ``tune_hyperparameters_wrds.py``).

Train optimizes on train; early stopping uses validation; **test** loss is reported
after each run with the best validation checkpoint.

Usage (from repo root, Anaconda Python recommended):

  set IPO_MODEL_TYPE=transformer
  python scripts/wrds_transformer_ablation_short_horizon.py

  # Single market vs IPO (faster, no sector baskets):
  python scripts/wrds_transformer_ablation_short_horizon.py --no-sector

  # Custom boundaries (must satisfy val_start < test_start):
  python scripts/wrds_transformer_ablation_short_horizon.py \\
    --start 2020-01-01 --end 2024-12-31 \\
    --val-start 2023-01-01 --test-start 2024-01-01
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import torch

from run_ipo_optimizer_wrds import (
    DEFAULTS,
    TRANSFORMER_CONFIG,
    close_wrds_connection,
    get_connection,
    prepare_data,
)
from src.data_layer import (
    build_rolling_windows,
    build_rolling_windows_sector_heads,
    train_val_test_split,
)
from src.train import run_training, run_training_sector_heads, validate, validate_sector_heads


def _base_training_cfg() -> dict:
    cfg = {**DEFAULTS, **TRANSFORMER_CONFIG, "model_type": "transformer"}
    cfg["epochs"] = 40
    cfg["patience"] = 10
    return cfg


def _compact_history(history: list[dict]) -> list[dict]:
    """Epoch, train, val only (keeps JSON small)."""
    return [
        {
            "epoch": int(h["epoch"]),
            "train_loss": float(h["train_loss"]),
            "val_loss": float(h["val_loss"]),
        }
        for h in history
    ]


def _training_kw(cfg: dict) -> dict:
    return {
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "batch_size": cfg["batch_size"],
        "patience": cfg["patience"],
        "lambda_vol": cfg["lambda_vol"],
        "lambda_cvar": cfg["lambda_cvar"],
        "lambda_diversify": cfg.get("lambda_diversify", 0.0),
        "min_weight": cfg.get("min_weight", 0.1),
        "lambda_vol_excess": cfg.get("lambda_vol_excess", 1.0),
        "target_vol_annual": cfg.get("target_vol_annual", 0.25),
        "hidden_size": cfg["hidden_size"],
        "num_layers": cfg.get("num_layers", 1),
        "model_type": cfg["model_type"],
        "verbose": False,
        "log_every": 0,
        "weight_decay": cfg.get("weight_decay", 1e-5),
        "dropout": cfg.get("dropout", 0.1),
        "cosine_lr": cfg.get("cosine_lr", False),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="WRDS Transformer ablations (short horizon, embargo splits)")
    p.add_argument("--start", default="2020-01-01", help="Data load start (inclusive)")
    p.add_argument("--end", default="2024-12-31", help="Data load end (inclusive)")
    p.add_argument(
        "--val-start",
        default="2023-01-01",
        help="First validation prediction date (embargo-aware)",
    )
    p.add_argument(
        "--test-start",
        default="2024-01-01",
        help="First test prediction date (embargo-aware)",
    )
    p.add_argument(
        "--no-sector",
        action="store_true",
        help="Single market vs IPO index (skip sector baskets)",
    )
    p.add_argument(
        "--output",
        default="results/wrds_transformer_ablation_2020_2024.json",
        help="JSON path for results (under repo root)",
    )
    args = p.parse_args()

    use_sectors = not args.no_sector

    base = _base_training_cfg()
    window_len = base["window_len"]

    experiments: list[tuple[str, dict]] = [
        ("baseline", {}),
        ("lr_1e-3", {"lr": 1e-3}),
        ("weight_decay_1e-4", {"weight_decay": 1e-4}),
        ("weight_decay_0", {"weight_decay": 0.0}),
        ("dropout_0.2", {"dropout": 0.2}),
        ("cosine_lr", {"cosine_lr": True}),
        ("batch_32", {"batch_size": 32}),
        ("batch_128", {"batch_size": 128}),
        ("hidden_64", {"hidden_size": 64}),
        ("lambda_vol_1.0", {"lambda_vol": 1.0}),
        ("lambda_cvar_0.5", {"lambda_cvar": 0.5}),
    ]

    print(
        f"[ablation] date range {args.start}–{args.end}  "
        f"val_start={args.val_start}  test_start={args.test_start}  "
        f"sector_portfolios={use_sectors}",
        flush=True,
    )

    print("Connecting to WRDS...", flush=True)
    conn = get_connection()
    print("Connected.", flush=True)
    try:
        data_prep = prepare_data(
            conn,
            start=args.start,
            end=args.end,
            sector_portfolios=use_sectors,
        )
    finally:
        close_wrds_connection(conn)

    df = data_prep["df"]
    feature_cols = data_prep["feature_cols"]
    sector_labels = data_prep.get("sector_labels") or []
    sector_ret_cols = data_prep.get("sector_ret_cols") or []

    if use_sectors:
        X, R, dates = build_rolling_windows_sector_heads(
            df,
            window_len=window_len,
            feature_cols=feature_cols,
            sector_ret_cols=sector_ret_cols,
        )
    else:
        X, R, dates = build_rolling_windows(
            df, window_len=window_len, feature_cols=feature_cols
        )

    if X.shape[0] < 80:
        print(f"[ablation] Too few windows ({X.shape[0]}). Widen date range or lower window_len.", flush=True)
        return 1

    (
        X_train,
        R_train,
        d_train,
        X_val,
        R_val,
        d_val,
        X_test,
        R_test,
        d_test,
    ) = train_val_test_split(
        X,
        R,
        dates,
        val_start=args.val_start,
        test_start=args.test_start,
        df_index=df.index,
        window_len=window_len,
    )

    print(
        f"[ablation] windows: train={X_train.shape[0]}  val={X_val.shape[0]}  test={X_test.shape[0]}",
        flush=True,
    )
    if X_train.shape[0] < 30 or X_val.shape[0] < 10 or X_test.shape[0] < 5:
        print("[ablation] Split too small; adjust --val-start / --test-start.", flush=True)
        return 1

    data_train = {
        "X_train": X_train,
        "R_train": R_train,
        "X_val": X_val,
        "R_val": R_val,
        "feature_cols": feature_cols,
        "df": df,
        "window_len": window_len,
    }
    if use_sectors:
        data_train["n_sectors"] = len(sector_labels)
    else:
        data_train["n_assets"] = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: list[dict] = []
    split_meta = {
        "data_start": args.start,
        "data_end": args.end,
        "val_start": args.val_start,
        "test_start": args.test_start,
        "window_len": window_len,
        "embargo": True,
        "train_windows": int(X_train.shape[0]),
        "val_windows": int(X_val.shape[0]),
        "test_windows": int(X_test.shape[0]),
        "sector_portfolios": use_sectors,
    }

    for name, overrides in experiments:
        cfg = {**base, **overrides}
        kw = _training_kw(cfg)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        if use_sectors:
            model, history = run_training_sector_heads(
                deepcopy(data_train),
                device=device,
                **kw,
            )
            test_loss, _ = validate_sector_heads(
                model,
                X_test,
                R_test,
                device,
                batch_size=min(cfg["batch_size"], 256),
                lambda_cvar=cfg["lambda_cvar"],
                lambda_turnover=0.01,
                lambda_vol=cfg["lambda_vol"],
                lambda_path=0.01,
                lambda_diversify=cfg.get("lambda_diversify", 0.0),
                min_weight=cfg.get("min_weight", 0.1),
                lambda_vol_excess=cfg.get("lambda_vol_excess", 1.0),
                target_vol_annual=cfg.get("target_vol_annual", 0.25),
            )
        else:
            data_2a = {
                "X_train": data_train["X_train"],
                "R_train": data_train["R_train"],
                "X_val": data_train["X_val"],
                "R_val": data_train["R_val"],
                "n_assets": 2,
                "window_len": window_len,
            }
            model, history = run_training(
                deepcopy(data_2a),
                device=device,
                **kw,
            )
            test_loss, _ = validate(
                model,
                X_test,
                R_test,
                device,
                batch_size=min(cfg["batch_size"], 256),
                lambda_cvar=cfg["lambda_cvar"],
                lambda_turnover=0.01,
                lambda_vol=cfg["lambda_vol"],
                lambda_path=0.01,
                lambda_diversify=cfg.get("lambda_diversify", 0.0),
                min_weight=cfg.get("min_weight", 0.1),
                lambda_vol_excess=cfg.get("lambda_vol_excess", 1.0),
                target_vol_annual=cfg.get("target_vol_annual", 0.25),
            )

        best_val = min(h["val_loss"] for h in history)
        results.append(
            {
                "name": name,
                "best_val_loss": float(best_val),
                "test_loss": float(test_loss),
                "epochs_ran": len(history),
                "overrides": overrides or None,
                "history": _compact_history(history),
            }
        )
        print(
            f"  {name:24s}  best_val={best_val:.6f}  test={test_loss:.6f}  epochs={len(history)}",
            flush=True,
        )

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"split": split_meta, "baseline_config": {k: base[k] for k in sorted(base)}, "runs": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
