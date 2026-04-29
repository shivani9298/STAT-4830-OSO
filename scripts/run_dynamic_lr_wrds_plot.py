#!/usr/bin/env python3
"""
Load WRDS data for a date range, train with ``lr_schedule=plateau`` (dynamic LR),
and save train/validation loss plots (raw + rolling mean over epochs).

Train/validation split uses ``train_val_test_split`` with the same ``window_len``
embargo as ``notebooks/tune_hyperparameters_wrds.py``.

Example (repo root)::

  python scripts/run_dynamic_lr_wrds_plot.py --start 2020-01-01 --end 2024-12-31

If the price panel starts after 2019, pick ``--val-start`` / ``--test-start`` so the
embargo leaves enough training history (e.g. ``--val-start 2022-01-01 --test-start 2023-01-01``).

Requires ``WRDS_USERNAME`` and ``WRDS_PASSWORD`` in ``.env`` (or env) so the connection is non-interactive.

Outputs:
  results/dynamic_lr_wrds_history.json
  figures/old diagrams/dynamic_lr_wrds_loss.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts.run_ipo_optimizer_wrds import (
    TEST_START as DEFAULT_TEST_START,
    VAL_START as DEFAULT_VAL_START,
    close_wrds_connection,
    load_best_config,
    prepare_data,
)
from src.data_layer import (
    build_rolling_windows,
    build_rolling_windows_sector_heads,
    train_val_test_split,
)
from src.plot_loss import slim_history_for_json
from src.train import run_training, run_training_sector_heads
from src.wrds_data import get_connection


DEFAULTS_MERGE_KEYS = frozenset(
    {
        "window_len",
        "epochs",
        "lr",
        "batch_size",
        "patience",
        "lambda_vol",
        "lambda_cvar",
        "lr_schedule",
        "lr_decay",
        "plateau_patience",
        "hidden_size",
        "model_type",
    }
)


def _roll(x: list[float], w: int) -> np.ndarray:
    s = pd.Series(x, dtype=float)
    return s.rolling(window=w, min_periods=1).mean().values


def main() -> int:
    p = argparse.ArgumentParser(description="WRDS train with plateau LR + loss plots")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument(
        "--val-start",
        default=None,
        help=f"First validation prediction date (default: {DEFAULT_VAL_START}, same as tune notebook)",
    )
    p.add_argument(
        "--test-start",
        default=None,
        help=f"First test prediction date (default: {DEFAULT_TEST_START}; used for embargo with val)",
    )
    p.add_argument("--rolling", type=int, default=5, help="Rolling epoch window for smoothed curves")
    p.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    p.add_argument(
        "--sector",
        action="store_true",
        help="Use sector IPO baskets (slower; default is single market vs IPO index)",
    )
    args = p.parse_args()
    val_start = args.val_start or DEFAULT_VAL_START
    test_start = args.test_start or DEFAULT_TEST_START

    print("[dynamic_lr] Connecting to WRDS...", flush=True)
    conn = get_connection()
    try:
        data_prep = prepare_data(
            conn,
            start=args.start,
            end=args.end,
            sector_portfolios=args.sector,
        )
    finally:
        close_wrds_connection(conn)

    df = data_prep["df"]
    feature_cols = data_prep["feature_cols"]
    use_sectors = bool(data_prep.get("sector_portfolios"))
    sector_labels = data_prep.get("sector_labels") or []
    sector_ret_cols = data_prep.get("sector_ret_cols") or []

    cfg = load_best_config()
    cfg["epochs"] = min(int(args.epochs), int(cfg.get("epochs", 50)))
    cfg["lr_schedule"] = "plateau"
    cfg["lr_decay"] = float(cfg.get("lr_decay", 0.5))
    cfg["plateau_patience"] = int(cfg.get("plateau_patience", 4))
    cfg["patience"] = max(15, cfg["epochs"])

    print(
        f"[dynamic_lr] Config: {cfg['lr_schedule']} lr_decay={cfg['lr_decay']} epochs={cfg['epochs']}  "
        f"val_start={val_start} test_start={test_start} (embargo split)",
        flush=True,
    )

    if use_sectors:
        X, R, dates = build_rolling_windows_sector_heads(
            df,
            window_len=cfg["window_len"],
            feature_cols=feature_cols,
            sector_ret_cols=sector_ret_cols,
        )
    else:
        X, R, dates = build_rolling_windows(
            df, window_len=cfg["window_len"], feature_cols=feature_cols
        )
    wlen = int(cfg["window_len"])
    X_train, R_train, d_train, X_val, R_val, d_val, X_test, R_test, d_test = train_val_test_split(
        X,
        R,
        dates,
        val_start=val_start,
        test_start=test_start,
        df_index=df.index,
        window_len=wlen,
    )

    data = {
        "X_train": X_train,
        "R_train": R_train,
        "dates_train": d_train,
        "X_val": X_val,
        "R_val": R_val,
        "dates_val": d_val,
        "feature_cols": feature_cols,
        "df": df,
        "window_len": cfg["window_len"],
    }
    if use_sectors:
        data["n_sectors"] = len(sector_labels)
        data["n_assets"] = 2
    else:
        data["n_assets"] = 2

    print(
        f"[dynamic_lr] train={X_train.shape[0]} val={X_val.shape[0]} test={X_test.shape[0]} "
        f"sector_mode={use_sectors}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw = dict(
        device=device,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        patience=cfg["patience"],
        lambda_vol=cfg["lambda_vol"],
        lambda_cvar=cfg["lambda_cvar"],
        lambda_diversify=cfg.get("lambda_diversify", 0.0),
        min_weight=cfg.get("min_weight", 0.1),
        lambda_vol_excess=cfg.get("lambda_vol_excess", 1.0),
        target_vol_annual=cfg.get("target_vol_annual", 0.25),
        hidden_size=cfg["hidden_size"],
        model_type=cfg.get("model_type", "gru"),
        verbose=True,
        log_every=1,
        weight_decay=cfg.get("weight_decay", 1e-5),
        dropout=cfg.get("dropout", 0.1),
        num_layers=int(cfg.get("num_layers", 1)),
        cosine_lr=False,
        lr_schedule="plateau",
        lr_decay=cfg["lr_decay"],
        plateau_patience=cfg["plateau_patience"],
        min_lr=float(cfg.get("min_lr", 1e-6)),
        exponential_gamma=float(cfg.get("exponential_gamma", 0.99)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        grad_norm_mode=str(cfg.get("grad_norm_mode", "clip")),
    )

    if use_sectors:
        _, history = run_training_sector_heads(data, **kw)
    else:
        _, history = run_training(data, **kw)

    epochs_x = [h["epoch"] for h in history]
    tr = [float(h["train_loss"]) for h in history]
    va = [float(h["val_loss"]) for h in history]
    lrs = [float(h["lr"]) for h in history] if history and "lr" in history[0] else None

    rw = max(1, int(args.rolling))
    tr_r = _roll(tr, rw)
    va_r = _roll(va, rw)

    out_dir = ROOT / "results"
    fig_dir = ROOT / "figures" / "old diagrams"
    out_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tag = "dynamic_lr_synthetic" if args.synthetic else "dynamic_lr_wrds"
    hist_path = out_dir / f"{tag}_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_start": args.start,
                "data_end": args.end,
                "val_start": val_start,
                "test_start": test_start,
                "split": "embargo_train_val_test",
                "rolling_epochs": rw,
                "sector_portfolios": use_sectors,
                "config": {k: cfg[k] for k in cfg if k in DEFAULTS_MERGE_KEYS},
                "history": slim_history_for_json(history),
            },
            f,
            indent=2,
        )
    print(f"[dynamic_lr] Wrote {hist_path}", flush=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax0 = axes[0]
    ax0.plot(epochs_x, tr, alpha=0.35, color="C0", label="Train (raw)")
    ax0.plot(epochs_x, va, alpha=0.35, color="C1", label="Val (raw)")
    ax0.plot(epochs_x, tr_r, color="C0", linewidth=2, label=f"Train (rolling-{rw})")
    ax0.plot(epochs_x, va_r, color="C1", linewidth=2, label=f"Val (rolling-{rw})")
    ax0.set_ylabel("Loss")
    title_src = f"WRDS {args.start}–{args.end}"
    ax0.set_title(
        f"{title_src}  |  ReduceLROnPlateau  |  "
        f"train={X_train.shape[0]} val={X_val.shape[0]} windows"
    )
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    if lrs:
        ax1.plot(epochs_x, lrs, color="green", marker=".", markersize=4)
        ax1.set_ylabel("Learning rate")
        ax1.set_yscale("log")
    else:
        ax1.text(0.5, 0.5, "lr not logged", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_xlabel("Epoch")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    png = fig_dir / f"{tag}_loss.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"[dynamic_lr] Wrote {png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
