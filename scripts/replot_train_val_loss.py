#!/usr/bin/env python3
"""
Regenerate **epoch-level** train/val **loss** figures from saved history (CSV or slim JSON).

This is **not** the dated validation path vs 50/50 benchmark — for that, use
``scripts/replot_daily_vs_50_50.py`` with ``ipo_optimizer_returns_val*.csv`` + weights.

  python3 scripts/replot_train_val_loss.py
  python3 scripts/replot_train_val_loss.py --csv results/training_history_lstm.csv --tag lstm

  # Quick smoke train (synthetic tensors, no WRDS/yfinance) + plots
  python3 scripts/replot_train_val_loss.py --synthetic-train --epochs 30
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from src.plot_loss import plot_train_val_rolling_and_test, plot_training_loss


def load_history(path: Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("JSON history must be a list of dicts")
        return raw
    df = pd.read_csv(path)
    need = {"epoch", "train_loss", "val_loss"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must have columns {need}, got {list(df.columns)}")
    rows = []
    for _, r in df.iterrows():
        row = {
            "epoch": int(r["epoch"]),
            "train_loss": float(r["train_loss"]),
            "val_loss": float(r["val_loss"]),
        }
        if "lr" in df.columns and pd.notna(r.get("lr")):
            row["lr"] = float(r["lr"])
        rows.append(row)
    return rows


def save_history_csv(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    keys = ["epoch", "train_loss", "val_loss", "lr"]
    flat = []
    for h in history:
        row = {k: h.get(k) for k in keys if k in h or k == "lr"}
        if "lr" not in row:
            row["lr"] = float("nan")
        flat.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in flat:
            w.writerow({k: row.get(k, "") for k in keys})


def synthetic_training(
    *,
    epochs: int,
    seed: int,
    lr_schedule: str | None,
    plateau_patience: int,
    warmup_epochs: int,
) -> list[dict]:
    import torch
    from src.train import run_training

    rng = np.random.default_rng(seed)
    n_tr, n_va = 450, 120
    t_win, n_feat, n_assets = 40, 5, 2
    # Mild structure so loss can improve (not pure noise)
    X_train = rng.standard_normal((n_tr, t_win, n_feat)).astype(np.float32)
    X_val = rng.standard_normal((n_va, t_win, n_feat)).astype(np.float32)
    sig_tr = X_train[:, -1, 0].astype(np.float64)
    sig_va = X_val[:, -1, 0].astype(np.float64)
    noise_m = 0.01
    noise_i = 0.015
    R_train = np.column_stack(
        [
            sig_tr * 0.002 + rng.normal(0, noise_m, n_tr),
            -sig_tr * 0.001 + rng.normal(0, noise_i, n_tr),
        ]
    ).astype(np.float32)
    R_val = np.column_stack(
        [
            sig_va * 0.002 + rng.normal(0, noise_m, n_va),
            -sig_va * 0.001 + rng.normal(0, noise_i, n_va),
        ]
    ).astype(np.float32)
    data = {
        "X_train": X_train,
        "R_train": R_train,
        "X_val": X_val,
        "R_val": R_val,
        "n_assets": n_assets,
    }
    device = torch.device("cpu")
    _, history = run_training(
        data,
        device=device,
        epochs=epochs,
        lr=3e-3,
        batch_size=48,
        patience=max(epochs, 50),
        model_type="gru",
        hidden_size=48,
        num_layers=1,
        verbose=True,
        log_every=5,
        lr_schedule=lr_schedule,
        plateau_patience=plateau_patience,
        warmup_epochs=warmup_epochs,
        max_grad_norm=1.0,
        grad_norm_mode="clip",
        lambda_cvar=0.3,
        lambda_vol=0.3,
        lambda_turnover=0.02,
        lambda_path=0.02,
        winsor_abs=0.08,
        cvar_temperature=0.15,
    )
    return history


def main() -> int:
    p = argparse.ArgumentParser(description="Replot train/val loss from CSV/JSON or synthetic train.")
    p.add_argument("--csv", type=Path, default=None, help="training_history.csv or slim JSON")
    p.add_argument("--tag", type=str, default="", help="output filename tag")
    p.add_argument("--out-dir", type=Path, default=ROOT / "figures" / "ipo_optimizer_replots")
    p.add_argument("--rolling", type=int, default=3)
    p.add_argument("--synthetic-train", action="store_true", help="Run quick CPU train on synthetic data first")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr-schedule", type=str, default="plateau")
    p.add_argument("--plateau-patience", type=int, default=3)
    p.add_argument("--warmup-epochs", type=int, default=0)
    args = p.parse_args()

    if args.synthetic_train:
        print("[replot] Running synthetic training (no WRDS)...", flush=True)
        hist = synthetic_training(
            epochs=args.epochs,
            seed=args.seed,
            lr_schedule=args.lr_schedule,
            plateau_patience=args.plateau_patience,
            warmup_epochs=args.warmup_epochs,
        )
        syn_csv = ROOT / "results" / "synthetic_smoke_training_history.csv"
        save_history_csv(hist, syn_csv)
        print(f"[replot] Wrote {syn_csv}", flush=True)
        history = hist
        tag = args.tag or "synthetic_smoke"
    else:
        csv_path = args.csv or (ROOT / "results" / "training_history.csv")
        if not csv_path.is_file():
            print(f"Missing {csv_path}; pass --csv or use --synthetic-train", file=sys.stderr)
            return 1
        history = load_history(csv_path)
        tag = args.tag or csv_path.stem

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    title_base = f"Train/val loss ({tag})"
    roll_w = max(1, args.rolling)
    semilogy_path = plot_training_loss(
        history,
        out_dir / f"loss_semilogy_{tag}.png",
        title=title_base,
        semilogy=True,
        rolling_epochs=roll_w,
    )
    linear_path = plot_training_loss(
        history,
        out_dir / f"loss_linear_{tag}.png",
        title=title_base,
        semilogy=False,
        rolling_epochs=roll_w,
    )
    roll_path = plot_train_val_rolling_and_test(
        history,
        out_dir / f"loss_rolling_{tag}.png",
        rolling_epochs=roll_w,
        title=f"{title_base} — rolling mean {roll_w}",
    )
    print(f"Wrote:\n  {semilogy_path}\n  {linear_path}\n  {roll_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
