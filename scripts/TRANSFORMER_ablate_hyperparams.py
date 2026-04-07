#!/usr/bin/env python3
"""
Sequential one-at-a-time ablations for SectorMultiHeadTransformerAllocator.

Uses fixed synthetic rolling-window data (same shapes as real sector runs) so results
are reproducible without WRDS. Best validation composite loss is the metric; use WRDS
re-runs to confirm on real data.

Run from repo root: python scripts/TRANSFORMER_ablate_hyperparams.py
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.train import run_training_sector_heads


def _synthetic_sector_data(
    seed: int,
    n_train: int = 240,
    n_val: int = 80,
    window_len: int = 126,
    n_features: int = 4,
    n_sectors: int = 5,
) -> dict:
    rng = np.random.default_rng(seed)
    def stack_split(n: int):
        X = rng.standard_normal((n, window_len, n_features)).astype(np.float32) * 0.02
        R = rng.standard_normal((n, n_sectors, 2)).astype(np.float32) * 0.01
        return X, R
    X_train, R_train = stack_split(n_train)
    X_val, R_val = stack_split(n_val)
    return {
        "X_train": X_train,
        "R_train": R_train,
        "X_val": X_val,
        "R_val": R_val,
        "n_sectors": n_sectors,
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _synthetic_sector_data(seed=42)

    baseline = {
        "epochs": 30,
        "patience": 8,
        "lr": 1e-3,
        "batch_size": 32,
        "lambda_cvar": 0.5,
        "lambda_vol": 0.5,
        "lambda_turnover": 0.01,
        "lambda_path": 0.01,
        "lambda_vol_excess": 1.0,
        "target_vol_annual": 0.25,
        "hidden_size": 64,
        "num_layers": 1,
        "model_type": "transformer",
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "cosine_lr": False,
    }

    experiments: list[tuple[str, dict]] = [
        ("baseline", {}),
        ("lr_3e-4", {"lr": 3e-4}),
        ("lr_3e-3", {"lr": 3e-3}),
        ("weight_decay_0", {"weight_decay": 0.0}),
        ("weight_decay_1e-2", {"weight_decay": 1e-2}),
        ("dropout_0.05", {"dropout": 0.05}),
        ("dropout_0.25", {"dropout": 0.25}),
        ("cosine_lr", {"cosine_lr": True}),
        ("batch_64", {"batch_size": 64}),
        ("batch_128", {"batch_size": 128}),
        ("hidden_128", {"hidden_size": 128}),
        ("lambda_vol_1.0", {"lambda_vol": 1.0}),
        ("lambda_cvar_1.0", {"lambda_cvar": 1.0}),
        ("lambda_vol_excess_0.5", {"lambda_vol_excess": 0.5}),
    ]

    results: list[dict] = []
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    for name, overrides in experiments:
        cfg = {**baseline, **overrides}
        kw = {k: cfg[k] for k in baseline}
        # Fresh data same arrays; re-seed model init per run for fair comparison
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        _, history = run_training_sector_heads(
            deepcopy(data),
            device=device,
            verbose=False,
            log_every=0,
            **kw,
        )
        best_val = min(h["val_loss"] for h in history)
        final_train = history[-1]["train_loss"] if history else float("nan")
        results.append({
            "name": name,
            "best_val_loss": float(best_val),
            "final_train_loss": float(final_train),
            "epochs": len(history),
            "overrides": overrides or None,
        })
        print(f"{name:28s}  best_val_loss={best_val:.6f}  epochs={len(history)}", flush=True)

    baseline_val = next(r["best_val_loss"] for r in results if r["name"] == "baseline")
    ranked = sorted(
        [r for r in results if r["name"] != "baseline"],
        key=lambda r: r["best_val_loss"],
    )

    out = {
        "baseline_best_val_loss": baseline_val,
        "device": str(device),
        "ranked_vs_baseline": [
            {
                "name": r["name"],
                "best_val_loss": r["best_val_loss"],
                "delta_vs_baseline": r["best_val_loss"] - baseline_val,
            }
            for r in ranked
        ],
        "all": results,
    }
    out_path = ROOT / "results" / "transformer_ablation_synthetic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
