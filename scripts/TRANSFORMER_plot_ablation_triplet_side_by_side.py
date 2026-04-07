#!/usr/bin/env python3
"""
Side-by-side train vs validation loss for three WRDS ablations:
  batch_32, hidden_64, lambda_vol_1.0

Reads results/wrds_transformer_ablation_2020_2024.json by default.

Usage:
  python scripts/TRANSFORMER_plot_ablation_triplet_side_by_side.py
  python scripts/TRANSFORMER_plot_ablation_triplet_side_by_side.py --json results/wrds_transformer_ablation_2020_2024.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

TRIPLET = ("batch_32", "hidden_64", "lambda_vol_1.0")


def _safe_abs_log(y: float) -> float:
    return max(abs(y), 1e-12)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--json",
        type=Path,
        default=ROOT / "results" / "wrds_transformer_ablation_2020_2024.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figures" / "wrds_ablation_loss" / "ablation_triplet_batch_32_hidden_64_lambda_vol.png",
    )
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    if not args.json.is_file():
        print(f"Missing {args.json}", file=sys.stderr)
        return 1

    with open(args.json, encoding="utf-8") as f:
        payload = json.load(f)

    by_name = {r.get("name"): r for r in (payload.get("runs") or [])}
    missing = [n for n in TRIPLET if n not in by_name or not by_name[n].get("history")]
    if missing:
        print(f"Missing runs or history: {missing}", file=sys.stderr)
        return 1

    titles = {
        "batch_32": "batch_size = 32",
        "hidden_64": "hidden_size = 64",
        "lambda_vol_1.0": "lambda_vol = 1.0",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), squeeze=True)
    for ax, name in zip(axes, TRIPLET):
        hist = by_name[name]["history"]
        ep = [h["epoch"] for h in hist]
        tr = [_safe_abs_log(h["train_loss"]) for h in hist]
        va = [_safe_abs_log(h["val_loss"]) for h in hist]
        ax.semilogy(ep, tr, label="Train", marker="o", markersize=3)
        ax.semilogy(ep, va, label="Val", marker="s", markersize=3)
        ax.set_title(titles[name])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("|Loss| (log)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Transformer ablations (2020–2024 short horizon): batch vs hidden vs lambda_vol",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
