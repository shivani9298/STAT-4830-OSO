#!/usr/bin/env python3
"""
Plot train vs validation loss (per epoch) for each run in a WRDS ablation JSON.

Requires ``history`` on each run (epoch, train_loss, val_loss). Re-run
``wrds_transformer_ablation_short_horizon.py`` with an updated script if your
JSON only has best_val_loss / test_loss.

Usage:
  python scripts/plot_wrds_ablation_losses.py
  python scripts/plot_wrds_ablation_losses.py --json results/wrds_transformer_ablation_2020_2024.json
  python scripts/plot_wrds_ablation_losses.py --out-dir figures/wrds_ablation_loss
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt


def _safe_abs_log(y: float) -> float:
    return max(abs(y), 1e-12)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--json",
        type=Path,
        default=ROOT / "results" / "wrds_transformer_ablation_2020_2024.json",
        help="Ablation results JSON (with per-run history)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "figures" / "wrds_ablation_loss",
        help="Directory for combined grid + per-run PNGs",
    )
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    path = args.json
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    with open(path) as f:
        payload = json.load(f)

    runs = payload.get("runs") or []
    if not runs:
        print("No runs in JSON.", file=sys.stderr)
        return 1

    missing = [r.get("name", "?") for r in runs if not r.get("history")]
    if missing:
        print(
            "These runs have no per-epoch history (re-run ablation with an updated "
            f"wrds_transformer_ablation_short_horizon.py): {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(runs)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)

    for idx, run in enumerate(runs):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        hist = run["history"]
        ep = [h["epoch"] for h in hist]
        tr = [_safe_abs_log(h["train_loss"]) for h in hist]
        va = [_safe_abs_log(h["val_loss"]) for h in hist]
        ax.semilogy(ep, tr, label="Train", marker="o", markersize=3)
        ax.semilogy(ep, va, label="Val", marker="s", markersize=3)
        ax.set_title(run.get("name", f"run_{idx}"))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("|Loss| (log)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        one = out_dir / f"{run.get('name', idx)}.png"
        fig_one, ax_one = plt.subplots(figsize=(7, 4))
        ax_one.semilogy(ep, tr, label="Train", marker="o", markersize=4)
        ax_one.semilogy(ep, va, label="Val", marker="s", markersize=4)
        ax_one.set_title(run.get("name", str(idx)))
        ax_one.set_xlabel("Epoch")
        ax_one.set_ylabel("|Loss| (log)")
        ax_one.legend()
        ax_one.grid(True, alpha=0.3)
        fig_one.tight_layout()
        fig_one.savefig(one, dpi=args.dpi)
        plt.close(fig_one)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.suptitle("WRDS Transformer ablation: train vs validation loss", fontsize=14, y=1.01)
    fig.tight_layout()
    grid_path = out_dir / "wrds_ablation_loss_grid.png"
    fig.savefig(grid_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {grid_path}")
    print(f"Per-run PNGs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
