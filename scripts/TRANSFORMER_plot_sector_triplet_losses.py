#!/usr/bin/env python3
"""
Side-by-side train vs validation loss for the three full WRDS sector-portfolio runs:

  batch_32, hidden_64, lambda_vol_1.0

Expects ``ipo_optimizer_training_history.json`` in each
``results/transformer_sector_runs/<run_id>/`` (written by ``run_ipo_optimizer_wrds.py``
and copied by ``scripts/TRANSFORMER_run_three_sector_models.py`` after each run).

Usage (repo root)::

  python scripts/TRANSFORMER_plot_sector_triplet_losses.py

  python scripts/TRANSFORMER_plot_sector_triplet_losses.py --out "figures/old diagrams/transformer_sector_runs/triplet_loss.png"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRIPLET: tuple[tuple[str, str], ...] = (
    ("batch_32", "batch_size = 32"),
    ("hidden_64", "hidden_size = 64"),
    ("lambda_vol_1.0", "lambda_vol = 1.0"),
)


def _safe_abs_log(y: float) -> float:
    return max(abs(y), 1e-12)


def _load_from_combined(path: Path) -> dict[str, list]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    runs = payload.get("runs") or []
    out: dict[str, list] = {}
    for r in runs:
        name = r.get("name")
        hist = r.get("history")
        if name and isinstance(hist, list) and hist:
            out[str(name)] = hist
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Plot 1x3 train/val loss for transformer sector runs (full WRDS)."
    )
    p.add_argument(
        "--combined-json",
        type=Path,
        default=None,
        help="Optional: one JSON with {\"runs\": [{\"name\": \"batch_32\", \"history\": [...]}, ...]} "
        "(same shape as wrds_transformer_ablation JSON). Overrides per-run files.",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "results" / "transformer_sector_runs",
        help="Directory containing batch_32, hidden_64, lambda_vol_1.0 subfolders",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figures" / "old diagrams" / "transformer_sector_runs" / "triplet_train_val_loss.png",
    )
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    combined: dict[str, list] | None = None
    if args.combined_json is not None:
        if not args.combined_json.is_file():
            print(f"Missing {args.combined_json}", file=sys.stderr)
            return 1
        combined = _load_from_combined(args.combined_json)

    series: list[tuple[str, str, list]] = []
    missing: list[str] = []
    for run_id, title in TRIPLET:
        if combined is not None:
            hist = combined.get(run_id)
            if not hist:
                missing.append(f"run '{run_id}' in combined JSON")
                continue
        else:
            path = args.runs_root / run_id / "ipo_optimizer_training_history.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            with open(path, encoding="utf-8") as f:
                hist = json.load(f)
            if not isinstance(hist, list) or not hist:
                missing.append(f"{path} (empty or invalid)")
                continue
        series.append((run_id, title, hist))

    if len(series) != len(TRIPLET):
        print(
            "Need all three history files. Missing or invalid:\n  "
            + "\n  ".join(missing),
            file=sys.stderr,
        )
        print(
            "\nRe-run: python scripts/TRANSFORMER_run_three_sector_models.py\n"
            "or copy ipo_optimizer_training_history.json into each run folder.",
            file=sys.stderr,
        )
        return 1

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), squeeze=True)
    for ax, (_, subtitle, hist) in zip(axes, series):
        ep = [int(h["epoch"]) for h in hist]
        tr = [_safe_abs_log(float(h["train_loss"])) for h in hist]
        va = [_safe_abs_log(float(h["val_loss"])) for h in hist]
        ax.semilogy(ep, tr, label="Train", marker="o", markersize=3)
        ax.semilogy(ep, va, label="Validation", marker="s", markersize=3)
        ax.set_title(subtitle)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("|Loss| (log scale)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Full WRDS sector-portfolio transformers: batch vs hidden vs lambda_vol",
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
