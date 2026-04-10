#!/usr/bin/env python3
"""
GRU vs LSTM comparison for the WRDS IPO sector-head optimizer.

Reads training history JSONs (no training rerun) and optional validation PNGs.

Writes:
  figures/ipo_optimizer/gru/train_and_val_loss.png
  figures/ipo_optimizer/lstm/train_and_val_loss.png
  figures/ipo_optimizer/comparison/gru_vs_lstm_train_loss.png
  figures/ipo_optimizer/comparison/gru_vs_lstm_val_loss.png
  figures/ipo_optimizer/comparison/gru_vs_lstm_train_and_val_loss.png  (two stacked panels)
  figures/ipo_optimizer/comparison/gru_vs_lstm_validation_returns_side_by_side.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def _load_history(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _epochs_and_losses(rows: list[dict]) -> tuple[list[int], list[float], list[float]]:
    e = [int(x["epoch"]) for x in rows]
    tr = [float(x["train_loss"]) for x in rows]
    va = [float(x["val_loss"]) for x in rows]
    return e, tr, va


def plot_per_model_retrospective(json_path: Path, out_path: Path, model_label: str) -> None:
    rows = _load_history(json_path)
    ep, tr, va = _epochs_and_losses(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ep, tr, label="Train loss", color="C0", linewidth=2)
    ax.plot(ep, va, label="Validation loss", color="C1", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"IPO optimizer (sector heads): {model_label} — train vs validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_train_loss_overlay(gru_path: Path, lstm_path: Path, out_path: Path) -> None:
    g = _load_history(gru_path)
    l = _load_history(lstm_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    eg, tg, _ = _epochs_and_losses(g)
    el, tl, _ = _epochs_and_losses(l)
    ax.plot(eg, tg, label="GRU", color="C0", linewidth=2)
    ax.plot(el, tl, label="LSTM", color="C1", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title("IPO optimizer (sector heads): train loss — GRU vs LSTM")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_val_loss_overlay(gru_path: Path, lstm_path: Path, out_path: Path) -> None:
    g = _load_history(gru_path)
    l = _load_history(lstm_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    eg, _, vg = _epochs_and_losses(g)
    el, _, vl = _epochs_and_losses(l)
    ax.plot(eg, vg, label="GRU", color="C0", linewidth=2)
    ax.plot(el, vl, label="LSTM", color="C1", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("IPO optimizer (sector heads): validation loss — GRU vs LSTM")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_stacked_train_val_comparison(gru_path: Path, lstm_path: Path, out_path: Path) -> None:
    g = _load_history(gru_path)
    l = _load_history(lstm_path)
    eg, tg, vg = _epochs_and_losses(g)
    el, tl, vl = _epochs_and_losses(l)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax0.plot(eg, tg, label="GRU", color="C0", linewidth=2)
    ax0.plot(el, tl, label="LSTM", color="C1", linewidth=2)
    ax0.set_ylabel("Train loss")
    ax0.set_title("IPO optimizer (sector heads): GRU vs LSTM")
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.3)
    ax1.plot(eg, vg, label="GRU", color="C0", linewidth=2)
    ax1.plot(el, vl, label="LSTM", color="C1", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_returns_side_by_side(gru_png: Path, lstm_png: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, path, title in (
        (axes[0], gru_png, "GRU"),
        (axes[1], lstm_png, "LSTM"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        im = mpimg.imread(path)
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(title, fontsize=12)
    fig.suptitle("Validation: cumulative growth vs 50/50 (same scale per image)", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    res = ROOT / "results"
    cmp_dir = ROOT / "figures" / "ipo_optimizer" / "comparison"
    gru_dir = ROOT / "figures" / "ipo_optimizer" / "gru"
    lstm_dir = ROOT / "figures" / "ipo_optimizer" / "lstm"
    gru_h = res / "ipo_optimizer_training_history_gru.json"
    lstm_h = res / "ipo_optimizer_training_history_lstm.json"
    gru_png = gru_dir / "validation_returns_vs_equal_weight.png"
    lstm_png = lstm_dir / "validation_returns_vs_equal_weight.png"

    missing_hist = [p for p in (gru_h, lstm_h) if not p.is_file()]
    if missing_hist:
        print("Missing history JSON:", file=sys.stderr)
        for p in missing_hist:
            print(f"  {p}", file=sys.stderr)
        return 1

    plot_per_model_retrospective(gru_h, gru_dir / "train_and_val_loss.png", "GRU")
    print(f"Wrote {gru_dir / 'train_and_val_loss.png'}")
    plot_per_model_retrospective(lstm_h, lstm_dir / "train_and_val_loss.png", "LSTM")
    print(f"Wrote {lstm_dir / 'train_and_val_loss.png'}")

    plot_train_loss_overlay(gru_h, lstm_h, cmp_dir / "gru_vs_lstm_train_loss.png")
    print(f"Wrote {cmp_dir / 'gru_vs_lstm_train_loss.png'}")
    plot_val_loss_overlay(gru_h, lstm_h, cmp_dir / "gru_vs_lstm_val_loss.png")
    print(f"Wrote {cmp_dir / 'gru_vs_lstm_val_loss.png'}")
    plot_stacked_train_val_comparison(gru_h, lstm_h, cmp_dir / "gru_vs_lstm_train_and_val_loss.png")
    print(f"Wrote {cmp_dir / 'gru_vs_lstm_train_and_val_loss.png'}")

    missing_png = [p for p in (gru_png, lstm_png) if not p.is_file()]
    if missing_png:
        print("Skipping returns side-by-side (missing PNG):", file=sys.stderr)
        for p in missing_png:
            print(f"  {p}", file=sys.stderr)
    else:
        plot_returns_side_by_side(gru_png, lstm_png, cmp_dir / "gru_vs_lstm_validation_returns_side_by_side.png")
        print(f"Wrote {cmp_dir / 'gru_vs_lstm_validation_returns_side_by_side.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
