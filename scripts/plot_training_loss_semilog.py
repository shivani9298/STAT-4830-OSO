#!/usr/bin/env python3
"""
Semilog train/val loss plot from results/training_history.csv (same style as run_ipo_optimizer_wrds).
|Loss| on the y-axis so negative losses are visible on log scale.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def main() -> None:
    path = RESULTS / "training_history.csv"
    df = pd.read_csv(path)
    epochs_x = df["epoch"].values
    train_loss = df["train_loss"].values.astype(float)
    val_loss = df["val_loss"].values.astype(float)

    smooth_window = min(10, len(train_loss))
    train_smooth = np.convolve(train_loss, np.ones(smooth_window) / smooth_window, mode="same")

    t_raw = np.clip(np.abs(train_loss), 1e-8, None)
    t_smooth = np.clip(np.abs(train_smooth), 1e-8, None)
    v_plot = np.clip(np.abs(val_loss), 1e-8, None)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(epochs_x, t_raw, alpha=0.25, color="#1f77b4", linewidth=0.8)
    ax.semilogy(
        epochs_x,
        t_smooth,
        color="#1f77b4",
        linewidth=2,
        label="Train loss (10-ep smoothed)",
    )
    ax.semilogy(
        epochs_x,
        v_plot,
        color="#ff7f0e",
        linewidth=2,
        marker="s",
        markersize=2,
        label="Validation loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|Loss| (log scale)")
    ax.set_title(
        f"IPO optimizer — train vs validation loss (semilog, |loss|)\n"
        f"{len(df)} epochs  |  from training_history.csv"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = FIG / "ipo_training_loss_semilog.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
