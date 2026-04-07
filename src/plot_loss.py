"""
Plot training vs validation loss from ``run_training`` / ``run_training_sector_heads`` history.
"""
from __future__ import annotations

import os

# Before matplotlib: avoid duplicate OpenMP on Windows (PyTorch/numpy + MPL) and Qt GUI backend.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_loss(
    history: list[dict[str, Any]],
    out_path: str | Path,
    *,
    title: str = "IPO Optimizer: Training and Validation Loss",
    dpi: int = 150,
    semilogy: bool = False,
) -> Path:
    """
    Save a line chart of ``train_loss`` and ``val_loss`` vs ``epoch``.

    Default is linear scale (train loss can be negative). Set ``semilogy=True`` for log
    scale of |loss| (small positive val losses).

    ``history`` entries must include keys ``epoch``, ``train_loss``, ``val_loss``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs_x = [int(h["epoch"]) for h in history]
    train_loss = [float(h["train_loss"]) for h in history]
    val_loss = [float(h["val_loss"]) for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    if semilogy:
        t_plot = [max(abs(x), 1e-8) for x in train_loss]
        v_plot = [max(abs(x), 1e-8) for x in val_loss]
        ax.semilogy(epochs_x, t_plot, label="Train loss (|.|)", marker="o", markersize=3)
        ax.semilogy(epochs_x, v_plot, label="Validation loss (|.|)", marker="s", markersize=3)
    else:
        ax.plot(epochs_x, train_loss, label="Train loss", marker="o", markersize=3)
        ax.plot(epochs_x, val_loss, label="Validation loss", marker="s", markersize=3)
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def slim_history_for_json(history: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    """Epoch / train_loss / val_loss only, JSON-serializable."""
    out = []
    for h in history:
        out.append(
            {
                "epoch": int(h["epoch"]),
                "train_loss": float(h["train_loss"]),
                "val_loss": float(h["val_loss"]),
            }
        )
    return out
