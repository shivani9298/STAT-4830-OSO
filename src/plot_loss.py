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

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rolling_mean(xs: list[float], window: int) -> list[float]:
    if window <= 1 or len(xs) == 0:
        return list(xs)
    out: list[float] = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        chunk = xs[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_train_val_rolling_and_test(
    history: list[dict[str, Any]],
    out_path: str | Path,
    *,
    test_loss: float | None = None,
    rolling_epochs: int = 3,
    title: str = "Training / validation loss (rolling) and test loss",
    dpi: int = 150,
) -> Path:
    """
    Plot per-epoch ``train_loss`` and ``val_loss`` with optional rolling means, plus a horizontal
    line for a single **test-set** loss evaluation (post-training; not per-epoch).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs_x = [int(h["epoch"]) for h in history]
    train_loss = [float(h["train_loss"]) for h in history]
    val_loss = [float(h["val_loss"]) for h in history]
    tr_r = _rolling_mean(train_loss, rolling_epochs)
    va_r = _rolling_mean(val_loss, rolling_epochs)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_x, train_loss, alpha=0.35, label="Train loss", color="C0")
    ax.plot(epochs_x, val_loss, alpha=0.35, label="Validation loss", color="C1")
    if rolling_epochs > 1:
        ax.plot(epochs_x, tr_r, label=f"Train (rolling-{rolling_epochs})", color="C0", linewidth=2)
        ax.plot(epochs_x, va_r, label=f"Val (rolling-{rolling_epochs})", color="C1", linewidth=2)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    if test_loss is not None and np.isfinite(test_loss):
        ax.axhline(
            float(test_loss),
            color="C2",
            linestyle="--",
            linewidth=2,
            label=f"Test loss (one-shot eval) = {float(test_loss):.6f}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


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


def plot_cumulative_returns_vs_equal_weight(
    weights: np.ndarray,
    R: np.ndarray,
    dates: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "Cumulative growth: model vs 50/50 (validation)",
    dpi: int = 150,
) -> Path:
    """
    Plot normalized cumulative wealth ``cumprod(1 + r)`` for the learned weights vs a fixed **50/50**
    allocation between market and IPO returns (per window row).

    - ``weights`` / ``R``: ``(N, 2)`` or sector heads ``(N, G, 2)``. For ``(N, G, 2)``, daily returns
      are averaged across sector sleeves so two comparable scalar series are plotted.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = np.asarray(weights, dtype=np.float64)
    r = np.asarray(R, dtype=np.float64)
    if np.isnan(w).any() or np.isnan(r).any():
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    if w.ndim == 2 and r.ndim == 2 and w.shape == r.shape:
        model_ret = (w * r).sum(axis=1)
        eq_ret = 0.5 * r[:, 0] + 0.5 * r[:, 1]
    elif w.ndim == 3 and r.ndim == 3 and w.shape == r.shape:
        model_g = (w * r).sum(axis=2)
        eq_g = 0.5 * r[:, :, 0] + 0.5 * r[:, :, 1]
        model_ret = np.mean(model_g, axis=1)
        eq_ret = np.mean(eq_g, axis=1)
    else:
        raise ValueError(
            f"weights and R must match with shapes (N,2) or (N,G,2); got {w.shape}, {r.shape}"
        )

    n = len(model_ret)
    if n == 0:
        raise ValueError("empty return series")

    def _wealth(x: np.ndarray) -> np.ndarray:
        c = np.cumprod(1.0 + x)
        return c / c[0] if c[0] != 0 else c

    wm = _wealth(model_ret)
    we = _wealth(eq_ret)
    tot_m = float(np.prod(1.0 + model_ret) - 1.0)
    tot_e = float(np.prod(1.0 + eq_ret) - 1.0)

    t = np.arange(n)
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(dates))
        t_plot = idx
        xlabel = "Date"
    except Exception:
        t_plot = t
        xlabel = "Window index"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_plot, wm, label=f"Learned allocator (total {tot_m:+.1%})", color="C0", linewidth=2)
    ax.plot(t_plot, we, label=f"50/50 market/IPO (total {tot_e:+.1%})", color="C1", linewidth=2, linestyle="--")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_ylabel("Growth of $1 (normalized)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def slim_history_for_json(history: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    """Epoch / train_loss / val_loss / optional lr, JSON-serializable."""
    out = []
    for h in history:
        row: dict[str, float | int] = {
            "epoch": int(h["epoch"]),
            "train_loss": float(h["train_loss"]),
            "val_loss": float(h["val_loss"]),
        }
        if "lr" in h:
            row["lr"] = float(h["lr"])
        out.append(row)
    return out
