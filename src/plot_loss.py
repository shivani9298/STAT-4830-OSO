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
    raw_alpha: float = 0.22,
    rolling_linewidth: float = 2.2,
    plot_lr: bool = True,
) -> Path:
    """
    Plot per-epoch ``train_loss`` and ``val_loss`` with optional rolling means, plus a horizontal
    line for a single **test-set** loss evaluation (post-training; not per-epoch).

    Raw per-epoch curves are de-emphasized (``raw_alpha``); rolling means are the main read.
    If ``plot_lr`` and history rows include ``lr``, draws learning rate on a right-hand axis.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs_x = [int(h["epoch"]) for h in history]
    train_loss = [float(h["train_loss"]) for h in history]
    val_loss = [float(h["val_loss"]) for h in history]
    tr_r = _rolling_mean(train_loss, rolling_epochs)
    va_r = _rolling_mean(val_loss, rolling_epochs)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_x, train_loss, alpha=raw_alpha, label="Train loss (per epoch)", color="C0")
    ax.plot(epochs_x, val_loss, alpha=raw_alpha, label="Validation loss (per epoch)", color="C1")
    if rolling_epochs > 1:
        ax.plot(
            epochs_x,
            tr_r,
            label=f"Train (rolling-{rolling_epochs})",
            color="C0",
            linewidth=rolling_linewidth,
        )
        ax.plot(
            epochs_x,
            va_r,
            label=f"Val (rolling-{rolling_epochs})",
            color="C1",
            linewidth=rolling_linewidth,
        )
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
    ax.grid(True, alpha=0.3)

    if plot_lr and history and "lr" in history[0]:
        lr_y = [float(h["lr"]) for h in history]
        ax2 = ax.twinx()
        ax2.plot(epochs_x, lr_y, color="0.45", linewidth=1.2, linestyle=":", label="Learning rate")
        ax2.set_ylabel("Learning rate", color="0.35")
        ax2.tick_params(axis="y", labelcolor="0.35")
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, loc="best", fontsize=8)
    else:
        ax.legend(loc="best", fontsize=8)

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
    rolling_epochs: int = 1,
) -> Path:
    """
    Save a line chart of ``train_loss`` and ``val_loss`` vs ``epoch``.

    Default is linear scale (train loss can be negative). Set ``semilogy=True`` for log
    scale of |loss| (small positive val losses).

    ``history`` entries must include keys ``epoch``, ``train_loss``, ``val_loss``.
    If ``rolling_epochs`` > 1, overlays smoothed curves (less spiky than raw per-epoch lines).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs_x = [int(h["epoch"]) for h in history]
    train_loss = [float(h["train_loss"]) for h in history]
    val_loss = [float(h["val_loss"]) for h in history]
    tr_r = _rolling_mean(train_loss, rolling_epochs) if rolling_epochs > 1 else train_loss
    va_r = _rolling_mean(val_loss, rolling_epochs) if rolling_epochs > 1 else val_loss

    fig, ax = plt.subplots(figsize=(8, 5))
    if semilogy:
        t_plot = [max(abs(x), 1e-8) for x in train_loss]
        v_plot = [max(abs(x), 1e-8) for x in val_loss]
        ax.semilogy(epochs_x, t_plot, label="Train (raw)", alpha=0.35, linewidth=1)
        ax.semilogy(epochs_x, v_plot, label="Val (raw)", alpha=0.35, linewidth=1)
        if rolling_epochs > 1:
            trs = [max(abs(x), 1e-8) for x in tr_r]
            vas = [max(abs(x), 1e-8) for x in va_r]
            ax.semilogy(
                epochs_x,
                trs,
                label=f"Train rolling-{rolling_epochs}",
                linewidth=2,
            )
            ax.semilogy(epochs_x, vas, label=f"Val rolling-{rolling_epochs}", linewidth=2)
    else:
        if rolling_epochs > 1:
            ax.plot(epochs_x, train_loss, label="Train (raw)", alpha=0.35, linewidth=1)
            ax.plot(epochs_x, val_loss, label="Val (raw)", alpha=0.35, linewidth=1)
            ax.plot(epochs_x, tr_r, label=f"Train rolling-{rolling_epochs}", linewidth=2.2, color="C0")
            ax.plot(epochs_x, va_r, label=f"Val rolling-{rolling_epochs}", linewidth=2.2, color="C1")
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
    excess_panel: bool = True,
    fill_between: bool = True,
) -> Path:
    """
    Plot normalized cumulative wealth ``cumprod(1 + r)`` for the learned weights vs a fixed **50/50**
    allocation between market and IPO returns (per window row).

    - ``weights`` / ``R``: ``(N, 2)`` or sector heads ``(N, G, 2)``. For ``(N, G, 2)``, daily returns
      are averaged across sector sleeves so two comparable scalar series are plotted.

    If ``excess_panel``, adds a second subplot: cumulative compound **excess** vs 50/50,
    ``Π(1 + r_model - r_50/50) - 1``, matching common holdout plots. ``fill_between`` shades the
    gap between model and benchmark on the wealth panel.
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
    excess_daily = model_ret - eq_ret
    cum_excess = np.cumprod(1.0 + excess_daily) - 1.0

    t = np.arange(n)
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(dates))
        t_plot = idx
        xlabel = "Date"
    except Exception:
        t_plot = t
        xlabel = "Window index"

    if excess_panel:
        fig, (ax, ax_ex) = plt.subplots(
            2, 1, figsize=(9, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]}
        )
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax_ex = None

    if fill_between:
        ax.fill_between(t_plot, we, wm, where=(wm >= we), interpolate=True, alpha=0.12, color="C0", label=None)
        ax.fill_between(t_plot, we, wm, where=(wm < we), interpolate=True, alpha=0.12, color="C1", label=None)
    ax.plot(t_plot, wm, label=f"Learned allocator (total {tot_m:+.1%})", color="C0", linewidth=2)
    ax.plot(t_plot, we, label=f"50/50 market/IPO (total {tot_e:+.1%})", color="C1", linewidth=2, linestyle="--")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_ylabel("Growth of $1 (normalized)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    if ax_ex is not None:
        ax_ex.axhline(0.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax_ex.plot(
            t_plot,
            cum_excess,
            color="C2",
            linewidth=1.8,
            label="Cumulative excess vs 50/50 (compound)",
        )
        ax_ex.set_ylabel("Excess vs 50/50")
        ax_ex.set_xlabel(xlabel)
        ax_ex.legend(loc="best", fontsize=8)
        ax_ex.grid(True, alpha=0.3)
    else:
        ax.set_xlabel(xlabel)

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
