#!/usr/bin/env python3
"""
Create a cleaner online-policy adaptation pipeline diagram as PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "architecture" / "online_gru_adaptive_pipeline_clean.png"


def box(ax, x, y, w, h, title, subtitle, fc="#EAF2FF", ec="#9DB6D8"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=8.6, color="#333333")


def arrow(ax, a, b, text=None, rad=0.0):
    ar = FancyArrowPatch(
        a, b,
        arrowstyle="->",
        mutation_scale=12,
        linewidth=1.2,
        color="#4D5A6A",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(ar)
    if text:
        mx = (a[0] + b[0]) / 2
        my = (a[1] + b[1]) / 2
        ax.text(mx, my + 0.015, text, fontsize=8.5, color="#2B3A4A", ha="center", va="bottom")


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header row
    box(ax, 0.03, 0.86, 0.16, 0.09, "Shared Features", "Same cleaned feature pipeline", fc="#DFECFF")
    box(ax, 0.22, 0.86, 0.16, 0.09, "Window Timeline", "Build rolling windows + schedule", fc="#DFECFF")
    box(ax, 0.41, 0.86, 0.16, 0.09, "Per-Step History Slice", "Expanding/fixed lookback to train_end", fc="#DFECFF")
    box(ax, 0.60, 0.86, 0.16, 0.09, "Warm-Start Update", "Fine-tune current GRU for epochs_step", fc="#DDF5E7", ec="#9CCBB1")
    box(ax, 0.79, 0.86, 0.17, 0.09, "Update Policy", "Cadence + gate rules", fc="#FFF3D6", ec="#D8BE83")
    arrow(ax, (0.19, 0.905), (0.22, 0.905))
    arrow(ax, (0.38, 0.905), (0.41, 0.905))
    arrow(ax, (0.57, 0.905), (0.60, 0.905))
    arrow(ax, (0.76, 0.905), (0.79, 0.905))

    # Loop container
    loop = FancyBboxPatch(
        (0.03, 0.25), 0.93, 0.54,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        linewidth=1.3, edgecolor="#A9B5C2", facecolor="#F9FBFD"
    )
    ax.add_patch(loop)
    ax.text(0.50, 0.77, "Per scheduled decision date (t)", ha="center", va="center", fontsize=10, color="#4A5562", fontweight="bold")

    # Inner boxes (clean branch-and-join)
    box(ax, 0.07, 0.61, 0.16, 0.10, "Is this an update step?", "If no, keep previous params", fc="#FFF3D6", ec="#D8BE83")
    box(ax, 0.28, 0.61, 0.15, 0.10, "Train/Val Split", "Chronological split", fc="#DFECFF")
    box(ax, 0.47, 0.61, 0.15, 0.10, "Update quality metrics", "val-loss delta + relative change", fc="#DFECFF")
    box(ax, 0.66, 0.61, 0.14, 0.10, "Accept/revert update?", "Threshold check (optional)", fc="#EFEFF6", ec="#B9B9C9")

    box(ax, 0.39, 0.42, 0.20, 0.10, "Infer weights for date t", "Action inference", fc="#DDF5E7", ec="#9CCBB1")
    box(ax, 0.63, 0.42, 0.21, 0.10, "Record realized outcomes", "Path accounting: return/costs/flags", fc="#F3E8FF", ec="#C6B0DE")

    # Connections
    arrow(ax, (0.23, 0.66), (0.28, 0.66), text="yes-update")
    arrow(ax, (0.43, 0.66), (0.47, 0.66))
    arrow(ax, (0.62, 0.66), (0.66, 0.66))
    arrow(ax, (0.80, 0.61), (0.49, 0.52), text="accepted/reverted params", rad=0.0)
    arrow(ax, (0.23, 0.61), (0.39, 0.47), text="no-update", rad=-0.05)
    arrow(ax, (0.59, 0.47), (0.63, 0.47))

    # Loop back t -> t+1
    arrow(ax, (0.74, 0.42), (0.09, 0.60), text="next decision date (t+1)", rad=0.30)

    # Top to loop hint
    arrow(ax, (0.875, 0.86), (0.16, 0.72), text="configured cadence + gate rules", rad=0.0)

    # Output
    box(ax, 0.26, 0.10, 0.48, 0.10, "Online Outputs", "online path + schedule log + diagnostics + update-benefit summary", fc="#EEDFFF", ec="#C8AFDE")
    arrow(ax, (0.74, 0.42), (0.50, 0.20))

    ax.text(0.03, 0.03, "Clean view: branch-and-join update logic + single t→t+1 loop", fontsize=9, color="#576474")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
