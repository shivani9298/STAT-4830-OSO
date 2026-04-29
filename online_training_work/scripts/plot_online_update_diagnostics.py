#!/usr/bin/env python3
"""
Diagnostics for online updates:
1) Per-update validation loss change vs subsequent k-day net return (scatter)
2) Update vs no-update subsequent return distributions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--online-path",
        type=Path,
        default=ROOT / "results" / "ipo_optimizer_online_path.csv",
    )
    p.add_argument(
        "--history-online",
        type=Path,
        default=ROOT / "results" / "training_history_online.csv",
    )
    p.add_argument("--horizon", type=int, default=5, help="Post-update horizon in days")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "figures" / "online_evaluation",
    )
    return p.parse_args()


def compute_post_horizon_returns(net_ret: np.ndarray, horizon: int) -> np.ndarray:
    r = np.asarray(net_ret, dtype=float)
    h = max(1, int(horizon))
    out = np.full((len(r),), np.nan, dtype=float)
    for i in range(len(r)):
        seg = r[i + 1 : i + 1 + h]
        if len(seg) == 0:
            continue
        out[i] = np.prod(1.0 + seg) - 1.0
    return out


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = pd.read_csv(args.online_path, parse_dates=["date"])
    hist = pd.read_csv(args.history_online)

    # One row per update step from training history.
    g = hist.sort_values(["online_step", "epoch"]).groupby("online_step")
    step = pd.DataFrame(
        {
            "online_step": g.size().index.astype(int),
            "first_val_loss": g["val_loss"].first().values,
            "last_val_loss": g["val_loss"].last().values,
        }
    )
    step["delta_val_loss"] = step["last_val_loss"] - step["first_val_loss"]

    # Map update events in the online path to online_step order.
    upd_idx = path.index[path["was_model_updated"].astype(int) == 1].to_numpy()
    postk = compute_post_horizon_returns(path["net_ret"].to_numpy(dtype=float), args.horizon)
    update_rows = min(len(upd_idx), len(step))
    step = step.iloc[:update_rows].copy()
    step["path_row"] = upd_idx[:update_rows]
    step["postk_net_return"] = postk[step["path_row"].to_numpy(dtype=int)]

    # Scatter: val-loss change vs post-update return.
    x = step["delta_val_loss"].to_numpy(dtype=float)
    y = step["postk_net_return"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x[mask], y[mask] * 100, alpha=0.75, s=24, color="#1f77b4")
    if mask.sum() >= 2:
        b1, b0 = np.polyfit(x[mask], y[mask], 1)
        xx = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xx, (b1 * xx + b0) * 100, color="#d62728", lw=1.5, label="Linear fit")
        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        ax.set_title(f"Per-update delta val-loss vs {args.horizon}d future net return (corr={corr:.2f})")
        ax.legend(fontsize=9)
    else:
        ax.set_title(f"Per-update delta val-loss vs {args.horizon}d future net return")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.axvline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("Delta val loss (last - first epoch)")
    ax.set_ylabel(f"Future {args.horizon}d net return (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = out_dir / "online_update_val_delta_vs_future_return.png"
    fig.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Distribution: update day vs no-update day post-horizon returns.
    post_all = compute_post_horizon_returns(path["net_ret"].to_numpy(dtype=float), args.horizon)
    upd_mask = path["was_model_updated"].astype(int).to_numpy() == 1
    a = post_all[upd_mask]
    b = post_all[~upd_mask]
    a = a[np.isfinite(a)] * 100
    b = b[np.isfinite(b)] * 100

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(a, bins=25, alpha=0.6, color="#1f77b4", label="After update")
    ax2.hist(b, bins=25, alpha=0.4, color="#ff7f0e", label="After no-update")
    ax2.axvline(np.mean(a) if len(a) else 0, color="#1f77b4", lw=1.5, ls="--")
    ax2.axvline(np.mean(b) if len(b) else 0, color="#ff7f0e", lw=1.5, ls="--")
    ax2.set_title(f"Distribution of future {args.horizon}d net returns")
    ax2.set_xlabel("Future net return (%)")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = out_dir / "online_update_vs_no_update_future_return_hist.png"
    fig2.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close(fig2)

    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
