"""
Balance metrics for multi-term portfolio loss: detect when one weighted term dominates.

Uses the same component definitions as :func:`src.losses.combined_loss` (validation batch averages).
"""
from __future__ import annotations

import math
from typing import Any

# Keys in ``validate`` / history rows prefixed with ``val_`` (excluding loss, lr, epoch).
CONTRIB_KEYS = (
    "mean_return",
    "mean_log1p_return",
    "cvar",
    "turnover",
    "volatility",
    "vol_excess",
    "weight_path",
    "diversify",
)


def _row_val_components(row: dict[str, Any]) -> dict[str, float]:
    """Extract val_* component dict from a history row."""
    out: dict[str, float] = {}
    for k in CONTRIB_KEYS:
        key = f"val_{k}"
        if key in row:
            out[k] = float(row[key])
    return out


def abs_weighted_contributions(cfg: dict[str, Any], val_components: dict[str, float]) -> dict[str, float]:
    """
    Absolute value of each term's contribution to total loss (same λ weighting as training).

    ``val_components`` uses raw L_* magnitudes from ``combined_loss`` (e.g. ``volatility`` = L_vol batch mean).
    """
    mw = float(cfg.get("mean_return_weight", 1.0))
    lg = float(cfg.get("log_growth_weight", 0.0))
    lc = float(cfg.get("lambda_cvar", 0.5))
    lt = float(cfg.get("lambda_turnover", 0.01))
    lv = float(cfg.get("lambda_vol", 0.5))
    lp = float(cfg.get("lambda_path", 0.01))
    lve = float(cfg.get("lambda_vol_excess", 0.0))
    ld = float(cfg.get("lambda_diversify", 0.0))

    mr = float(val_components.get("mean_return", 0.0))
    L_mean = -mr
    out: dict[str, float] = {}
    out["mean"] = abs(mw * L_mean)

    mlog = float(val_components.get("mean_log1p_return", 0.0))
    L_log = -mlog
    out["log_growth"] = abs(lg * L_log)

    cvar_v = float(val_components.get("cvar", 0.0))
    L_cvar = -cvar_v
    out["cvar"] = abs(lc * L_cvar)

    out["turnover"] = abs(lt * float(val_components.get("turnover", 0.0)))
    out["volatility"] = abs(lv * float(val_components.get("volatility", 0.0)))
    out["vol_excess"] = abs(lve * float(val_components.get("vol_excess", 0.0)))
    out["weight_path"] = abs(lp * float(val_components.get("weight_path", 0.0)))
    out["diversify"] = abs(ld * float(val_components.get("diversify", 0.0)))
    return out


def coefficient_of_variation_abs_contribs(contribs: dict[str, float], *, eps: float = 1e-12) -> float:
    """CV of non-negligible absolute contributions; 0 if fewer than 2 positive terms."""
    vals = [v for v in contribs.values() if v > eps]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    if m <= eps:
        return 0.0
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return float(math.sqrt(var) / m)


def history_row_at_best_val_loss(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Row where ``val_loss`` is minimal (first occurrence)."""
    if not history:
        return None
    best = min(history, key=lambda r: float(r.get("val_loss", float("inf"))))
    return best


def balance_metrics_for_config(cfg: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    """
    At best-val epoch: absolute weighted contributions + CV imbalance + optional composite score.

    Returns keys: ``abs_contributions``, ``imbalance_cv``, ``best_epoch``, ``val_loss_at_best``.
    """
    row = history_row_at_best_val_loss(history)
    if row is None:
        return {
            "abs_contributions": {},
            "imbalance_cv": float("nan"),
            "best_epoch": None,
            "val_loss_at_best": float("nan"),
        }
    vc = _row_val_components(row)
    ac = abs_weighted_contributions(cfg, vc)
    icv = coefficient_of_variation_abs_contribs(ac)
    return {
        "abs_contributions": ac,
        "imbalance_cv": icv,
        "best_epoch": int(row.get("epoch", -1)),
        "val_loss_at_best": float(row.get("val_loss", float("nan"))),
    }


def composite_tune_score(
    val_loss: float,
    imbalance_cv: float,
    *,
    balance_weight: float,
) -> float:
    """Lower is better: val loss + penalty * CV of absolute weighted contributions."""
    return float(val_loss) + float(balance_weight) * float(imbalance_cv)
