"""Tests for loss term balance metrics."""
import math

from src.loss_balance import (
    abs_weighted_contributions,
    coefficient_of_variation_abs_contribs,
    composite_tune_score,
)


def test_cv_balanced_vs_imbalanced():
    balanced = {"a": 1.0, "b": 1.1, "c": 0.9}
    imbalanced = {"a": 10.0, "b": 0.1, "c": 0.1}
    assert coefficient_of_variation_abs_contribs(balanced) < coefficient_of_variation_abs_contribs(imbalanced)


def test_composite_score():
    s = composite_tune_score(0.05, 0.5, balance_weight=0.1)
    assert math.isclose(s, 0.05 + 0.1 * 0.5)


def test_abs_weighted_contributions():
    cfg = {
        "mean_return_weight": 1.0,
        "log_growth_weight": 0.0,
        "lambda_cvar": 0.5,
        "lambda_turnover": 0.01,
        "lambda_vol": 0.5,
        "lambda_path": 0.01,
        "lambda_vol_excess": 1.0,
        "lambda_diversify": 0.0,
    }
    vc = {
        "mean_return": 0.001,
        "mean_log1p_return": 0.0,
        "cvar": -0.01,
        "turnover": 0.1,
        "volatility": 0.0004,
        "vol_excess": 0.0,
        "weight_path": 0.0,
        "diversify": 0.0,
    }
    ac = abs_weighted_contributions(cfg, vc)
    assert "mean" in ac and "cvar" in ac
    assert all(v >= 0 for v in ac.values())
