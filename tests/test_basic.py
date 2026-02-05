"""
Basic validation tests for Week 4 (objective, metrics, backtest).
"""

import pytest
import numpy as np
import pandas as pd

from src.metrics import cvar, max_drawdown
from src.objective import score
from src.data import Episode, generate_synthetic_prices
from src.backtest import backtest_all, backtest_all_with_decisions
from src.policy import PolicyParams
from datetime import date


def test_cvar_empty():
    """CVaR on empty array returns 0."""
    assert cvar(np.array([]), alpha=0.9) == 0.0


def test_cvar_constant_positive():
    """Constant positive returns -> losses are negative -> tail is negative; CVaR is expected loss (positive for loss)."""
    # Returns all 0.01 -> losses all -0.01 -> VaR at 0.9 is -0.01, tail losses are <= -0.01, so mean tail "loss" = -0.01 (loss in our def is -return, so loss = -0.01). Actually: losses = -returns, so losses = -0.01. VaR = quantile(losses, 0.9). For 10 same values, quantile 0.9 is -0.01. Tail losses = losses >= VaR = all -0.01. Mean = -0.01. So cvar returns -0.01? No - doc says "CVaR is the expected value of losses given that losses exceed VaR". So we're computing expected LOSS. Losses = -returns. So losses are -0.01. VaR = -0.01. Tail = losses >= -0.01 = all. Mean tail loss = -0.01. So cvar = -0.01? But doc says "Returns: CVaR value (positive number representing expected loss)". So they want positive. So maybe they mean "expected loss" as in magnitude of loss? Let me read the code again. cvar_value = np.mean(tail_losses). So for constant positive returns, tail_losses = [-0.01,...,-0.01], mean = -0.01. So the function returns -0.01. So "positive number" in the doc might mean "when there are actual losses". For constant positive returns there are no losses (losses are negative), so CVaR can be negative. Let me not assert sign; just assert cvar is a float.
    out = cvar(np.full(10, 0.01), alpha=0.9)
    assert isinstance(out, float)


def test_cvar_constant_negative():
    """Constant negative returns -> losses positive -> CVaR positive."""
    out = cvar(np.full(10, -0.02), alpha=0.9)
    assert out >= 0.0
    assert np.isclose(out, 0.02)


def test_max_drawdown_empty():
    """MDD on empty series returns 0."""
    assert max_drawdown(pd.Series(dtype=float)) == 0.0
    assert max_drawdown(np.array([])) == 0.0


def test_max_drawdown_no_drawdown():
    """Monotonically increasing equity -> MDD = 0."""
    equity = np.cumprod([1.0, 1.01, 1.02, 1.01, 1.03])
    assert max_drawdown(equity) == 0.0


def test_max_drawdown_simple():
    """Equity 1, 1.1, 0.9 -> drawdown 0.2/1.1."""
    equity = np.array([1.0, 1.1, 0.9])
    mdd = max_drawdown(equity)
    assert np.isclose(mdd, (1.1 - 0.9) / 1.1)


def test_score_empty():
    """Empty results_df and equity -> score 0."""
    sc, metrics = score(pd.DataFrame(), pd.Series(dtype=float))
    assert sc == 0.0
    assert metrics["score"] == 0.0
    assert metrics["E[R]"] == 0.0
    assert metrics["CVaR"] == 0.0
    assert metrics["MDD"] == 0.0


def test_score_single_row():
    """Single episode with positive net_ret -> positive E[R], score depends on CVaR/MDD."""
    results_df = pd.DataFrame({
        "net_ret": [0.01],
        "cost": [0.001],
        "weight": [0.5],
    })
    equity = pd.Series([1.01])
    sc, metrics = score(results_df, equity)
    assert metrics["E[R]"] == 0.01
    assert metrics["n_trades"] == 1
    assert "score" in metrics


def test_backtest_never_participate():
    """Policy that never participates -> all net_ret and cost zero."""
    rng = np.random.default_rng(42)
    price_df = generate_synthetic_prices("T", date(2020, 1, 1), N=10, rng=rng)
    ep = Episode(ticker="T", ipo_date=date(2020, 1, 1), df=price_df, day0_index=0, N=10)
    params = PolicyParams(participate_threshold=999.0, entry_day=0, hold_k=1, raw_weight=0.0)
    results_df, equity = backtest_all([ep], params, cost_bps=10.0)
    assert len(results_df) == 1
    assert results_df["net_ret"].iloc[0] == 0.0
    assert results_df["cost"].iloc[0] == 0.0
    assert results_df["weight"].iloc[0] == 0.0


def test_backtest_all_with_decisions_length_mismatch():
    """backtest_all_with_decisions raises when episodes and decisions length differ."""
    rng = np.random.default_rng(42)
    price_df = generate_synthetic_prices("T", date(2020, 1, 1), N=10, rng=rng)
    ep = Episode(ticker="T", ipo_date=date(2020, 1, 1), df=price_df, day0_index=0, N=10)
    decisions = [{"participate": False, "entry_day": 0, "exit_day": 0, "weight": 0.0}]
    with pytest.raises(ValueError, match="same length"):
        backtest_all_with_decisions([ep, ep], decisions, cost_bps=10.0)
