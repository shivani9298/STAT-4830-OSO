"""
Objective module - computes objective function scores.
Owned by Person D.
"""

from typing import Dict
import pandas as pd
import numpy as np

from src.metrics import cvar, max_drawdown, summarize_costs


def score(
    results_df: pd.DataFrame,
    equity: pd.Series,
    lam: float = 1.0,
    alpha: float = 0.9,
    kappa: float = 1.0,
    mu: float = 1.0
) -> tuple:
    """
    Compute objective function score.
    
    Score = E[R] - λ·CVaR_α - κ·E[Cost] - μ·MDD
    
    Args:
        results_df: Results DataFrame from backtest_all
        equity: Equity curve Series from backtest_all
        lam: CVaR penalty weight
        alpha: CVaR confidence level
        kappa: Cost penalty weight
        mu: Maximum drawdown penalty weight
        
    Returns:
        Tuple of (scalar_score, metrics_dict)
    """
    if len(results_df) == 0:
        return 0.0, {
            "E[R]": 0.0,
            "CVaR": 0.0,
            "E[Cost]": 0.0,
            "MDD": 0.0,
            "score": 0.0
        }
    
    # E[R]: Expected return (mean of net returns)
    net_returns = results_df["net_ret"].values
    E_R = np.mean(net_returns)
    
    # CVaR_alpha: Conditional Value at Risk
    cvar_value = cvar(net_returns, alpha)
    
    # E[Cost]: Expected cost (mean of costs)
    costs = results_df["cost"].values
    E_Cost = np.mean(costs)
    
    # MDD: Maximum drawdown
    mdd = max_drawdown(equity)
    
    # Compute score
    score_value = E_R - lam * cvar_value - kappa * E_Cost - mu * mdd
    
    metrics = {
        "E[R]": float(E_R),
        "CVaR": float(cvar_value),
        "E[Cost]": float(E_Cost),
        "MDD": float(mdd),
        "score": float(score_value)
    }
    
    return float(score_value), metrics
