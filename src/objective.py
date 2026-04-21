"""
Objective module - computes objective function scores.
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

    Default constants (lam, alpha, kappa, mu) are conventional choices, not estimated
    from data. You can override them via run_week3.py / run_pytorch.py (--lam, --alpha,
    --kappa, --mu) or when calling score().

    - lam (λ = 1.0): CVaR penalty weight. One unit of CVaR (tail loss) subtracts one
      unit of "return." Puts risk and return on the same scale; common in mean–CVaR
      optimization to use λ in [0.5, 2] (1 = balanced).
    - alpha (α = 0.9): CVaR confidence level. 0.9 = "expected loss in the worst 10% of
      outcomes." Standard in risk management (90% or 95% are typical).
    - kappa (κ = 1.0): Cost penalty weight. One unit of E[Cost] subtracts one unit of
      return; cost is in decimal (e.g. 0.0001 = 1 bps), so this scales cost in return space.
    - mu (μ = 1.0): Maximum-drawdown penalty weight. One unit of MDD (e.g. 0.1 = 10%
      drawdown) subtracts 0.1 from the score, so 10% drawdown is penalized like losing
      10% return.

    Args:
        results_df: Results DataFrame from backtest_all
        equity: Equity curve Series from backtest_all
        lam: CVaR penalty weight (default 1.0)
        alpha: CVaR confidence level (default 0.9)
        kappa: Cost penalty weight (default 1.0)
        mu: Maximum drawdown penalty weight (default 1.0)

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
    
    # E[R]: Expected return per opportunity (mean of net returns over all episodes)
    net_returns = results_df["net_ret"].values
    E_R = np.mean(net_returns)
    
    # E[R] per executed trade (mean over episodes where we actually traded)
    participated = results_df["weight"].values > 0 if "weight" in results_df.columns else np.ones(len(results_df), dtype=bool)
    n_trades = int(np.sum(participated))
    if n_trades > 0:
        E_R_executed = float(np.mean(net_returns[participated]))
    else:
        E_R_executed = 0.0
    
    # CVaR_alpha: Conditional Value at Risk
    cvar_value = cvar(net_returns, alpha)
    
    # E[Cost]: Expected cost (mean of costs)
    costs = results_df["cost"].values
    E_Cost = np.mean(costs)
    
    # MDD: Maximum drawdown
    mdd = max_drawdown(equity)
    
    # Compute score (uses per-opportunity E[R] so baselines are comparable)
    score_value = E_R - lam * cvar_value - kappa * E_Cost - mu * mdd
    
    metrics = {
        "E[R]": float(E_R),
        "E[R]_per_trade": float(E_R_executed),
        "n_trades": n_trades,
        "CVaR": float(cvar_value),
        "E[Cost]": float(E_Cost),
        "MDD": float(mdd),
        "score": float(score_value)
    }
    
    return float(score_value), metrics
