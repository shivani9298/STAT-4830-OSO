"""
Metrics module - computes performance metrics (CVaR, MDD, turnover, etc.).
"""

from typing import Union, Dict
import numpy as np
import pandas as pd


def cvar(returns: Union[np.ndarray, pd.Series], alpha: float = 0.9) -> float:
    """
    Compute Conditional Value at Risk (CVaR) on losses.
    
    CVaR is the expected value of losses given that losses exceed VaR.
    
    Args:
        returns: Array/Series of net returns
        alpha: Confidence level (e.g., 0.9 for 90%)
        
    Returns:
        CVaR value (positive number representing expected loss)
    """
    if len(returns) == 0:
        return 0.0
    
    # Convert to numpy array
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Compute losses (negative returns)
    losses = -returns
    
    # VaR is the alpha quantile of losses
    var = np.quantile(losses, alpha)
    
    # CVaR is the mean of losses that exceed VaR
    tail_losses = losses[losses >= var]
    
    if len(tail_losses) == 0:
        return 0.0
    
    cvar_value = np.mean(tail_losses)
    
    return float(cvar_value)


def max_drawdown(equity_curve: Union[np.ndarray, pd.Series]) -> float:
    """
    Compute maximum drawdown from equity curve.
    
    Args:
        equity_curve: Array/Series of portfolio equity values
        
    Returns:
        Maximum drawdown (positive number, e.g., 0.2 for 20% drawdown)
    """
    if len(equity_curve) == 0:
        return 0.0
    
    # Convert to numpy array
    if isinstance(equity_curve, pd.Series):
        equity_curve = equity_curve.values
    
    # Compute running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Drawdown at each point
    drawdowns = (running_max - equity_curve) / running_max
    
    # Maximum drawdown
    mdd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    return float(mdd)


def summarize_costs(results_df: pd.DataFrame) -> Dict:
    """
    Summarize trading costs.
    
    Args:
        results_df: Results DataFrame from backtest_all
        
    Returns:
        Dict with cost summary statistics
    """
    if len(results_df) == 0:
        return {
            "total_cost": 0.0,
            "avg_cost_per_trade": 0.0,
            "total_trades": 0
        }
    
    participating = results_df[results_df["weight"] > 0]
    
    return {
        "total_cost": float(participating["cost"].sum()),
        "avg_cost_per_trade": float(participating["cost"].mean()) if len(participating) > 0 else 0.0,
        "total_trades": len(participating)
    }


def turnover(results_df: pd.DataFrame) -> float:
    """
    Compute turnover (sum of absolute weights for participating trades).
    
    Args:
        results_df: Results DataFrame from backtest_all
        
    Returns:
        Total turnover
    """
    if len(results_df) == 0:
        return 0.0
    
    participating = results_df[results_df["weight"] > 0]
    
    return float(participating["weight"].abs().sum())
