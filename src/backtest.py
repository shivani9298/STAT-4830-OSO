"""
Backtest module - simulates trading strategy execution.
"""

from typing import Dict, List
import pandas as pd
import numpy as np

from src.data import Episode
from src.policy import PolicyParams, decide_trade


def backtest_episode(
    episode: Episode,
    decision: Dict,
    cost_bps: float = 10.0
) -> Dict:
    """
    Backtest a single episode.

    Args:
        episode: Trading episode
        decision: Decision dict from decide_trade
        cost_bps: Transaction cost in basis points (1e-4)

    Returns:
        Dict with keys: ticker, entry_day, exit_day, weight, entry_px, exit_px,
        gross_ret, benchmark_ret, excess_ret, cost, net_ret, pnl
    """
    if not decision["participate"]:
        return {
            "ticker": episode.ticker,
            "entry_day": 0,
            "exit_day": 0,
            "weight": 0.0,
            "entry_px": 0.0,
            "exit_px": 0.0,
            "gross_ret": 0.0,
            "benchmark_ret": 0.0,
            "excess_ret": 0.0,
            "cost": 0.0,
            "net_ret": 0.0,
            "pnl": 0.0
        }

    entry_day = decision["entry_day"]
    exit_day = decision["exit_day"]
    weight = decision["weight"]

    # Get stock prices
    entry_px = episode.df.iloc[entry_day]['close']
    exit_px = episode.df.iloc[exit_day]['close']

    # Calculate stock return
    if entry_px > 0:
        gross_ret = (exit_px / entry_px) - 1.0
    else:
        gross_ret = 0.0

    # Calculate benchmark return if available
    benchmark_ret = 0.0
    if 'benchmark_close' in episode.df.columns:
        bench_entry = episode.df.iloc[entry_day]['benchmark_close']
        bench_exit = episode.df.iloc[exit_day]['benchmark_close']
        if bench_entry > 0:
            benchmark_ret = (bench_exit / bench_entry) - 1.0

    # Excess return = stock return - benchmark return
    excess_ret = gross_ret - benchmark_ret

    # Calculate cost (simple: cost_bps/1e4 * abs(weight))
    cost = (cost_bps / 1e4) * abs(weight)

    # Net return: weight * excess_ret - cost (now using EXCESS return)
    net_ret = weight * excess_ret - cost

    # PnL (if equity starts at 1 per episode)
    pnl = net_ret

    return {
        "ticker": episode.ticker,
        "entry_day": entry_day,
        "exit_day": exit_day,
        "weight": weight,
        "entry_px": entry_px,
        "exit_px": exit_px,
        "gross_ret": gross_ret,
        "benchmark_ret": benchmark_ret,
        "excess_ret": excess_ret,
        "cost": cost,
        "net_ret": net_ret,
        "pnl": pnl
    }


def backtest_all_with_decisions(
    episodes: List[Episode],
    decisions: List[Dict],
    cost_bps: float = 10.0
) -> tuple:
    """
    Backtest using explicit decision dicts (e.g. from policy network).
    Returns (results_df, equity_curve) same as backtest_all.
    """
    if len(episodes) != len(decisions):
        raise ValueError("episodes and decisions must have same length")
    results = []
    equity_curve = [1.0]
    for episode, decision in zip(episodes, decisions):
        result = backtest_episode(episode, decision, cost_bps)
        results.append(result)
        new_equity = equity_curve[-1] * (1 + result["net_ret"])
        equity_curve.append(new_equity)
    results_df = pd.DataFrame(results)
    equity_series = pd.Series(equity_curve[1:])
    return results_df, equity_series


def backtest_all(
    episodes: List[Episode],
    params: PolicyParams,
    cost_bps: float = 10.0
) -> tuple:
    """
    Backtest all episodes.
    
    Args:
        episodes: List of trading episodes
        params: Policy parameters
        cost_bps: Transaction cost in basis points
        
    Returns:
        Tuple of (results_df, equity_curve)
        - results_df: DataFrame with columns: ticker, entry_day, exit_day, weight,
          entry_px, exit_px, gross_ret, cost, net_ret, pnl
        - equity_curve: 1D array/Series of portfolio equity over IPO episodes
    """
    results = []
    equity_curve = [1.0]  # Start with equity = 1
    
    for episode in episodes:
        decision = decide_trade(episode, params)
        result = backtest_episode(episode, decision, cost_bps)
        results.append(result)
        
        # Update equity curve (cumulative product of (1 + net_ret))
        if len(equity_curve) > 0:
            new_equity = equity_curve[-1] * (1 + result["net_ret"])
            equity_curve.append(new_equity)
        else:
            equity_curve.append(1.0 + result["net_ret"])
    
    results_df = pd.DataFrame(results)
    equity_series = pd.Series(equity_curve[1:])  # Remove initial 1.0
    
    return results_df, equity_series
