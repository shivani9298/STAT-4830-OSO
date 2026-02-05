"""
Online Portfolio Optimization Package.

This package provides tools for building and optimizing portfolios
using Online Gradient Descent between S&P 500 and IPO indices.
"""

from .model import OnlineOGDAllocator, project_to_simplex, max_drawdown_from_returns
from .utils import fetch_price_and_shares, build_ipo_index, calculate_metrics, run_backtest

__all__ = [
    'OnlineOGDAllocator',
    'project_to_simplex',
    'max_drawdown_from_returns',
    'fetch_price_and_shares',
    'build_ipo_index',
    'calculate_metrics',
    'run_backtest'
]
