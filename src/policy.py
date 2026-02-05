"""
Policy module - defines trading decision logic.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd

from src.data import Episode


def _safe_float_meta(meta: Optional[Dict], key: str, default: float = 0.0) -> float:
    if not meta:
        return default
    v = meta.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


@dataclass
class PolicyParams:
    """Policy parameters for trading decisions."""
    participate_threshold: float = 0.5  # Threshold for participation decision
    entry_day: int = 0  # Entry day (0 = day0, 1 = day1, etc.)
    hold_k: int = 1  # Hold for k days after entry
    w_max: float = 1.0  # Maximum position weight
    raw_weight: float = 0.1  # Base position size
    use_volume_cap: bool = False  # Whether to use volume-based position capping
    vol_cap_mult: float = 0.1  # Volume cap multiplier (position <= vol_cap_mult * daily_volume)
    # Meta-based filters (from episode.meta; only applied when meta is present)
    min_offer_amount: Optional[float] = None  # Participate only if Offer Amount >= this (e.g. 1e6)
    max_offer_amount: Optional[float] = None  # Participate only if Offer Amount <= this (None = no cap)


def sample_params(rng: np.random.Generator, max_hold: int = 252) -> PolicyParams:
    """
    Sample random policy parameters for optimization.

    Args:
        rng: Random number generator
        max_hold: Maximum hold period (default 252 = 1 year of trading days)

    Returns:
        PolicyParams with random values
    """
    return PolicyParams(
        participate_threshold=rng.uniform(0.0, 0.1),  # Lower threshold - more likely to participate
        entry_day=rng.integers(0, 3),  # 0, 1, or 2
        hold_k=rng.integers(1, max_hold + 1),  # 1 to max_hold days
        w_max=rng.uniform(0.5, 1.0),
        raw_weight=rng.uniform(0.1, 1.0),
        use_volume_cap=False,
        vol_cap_mult=0.1,
        min_offer_amount=None,
        max_offer_amount=None,
    )


def decide_trade(episode: Episode, params: PolicyParams) -> Dict:
    """
    Decide whether to trade and return decision dict.
    
    Returns a dict with keys:
        - participate: bool
        - entry_day: int (0..N-1)
        - exit_day: int (entry_day+1..N)
        - weight: float (0..w_max)
    
    Args:
        episode: Trading episode
        params: Policy parameters
        
    Returns:
        Decision dictionary
    """
    # Handle empty or too-short episodes
    if len(episode.df) < 2:
        return {
            "participate": False,
            "entry_day": 0,
            "exit_day": 0,
            "weight": 0.0
        }
    
    # Meta-based filters (when episode.meta is available, e.g. from rich CSV)
    if episode.meta:
        offer_amount = _safe_float_meta(episode.meta, "Offer Amount", 0.0)
        if params.min_offer_amount is not None and offer_amount < params.min_offer_amount:
            return {
                "participate": False,
                "entry_day": 0,
                "exit_day": 0,
                "weight": 0.0
            }
        if params.max_offer_amount is not None and offer_amount > params.max_offer_amount:
            return {
                "participate": False,
                "entry_day": 0,
                "exit_day": 0,
                "weight": 0.0
            }
    
    # Participation rule: check early excess return signal
    # If benchmark data available, use excess return; otherwise use price change
    day0_close = episode.df.iloc[0]['close']

    # Look at first few days to decide participation
    lookback = min(5, len(episode.df) - 1)
    if lookback < 1:
        lookback = 1

    lookback_close = episode.df.iloc[lookback]['close']
    stock_change = (lookback_close - day0_close) / day0_close if day0_close > 0 else 0.0

    # Check for benchmark (excess return signal)
    if 'benchmark_close' in episode.df.columns:
        bench_day0 = episode.df.iloc[0]['benchmark_close']
        bench_lookback = episode.df.iloc[lookback]['benchmark_close']
        bench_change = (bench_lookback - bench_day0) / bench_day0 if bench_day0 > 0 else 0.0
        signal = stock_change - bench_change  # excess return signal
    else:
        signal = stock_change

    # Participate if signal exceeds threshold (or if threshold is very low, almost always participate)
    participate = signal >= params.participate_threshold or params.participate_threshold < 0.01
    
    if not participate:
        return {
            "participate": False,
            "entry_day": 0,
            "exit_day": 0,
            "weight": 0.0
        }
    
    # Determine entry day
    entry_day = min(params.entry_day, len(episode.df) - 1)
    
    # Determine exit day (entry_day + hold_k, but not beyond episode length)
    exit_day = min(entry_day + params.hold_k, len(episode.df) - 1)
    
    # Ensure exit_day > entry_day
    if exit_day <= entry_day:
        exit_day = min(entry_day + 1, len(episode.df) - 1)
    
    # Calculate weight
    weight = params.raw_weight
    
    # Apply volume cap if enabled and volume data available
    if params.use_volume_cap and 'volume' in episode.df.columns:
        entry_volume = episode.df.iloc[entry_day]['volume']
        if not pd.isna(entry_volume) and entry_volume > 0:
            # Cap weight based on volume
            vol_cap = params.vol_cap_mult * entry_volume / (day0_close * entry_volume) if day0_close > 0 else weight
            weight = min(weight, vol_cap)
    
    # Clip weight to [0, w_max]
    weight = max(0.0, min(weight, params.w_max))
    
    return {
        "participate": True,
        "entry_day": entry_day,
        "exit_day": exit_day,
        "weight": weight
    }
