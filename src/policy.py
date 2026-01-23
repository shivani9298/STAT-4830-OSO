"""
Policy module - defines trading decision logic.
Owned by Person B.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd

from src.data import Episode


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


def sample_params(rng: np.random.Generator) -> PolicyParams:
    """
    Sample random policy parameters for optimization.
    
    Args:
        rng: Random number generator
        
    Returns:
        PolicyParams with random values
    """
    return PolicyParams(
        participate_threshold=rng.uniform(0.0, 1.0),
        entry_day=rng.integers(0, 2),  # 0 or 1
        hold_k=rng.integers(1, 10),  # 1 to 9 days
        w_max=rng.uniform(0.1, 1.0),
        raw_weight=rng.uniform(0.05, 0.5),
        use_volume_cap=rng.choice([True, False]),
        vol_cap_mult=rng.uniform(0.05, 0.2)
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
    
    # Simple participation rule: check if day0/day1 price movement meets threshold
    # Use day0 close vs day1 close (or day0 close if only one day available)
    day0_close = episode.df.iloc[0]['close']
    
    if len(episode.df) > 1:
        day1_close = episode.df.iloc[1]['close']
        price_change = (day1_close - day0_close) / day0_close
    else:
        price_change = 0.0
    
    # Participate if price change exceeds threshold (simple rule)
    participate = abs(price_change) >= params.participate_threshold
    
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
