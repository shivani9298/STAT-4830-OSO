"""
Policy module for IPO trading decisions.

This module implements a simple rule-based policy for IPO participation decisions.
The policy decides whether to participate, when to enter, when to exit, and position sizing.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
import numpy as np


@dataclass
class PolicyParams:
    """
    Parameters for the IPO trading policy.
    
    Attributes:
        participate_threshold: Threshold value for participation decision (float)
        entry_day: Day to enter position (0 = day0 close, 1 = day1 close)
        hold_k: Number of days to hold position (int in [1..N])
        w_max: Maximum position weight (float, typically in [0, 1])
        raw_weight: Raw position sizing knob (float)
        use_volume_cap: Whether to apply volume-based liquidity cap (bool)
        vol_cap_mult: Multiplier for volume cap calculation (float)
    """
    participate_threshold: float
    entry_day: int
    hold_k: int
    w_max: float
    raw_weight: float
    use_volume_cap: bool = False
    vol_cap_mult: float = 1.0


def sample_params(rng: np.random.Generator) -> PolicyParams:
    """
    Sample random policy parameters for optimization.
    
    Args:
        rng: NumPy random number generator for reproducibility
        
    Returns:
        PolicyParams: Randomly sampled policy parameters
        
    Example:
        >>> rng = np.random.default_rng(0)
        >>> params = sample_params(rng)
        >>> isinstance(params, PolicyParams)
        True
    """
    # Sample participate_threshold from reasonable range (e.g., -0.1 to 0.1 for price change)
    participate_threshold = rng.uniform(-0.1, 0.1)
    
    # Entry day: 0 (day0 close) or 1 (day1 close)
    entry_day = rng.choice([0, 1])
    
    # Hold period: 1 to 10 days (typical range for IPO episodes)
    hold_k = rng.integers(1, 11)
    
    # Maximum weight: reasonable range for position sizing
    w_max = rng.uniform(0.05, 0.25)
    
    # Raw weight: can be larger than w_max, will be clipped
    raw_weight = rng.uniform(0.01, 0.5)
    
    # Volume cap settings (optional feature)
    use_volume_cap = rng.choice([True, False])
    vol_cap_mult = rng.uniform(0.5, 2.0)
    
    return PolicyParams(
        participate_threshold=participate_threshold,
        entry_day=entry_day,
        hold_k=hold_k,
        w_max=w_max,
        raw_weight=raw_weight,
        use_volume_cap=use_volume_cap,
        vol_cap_mult=vol_cap_mult
    )


def decide_trade(
    episode: Dict[str, Any],
    params: PolicyParams
) -> Dict[str, Any]:
    """
    Decide whether to trade an IPO episode and return trade decision.
    
    The decision is based on:
    - Participation rule using day0/day1 information
    - Entry day (day0 or day1 close)
    - Exit day (entry_day + hold_k)
    - Position weight (clipped to constraints)
    
    Args:
        episode: Dictionary containing episode data with keys:
            - 'prices': List or array of daily close prices (indexed by day relative to IPO)
            - 'volumes': Optional list/array of daily volumes
            - 'ticker': Optional ticker symbol
            - 'ipo_date': Optional IPO date
        params: Policy parameters
        
    Returns:
        Dictionary with keys:
            - 'participate': bool, whether to participate
            - 'entry_day': int, day to enter (relative to IPO day0)
            - 'exit_day': int, day to exit (relative to IPO day0)
            - 'weight': float, position weight (0 if not participating)
            
    Constraints enforced:
        - 0 <= weight <= w_max
        - exit_day > entry_day
        - Handles empty/too-short episodes gracefully (participate=False)
    """
    # Extract prices from episode
    prices = episode.get('prices', [])
    
    # Convert to numpy array if needed (do this before length checks to handle numpy arrays properly)
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices)
    
    # Handle empty or too-short episodes
    if len(prices) == 0 or len(prices) < 2:
        return {
            'participate': False,
            'entry_day': params.entry_day,
            'exit_day': params.entry_day + params.hold_k,
            'weight': 0.0
        }
    
    # Ensure we have at least day0 and day1 prices
    if len(prices) <= params.entry_day:
        return {
            'participate': False,
            'entry_day': params.entry_day,
            'exit_day': params.entry_day + params.hold_k,
            'weight': 0.0
        }
    
    # Participation rule: compare day0 to day1 price change
    # Simple rule: participate if day1 price change exceeds threshold
    if len(prices) >= 2:
        day0_price = prices[0]
        day1_price = prices[1] if len(prices) > 1 else day0_price
        price_change = (day1_price - day0_price) / day0_price if day0_price > 0 else 0.0
        
        # Participate if price change exceeds threshold
        should_participate = price_change >= params.participate_threshold
    else:
        should_participate = False
    
    # Calculate exit day
    exit_day = params.entry_day + params.hold_k
    
    # Check if we have enough data for the full holding period
    if exit_day >= len(prices):
        # Not enough data for full holding period
        should_participate = False
    
    if not should_participate:
        return {
            'participate': False,
            'entry_day': params.entry_day,
            'exit_day': exit_day,
            'weight': 0.0
        }
    
    # Calculate position weight
    weight = params.raw_weight
    
    # Apply maximum weight constraint
    weight = min(weight, params.w_max)
    
    # Apply volume cap if enabled and volume data exists
    if params.use_volume_cap:
        volumes = episode.get('volumes', None)
        if volumes is not None:
            if not isinstance(volumes, np.ndarray):
                volumes = np.array(volumes)
            
            # Use entry day volume for liquidity cap
            if params.entry_day < len(volumes) and volumes[params.entry_day] > 0:
                # Simple liquidity cap: weight proportional to volume
                # Normalize by some reference volume (e.g., average volume)
                avg_volume = np.mean(volumes[volumes > 0]) if np.any(volumes > 0) else volumes[params.entry_day]
                if avg_volume > 0:
                    vol_cap = params.vol_cap_mult * (volumes[params.entry_day] / avg_volume)
                    weight = min(weight, vol_cap)
    
    # Ensure weight is non-negative
    weight = max(0.0, weight)
    
    # Final constraint: ensure weight doesn't exceed w_max
    weight = min(weight, params.w_max)
    
    return {
        'participate': True,
        'entry_day': params.entry_day,
        'exit_day': exit_day,
        'weight': weight
    }
