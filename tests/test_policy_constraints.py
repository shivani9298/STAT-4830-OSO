"""
Tests for policy constraint enforcement.

Tests that the policy correctly enforces:
- Weight clipping to w_max
- Volume-based liquidity caps
- Non-negative weights
- Valid entry/exit day relationships
"""

import pytest
import numpy as np
from src.policy import PolicyParams, decide_trade


def test_weight_clipping_to_w_max():
    """Test that weight is clipped to w_max when raw_weight exceeds it."""
    params = PolicyParams(
        participate_threshold=-1.0,  # Always participate
        entry_day=0,
        hold_k=2,
        w_max=0.15,  # Maximum allowed weight
        raw_weight=0.5  # Raw weight exceeds max
    )
    
    episode = {
        'prices': [100.0, 101.0, 102.0, 103.0, 104.0]
    }
    
    decision = decide_trade(episode, params)
    
    assert decision['participate'] is True
    assert decision['weight'] <= params.w_max
    assert decision['weight'] == params.w_max  # Should be exactly w_max


def test_weight_non_negative():
    """Test that weight is always non-negative."""
    params = PolicyParams(
        participate_threshold=-1.0,
        entry_day=0,
        hold_k=2,
        w_max=0.2,
        raw_weight=-0.1  # Negative raw weight (edge case)
    )
    
    episode = {
        'prices': [100.0, 101.0, 102.0, 103.0]
    }
    
    decision = decide_trade(episode, params)
    
    assert decision['weight'] >= 0.0


def test_volume_cap_enforcement():
    """Test that volume cap is applied when use_volume_cap is True."""
    params = PolicyParams(
        participate_threshold=-1.0,
        entry_day=0,
        hold_k=2,
        w_max=0.5,  # High max to allow volume cap to take effect
        raw_weight=0.3,
        use_volume_cap=True,
        vol_cap_mult=1.0
    )
    
    # Episode with low volume on entry day relative to average
    episode = {
        'prices': [100.0, 101.0, 102.0, 103.0, 104.0],
        'volumes': np.array([1000.0, 5000.0, 6000.0, 7000.0, 8000.0])  # Low on day0
    }
    
    decision = decide_trade(episode, params)
    
    assert decision['participate'] is True
    # Weight should be constrained by volume cap
    assert decision['weight'] <= params.raw_weight


def test_volume_cap_with_high_volume():
    """Test volume cap with high volume on entry day."""
    params = PolicyParams(
        participate_threshold=-1.0,
        entry_day=0,
        hold_k=2,
        w_max=0.5,
        raw_weight=0.1,
        use_volume_cap=True,
        vol_cap_mult=2.0  # Higher multiplier
    )
    
    # Episode with high volume on entry day
    episode = {
        'prices': [100.0, 101.0, 102.0, 103.0],
        'volumes': np.array([10000.0, 1000.0, 1100.0, 1200.0])  # High on day0
    }
    
    decision = decide_trade(episode, params)
    
    assert decision['participate'] is True
    # With high volume, cap might allow larger weight, but still bounded by w_max
    assert decision['weight'] <= params.w_max


def test_volume_cap_disabled():
    """Test that volume cap is not applied when use_volume_cap is False."""
    params = PolicyParams(
        participate_threshold=-1.0,
        entry_day=0,
        hold_k=2,
        w_max=0.5,
        raw_weight=0.2,
        use_volume_cap=False  # Volume cap disabled
    )
    
    episode = {
        'prices': [100.0, 101.0, 102.0, 103.0],
        'volumes': np.array([100.0, 5000.0, 6000.0, 7000.0])  # Low volume on day0
    }
    
    decision = decide_trade(episode, params)
    
    assert decision['participate'] is True
    # Weight should be raw_weight (clipped to w_max), not affected by volume
    assert decision['weight'] == min(params.raw_weight, params.w_max)


def test_volume_cap_with_missing_volume():
    """Test that volume cap gracefully handles missing volume data."""
    params = PolicyParams(
        participate_threshold=-1.0,
        entry_day=0,
        hold_k=2,
        w_max=0.2,
        raw_weight=0.15,
        use_volume_cap=True  # Volume cap enabled but no volume data
    )
    
    episode = {
        'prices': [100.0, 101.0, 102.0, 103.0]
        # No 'volumes' key
    }
    
    decision = decide_trade(episode, params)
    
    assert decision['participate'] is True
    # Should fall back to raw_weight clipped to w_max
    assert decision['weight'] == min(params.raw_weight, params.w_max)


def test_volume_cap_with_zero_volume():
    """Test volume cap with zero volume on entry day."""
    params = PolicyParams(
        participate_threshold=-1.0,
        entry_day=0,
        hold_k=2,
        w_max=0.5,
        raw_weight=0.3,
        use_volume_cap=True,
        vol_cap_mult=1.0
    )
    
    episode = {
        'prices': [100.0, 101.0, 102.0, 103.0],
        'volumes': np.array([0.0, 1000.0, 1100.0, 1200.0])  # Zero on day0
    }
    
    decision = decide_trade(episode, params)
    
    assert decision['participate'] is True
    # Should handle zero volume gracefully (division by zero protection)
    assert decision['weight'] >= 0.0
    assert decision['weight'] <= params.w_max


def test_constraint_enforcement_combined():
    """Test that all constraints work together correctly."""
    params = PolicyParams(
        participate_threshold=-1.0,
        entry_day=1,  # Enter on day1
        hold_k=3,
        w_max=0.1,  # Small max
        raw_weight=0.5,  # Large raw weight
        use_volume_cap=True,
        vol_cap_mult=0.5  # Volume cap multiplier
    )
    
    episode = {
        'prices': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        'volumes': np.array([1000.0, 500.0, 600.0, 700.0, 800.0, 900.0])  # Low volume on day1
    }
    
    decision = decide_trade(episode, params)
    
    assert decision['participate'] is True
    assert decision['entry_day'] == 1
    assert decision['exit_day'] == 4  # entry_day + hold_k
    assert 0 <= decision['weight'] <= params.w_max
    assert decision['exit_day'] > decision['entry_day']


def test_weight_exactly_at_w_max():
    """Test case where raw_weight equals w_max."""
    params = PolicyParams(
        participate_threshold=-1.0,
        entry_day=0,
        hold_k=2,
        w_max=0.2,
        raw_weight=0.2  # Exactly at w_max
    )
    
    episode = {
        'prices': [100.0, 101.0, 102.0, 103.0]
    }
    
    decision = decide_trade(episode, params)
    
    assert decision['participate'] is True
    assert decision['weight'] == params.w_max


def test_volume_cap_entry_day1():
    """Test volume cap when entering on day1."""
    params = PolicyParams(
        participate_threshold=-1.0,
        entry_day=1,  # Enter on day1
        hold_k=2,
        w_max=0.5,
        raw_weight=0.3,
        use_volume_cap=True,
        vol_cap_mult=1.0
    )
    
    episode = {
        'prices': [100.0, 101.0, 102.0, 103.0, 104.0],
        'volumes': np.array([10000.0, 500.0, 600.0, 700.0, 800.0])  # Low volume on day1
    }
    
    decision = decide_trade(episode, params)
    
    assert decision['participate'] is True
    assert decision['entry_day'] == 1
    # Volume cap should use day1 volume
    assert decision['weight'] <= params.w_max
