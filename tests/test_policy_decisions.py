"""
Tests for policy decision logic.

Tests the decide_trade function to ensure it correctly:
- Makes participation decisions based on thresholds
- Handles edge cases (empty episodes, short episodes)
- Returns valid entry/exit days
- Enforces weight constraints
"""

import pytest
import numpy as np
from src.policy import PolicyParams, decide_trade


def test_decide_trade_participates_when_threshold_met():
    """Test that policy participates when price change exceeds threshold."""
    params = PolicyParams(
        participate_threshold=0.05,  # 5% threshold
        entry_day=0,
        hold_k=3,
        w_max=0.2,
        raw_weight=0.15
    )
    
    # Episode with 10% price increase from day0 to day1
    episode = {
        'prices': [100.0, 110.0, 115.0, 120.0, 125.0],
        'ticker': 'TEST'
    }
    
    decision = decide_trade(episode, params)
    
    assert decision['participate'] is True
    assert decision['entry_day'] == 0
    assert decision['exit_day'] == 3
    assert decision['weight'] > 0
    assert decision['weight'] <= params.w_max


def test_decide_trade_skips_when_threshold_not_met():
    """Test that policy skips when price change is below threshold."""
    params = PolicyParams(
        participate_threshold=0.05,  # 5% threshold
        entry_day=0,
        hold_k=3,
        w_max=0.2,
        raw_weight=0.15
    )
    
    # Episode with only 2% price increase
    episode = {
        'prices': [100.0, 102.0, 103.0, 104.0, 105.0],
        'ticker': 'TEST'
    }
    
    decision = decide_trade(episode, params)
    
    assert decision['participate'] is False
    assert decision['weight'] == 0.0


def test_decide_trade_handles_empty_episode():
    """Test that policy handles empty episodes gracefully."""
    params = PolicyParams(
        participate_threshold=0.05,
        entry_day=0,
        hold_k=3,
        w_max=0.2,
        raw_weight=0.15
    )
    
    # Empty episode
    episode = {'prices': []}
    
    decision = decide_trade(episode, params)
    
    assert decision['participate'] is False
    assert decision['weight'] == 0.0


def test_decide_trade_handles_short_episode():
    """Test that policy handles episodes shorter than required holding period."""
    params = PolicyParams(
        participate_threshold=-1.0,  # Very low threshold (will always participate if data exists)
        entry_day=0,
        hold_k=5,  # Need 5 days
        w_max=0.2,
        raw_weight=0.15
    )
    
    # Episode with only 3 days of data
    episode = {
        'prices': [100.0, 101.0, 102.0]
    }
    
    decision = decide_trade(episode, params)
    
    # Should not participate because we don't have enough data for full holding period
    assert decision['participate'] is False
    assert decision['weight'] == 0.0


def test_decide_trade_entry_day1():
    """Test that policy can enter on day1 instead of day0."""
    params = PolicyParams(
        participate_threshold=0.05,
        entry_day=1,  # Enter on day1
        hold_k=2,
        w_max=0.2,
        raw_weight=0.15
    )
    
    episode = {
        'prices': [100.0, 110.0, 115.0, 120.0, 125.0]
    }
    
    decision = decide_trade(episode, params)
    
    assert decision['entry_day'] == 1
    assert decision['exit_day'] == 3  # entry_day + hold_k


def test_decide_trade_exit_day_greater_than_entry():
    """Test that exit_day is always greater than entry_day."""
    params = PolicyParams(
        participate_threshold=-1.0,  # Always participate
        entry_day=0,
        hold_k=3,
        w_max=0.2,
        raw_weight=0.15
    )
    
    episode = {
        'prices': [100.0, 110.0, 115.0, 120.0, 125.0, 130.0]
    }
    
    decision = decide_trade(episode, params)
    
    assert decision['exit_day'] > decision['entry_day']
    assert decision['exit_day'] == decision['entry_day'] + params.hold_k


def test_decide_trade_weight_constraints():
    """Test that weight is constrained between 0 and w_max."""
    params = PolicyParams(
        participate_threshold=-1.0,  # Always participate
        entry_day=0,
        hold_k=2,
        w_max=0.1,  # Small max weight
        raw_weight=0.5  # Large raw weight (should be clipped)
    )
    
    episode = {
        'prices': [100.0, 101.0, 102.0, 103.0, 104.0]
    }
    
    decision = decide_trade(episode, params)
    
    assert decision['participate'] is True
    assert 0 <= decision['weight'] <= params.w_max
    assert decision['weight'] == params.w_max  # Should be clipped to max


def test_decide_trade_negative_price_change():
    """Test policy with negative price change."""
    params = PolicyParams(
        participate_threshold=0.05,  # Positive threshold
        entry_day=0,
        hold_k=2,
        w_max=0.2,
        raw_weight=0.15
    )
    
    # Episode with price decline
    episode = {
        'prices': [100.0, 95.0, 90.0, 85.0]
    }
    
    decision = decide_trade(episode, params)
    
    # Should not participate (negative change < positive threshold)
    assert decision['participate'] is False


def test_decide_trade_with_numpy_arrays():
    """Test that policy works with numpy arrays."""
    params = PolicyParams(
        participate_threshold=0.05,
        entry_day=0,
        hold_k=2,
        w_max=0.2,
        raw_weight=0.15
    )
    
    episode = {
        'prices': np.array([100.0, 110.0, 115.0, 120.0])
    }
    
    decision = decide_trade(episode, params)
    
    assert decision['participate'] is True
    assert decision['weight'] > 0
