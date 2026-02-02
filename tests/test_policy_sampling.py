"""
Tests for policy parameter sampling.

Tests that sample_params works correctly and produces valid parameters.
"""

import pytest
import numpy as np
from src.policy import PolicyParams, sample_params


def test_sample_params_returns_policy_params():
    """Test that sample_params returns a PolicyParams instance."""
    rng = np.random.default_rng(0)
    params = sample_params(rng)
    
    assert isinstance(params, PolicyParams)


def test_sample_params_deterministic():
    """Test that sample_params is deterministic with same seed."""
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    
    params1 = sample_params(rng1)
    params2 = sample_params(rng2)
    
    assert params1.participate_threshold == params2.participate_threshold
    assert params1.entry_day == params2.entry_day
    assert params1.hold_k == params2.hold_k
    assert params1.w_max == params2.w_max
    assert params1.raw_weight == params2.raw_weight
    assert params1.use_volume_cap == params2.use_volume_cap
    assert params1.vol_cap_mult == params2.vol_cap_mult


def test_sample_params_valid_ranges():
    """Test that sampled parameters are within valid ranges."""
    rng = np.random.default_rng(42)
    
    for _ in range(100):  # Sample many times
        params = sample_params(rng)
        
        # Check participate_threshold range (should be in reasonable range)
        assert -0.1 <= params.participate_threshold <= 0.1
        
        # Entry day should be 0 or 1
        assert params.entry_day in [0, 1]
        
        # Hold period should be 1 to 10
        assert 1 <= params.hold_k <= 10
        
        # w_max should be positive and reasonable
        assert 0 < params.w_max <= 0.25
        
        # raw_weight should be positive
        assert params.raw_weight > 0
        
        # vol_cap_mult should be positive
        assert params.vol_cap_mult > 0


def test_sample_params_works_with_default_rng():
    """Test that sample_params works with np.random.default_rng(0) as specified."""
    rng = np.random.default_rng(0)
    params = sample_params(rng)
    
    assert isinstance(params, PolicyParams)
    assert params.entry_day in [0, 1]
    assert 1 <= params.hold_k <= 10


def test_sample_params_fast():
    """Test that sample_params is fast (should complete quickly)."""
    import time
    
    rng = np.random.default_rng(0)
    
    start = time.time()
    for _ in range(1000):
        _ = sample_params(rng)
    elapsed = time.time() - start
    
    # Should complete 1000 samples in well under a second
    assert elapsed < 1.0, f"sample_params too slow: {elapsed:.3f}s for 1000 samples"
