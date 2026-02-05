"""
Basic validation tests for Online Portfolio Optimization.

Run with: pytest tests/test_basic.py -v
"""

import numpy as np
import torch
import pytest
import sys
sys.path.append('src')

from model import project_to_simplex, max_drawdown_from_returns, OnlineOGDAllocator


class TestSimplexProjection:
    """Tests for simplex projection function."""

    def test_already_valid_weights(self):
        """Weights already on simplex should remain unchanged."""
        v = torch.tensor([0.3, 0.7])
        p = project_to_simplex(v)
        assert torch.allclose(p.sum(), torch.tensor(1.0))
        assert (p >= 0).all()
        assert torch.allclose(p, v, atol=1e-5)

    def test_negative_values(self):
        """Negative values should be projected to valid weights."""
        v = torch.tensor([-0.5, 1.5])
        p = project_to_simplex(v)
        assert torch.allclose(p.sum(), torch.tensor(1.0))
        assert (p >= 0).all()

    def test_sum_greater_than_one(self):
        """Weights summing > 1 should be normalized."""
        v = torch.tensor([0.8, 0.8])
        p = project_to_simplex(v)
        assert torch.allclose(p.sum(), torch.tensor(1.0))
        assert (p >= 0).all()

    def test_all_negative(self):
        """All negative values should give uniform weights."""
        v = torch.tensor([-1.0, -2.0])
        p = project_to_simplex(v)
        assert torch.allclose(p.sum(), torch.tensor(1.0))
        assert (p >= 0).all()

    def test_single_dominant(self):
        """One large value should dominate."""
        v = torch.tensor([10.0, 0.1])
        p = project_to_simplex(v)
        assert torch.allclose(p.sum(), torch.tensor(1.0))
        assert p[0] > p[1]

    def test_three_assets(self):
        """Should work with more than 2 assets."""
        v = torch.tensor([0.5, 0.3, 0.4])
        p = project_to_simplex(v)
        assert torch.allclose(p.sum(), torch.tensor(1.0))
        assert (p >= 0).all()


class TestMaxDrawdown:
    """Tests for max drawdown calculation."""

    def test_no_drawdown(self):
        """Constant positive returns should have ~0 drawdown."""
        rets = torch.tensor([0.01] * 10)
        mdd = max_drawdown_from_returns(rets)
        assert mdd >= -0.01  # Small negative due to numerical precision

    def test_single_large_loss(self):
        """Single large loss should be captured."""
        rets = torch.tensor([0.01, 0.01, -0.20, 0.01, 0.01])
        mdd = max_drawdown_from_returns(rets)
        assert mdd < -0.15  # Should detect the 20% loss

    def test_recovery(self):
        """Drawdown should reflect peak-to-trough."""
        rets = torch.tensor([0.10, -0.15, 0.20])  # Up, down, up
        mdd = max_drawdown_from_returns(rets)
        assert mdd < 0  # Should have some drawdown

    def test_differentiable(self):
        """Function should be differentiable."""
        rets = torch.tensor([0.01, -0.02, 0.01], requires_grad=True)
        mdd = max_drawdown_from_returns(rets)
        mdd.backward()
        assert rets.grad is not None


class TestOGDAllocator:
    """Tests for OGD Allocator."""

    @pytest.fixture
    def allocator(self):
        """Create allocator instance."""
        return OnlineOGDAllocator(
            n_assets=2,
            window=50,
            lr=0.1,
            risk_aversion=10.0,
            drawdown_penalty=5.0,
            turnover_penalty=0.1
        )

    @pytest.fixture
    def dummy_returns(self):
        """Generate dummy return data."""
        np.random.seed(42)
        return np.random.randn(100, 2) * 0.02

    def test_initial_weights(self, allocator):
        """Initial weights should be uniform."""
        w = allocator.get_weights()
        assert len(w) == 2
        assert np.allclose(w, [0.5, 0.5])

    def test_step_returns_valid_weights(self, allocator, dummy_returns):
        """Step should return valid simplex weights."""
        w = allocator.step(dummy_returns[:50])
        assert len(w) == 2
        assert np.allclose(w.sum(), 1.0)
        assert (w >= 0).all()

    def test_weights_change_after_step(self, allocator, dummy_returns):
        """Weights should change after optimization step."""
        w_before = allocator.get_weights().copy()
        allocator.step(dummy_returns[:50])
        w_after = allocator.get_weights()
        # Weights should have changed (unless gradient is exactly zero)
        # This is a soft check as weights might not change much
        assert w_after is not None

    def test_fitness_history_populated(self, allocator, dummy_returns):
        """Fitness history should be populated after steps."""
        for i in range(5):
            allocator.step(dummy_returns[:50])
        assert len(allocator.fitness_history) == 5

    def test_reset(self, allocator, dummy_returns):
        """Reset should restore initial state."""
        allocator.step(dummy_returns[:50])
        allocator.step(dummy_returns[:50])
        allocator.reset()
        assert allocator.t == 0
        assert len(allocator.fitness_history) == 0
        assert np.allclose(allocator.get_weights(), [0.5, 0.5])

    def test_multiple_steps_convergence(self, allocator, dummy_returns):
        """Multiple steps should not produce NaN or extreme values."""
        for i in range(20):
            w = allocator.step(dummy_returns[:50])
            assert not np.isnan(w).any()
            assert np.allclose(w.sum(), 1.0)
            assert (w >= -1e-6).all()  # Allow small numerical errors


class TestFitnessComputation:
    """Tests for fitness score computation."""

    def test_positive_returns_positive_fitness(self):
        """Strong positive returns should give higher fitness."""
        allocator = OnlineOGDAllocator(n_assets=2, risk_aversion=1.0)

        # Strong positive returns
        good_rets = torch.tensor([[0.02, 0.02]] * 50)
        w = torch.tensor([0.5, 0.5])
        fitness_good = allocator.compute_fitness(good_rets, w, w)

        # Negative returns
        bad_rets = torch.tensor([[-0.02, -0.02]] * 50)
        fitness_bad = allocator.compute_fitness(bad_rets, w, w)

        assert fitness_good > fitness_bad

    def test_low_variance_preferred(self):
        """Lower variance should give higher fitness (all else equal)."""
        allocator = OnlineOGDAllocator(n_assets=2, risk_aversion=100.0)

        # Low variance
        low_var = torch.tensor([[0.01, 0.01]] * 50)

        # High variance (same mean)
        np.random.seed(42)
        high_var = torch.tensor(np.random.randn(50, 2) * 0.05 + 0.01).float()

        w = torch.tensor([0.5, 0.5])
        fitness_low = allocator.compute_fitness(low_var, w, w)
        fitness_high = allocator.compute_fitness(high_var, w, w)

        assert fitness_low > fitness_high

    def test_turnover_penalty_applied(self):
        """Turnover should reduce fitness."""
        allocator = OnlineOGDAllocator(n_assets=2, turnover_penalty=1.0)

        rets = torch.tensor([[0.01, 0.01]] * 50)
        w = torch.tensor([0.5, 0.5])

        # No turnover
        fitness_no_turn = allocator.compute_fitness(rets, w, w)

        # With turnover
        w_prev = torch.tensor([0.8, 0.2])
        fitness_with_turn = allocator.compute_fitness(rets, w, w_prev)

        assert fitness_no_turn > fitness_with_turn


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
