"""
Core optimization model for Online Portfolio Optimization.

This module implements the Online Gradient Descent (OGD) allocator
for dynamic portfolio allocation between S&P 500 and IPO Index.
"""

import numpy as np
import torch
from typing import Optional, Tuple


def project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    """
    Euclidean projection onto the probability simplex.

    Ensures weights are:
    - Non-negative: w_i >= 0
    - Sum to one: sum(w_i) = 1

    Algorithm: O(n log n) via sorting
    Reference: https://arxiv.org/abs/1309.1541

    Parameters
    ----------
    v : torch.Tensor
        1D tensor to project

    Returns
    -------
    torch.Tensor
        Projected weights on simplex
    """
    if v.ndim != 1:
        raise ValueError("v must be a 1D tensor")

    n = v.numel()
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0

    if not torch.any(cond):
        return torch.ones_like(v) / n

    rho = torch.nonzero(cond, as_tuple=False)[-1].item()
    theta = cssv[rho] / (rho + 1.0)
    w = torch.clamp(v - theta, min=0.0)
    s = w.sum()

    return w / s if s > 0 else torch.ones_like(v) / n


def max_drawdown_from_returns(port_ret: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute differentiable maximum drawdown from return series.

    Parameters
    ----------
    port_ret : torch.Tensor
        1D tensor of portfolio returns
    eps : float
        Small constant for numerical stability

    Returns
    -------
    torch.Tensor
        Scalar tensor with max drawdown (negative value, e.g., -0.25 for 25% DD)
    """
    cum = torch.cumprod(1.0 + port_ret, dim=0)
    peak, _ = torch.cummax(cum, dim=0)
    dd = cum / (peak + eps) - 1.0
    return torch.min(dd)


class OnlineOGDAllocator:
    """
    Online Gradient Descent Portfolio Allocator.

    Optimizes portfolio weights to maximize a fitness score using
    projected gradient ascent with adaptive learning rates.

    Fitness Score:
        F(w) = mean_return - λ1*variance + λ2*max_drawdown - λ3*turnover

    Parameters
    ----------
    n_assets : int
        Number of assets in portfolio
    window : int
        Lookback window for computing gradients (default: 126)
    lr : float
        Initial learning rate (default: 0.10)
    lr_decay : float
        Learning rate decay factor per step (default: 0.999)
    risk_aversion : float
        λ1 - Penalty coefficient for variance (default: 20.0)
    drawdown_penalty : float
        λ2 - Penalty coefficient for max drawdown (default: 8.0)
    turnover_penalty : float
        λ3 - Penalty coefficient for turnover (default: 0.15)
    device : str
        PyTorch device ('cpu' or 'cuda')

    Attributes
    ----------
    w : torch.Tensor
        Current portfolio weights
    fitness_history : list
        History of fitness scores
    t : int
        Current time step
    """

    def __init__(
        self,
        n_assets: int,
        window: int = 126,
        lr: float = 0.10,
        lr_decay: float = 0.999,
        risk_aversion: float = 20.0,
        drawdown_penalty: float = 8.0,
        turnover_penalty: float = 0.15,
        device: str = "cpu"
    ):
        self.n_assets = n_assets
        self.window = int(window)
        self.lr0 = float(lr)
        self.lr_decay = float(lr_decay)
        self.risk_aversion = float(risk_aversion)
        self.drawdown_penalty = float(drawdown_penalty)
        self.turnover_penalty = float(turnover_penalty)
        self.device = device

        # Initialize weights uniformly
        self.t = 0
        self.w = torch.ones(n_assets, device=device) / n_assets
        self.w_prev = self.w.clone()
        self.fitness_history = []

    def compute_fitness(
        self,
        returns: torch.Tensor,
        weights: torch.Tensor,
        w_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute fitness score for given weights.

        Fitness = mean_return - λ1*variance + λ2*mdd - λ3*turnover

        Parameters
        ----------
        returns : torch.Tensor
            Return matrix of shape (T, n_assets)
        weights : torch.Tensor
            Portfolio weights of shape (n_assets,)
        w_prev : torch.Tensor
            Previous weights for turnover calculation

        Returns
        -------
        torch.Tensor
            Scalar fitness score
        """
        port = returns @ weights
        mu = port.mean()
        var = port.var(unbiased=False)
        mdd = max_drawdown_from_returns(port)
        turnover = torch.sum(torch.abs(weights - w_prev))

        fitness = (
            mu
            - self.risk_aversion * var
            + self.drawdown_penalty * mdd
            - self.turnover_penalty * turnover
        )
        return fitness

    def step(self, window_returns: np.ndarray) -> np.ndarray:
        """
        Perform one step of online gradient descent.

        Parameters
        ----------
        window_returns : np.ndarray
            Return matrix of shape (T, n_assets)

        Returns
        -------
        np.ndarray
            Updated portfolio weights
        """
        self.t += 1
        lr = self.lr0 * (self.lr_decay ** (self.t - 1))

        R = torch.tensor(window_returns, device=self.device, dtype=torch.float32)
        w_var = self.w.clone().detach().requires_grad_(True)

        # Compute fitness and gradient
        fitness = self.compute_fitness(R, w_var, self.w_prev)
        self.fitness_history.append(fitness.item())

        loss = -fitness  # Minimize negative fitness = maximize fitness
        loss.backward()
        grad = w_var.grad.detach()

        # Gradient ascent step
        w_new = self.w - lr * grad

        # Project onto simplex
        w_new = project_to_simplex(w_new)

        # Update state
        self.w_prev = self.w.clone()
        self.w = w_new.detach()

        return self.w.cpu().numpy()

    def get_weights(self) -> np.ndarray:
        """Return current weights as numpy array."""
        return self.w.cpu().numpy()

    def reset(self):
        """Reset allocator to initial state."""
        self.t = 0
        self.w = torch.ones(self.n_assets, device=self.device) / self.n_assets
        self.w_prev = self.w.clone()
        self.fitness_history = []
