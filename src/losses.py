"""
Differentiable loss components for IPO portfolio optimization.

L = -mean_return + lambda_cvar*CVaR + lambda_turnover*turnover
    + lambda_vol*return_variance + lambda_path*weight_instability
"""
from __future__ import annotations

import torch
from typing import Optional


def portfolio_returns(weights: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """
    weights: (B, n_assets), returns: (B, n_assets) -> (B,)
    """
    return (weights * returns).sum(dim=-1)


def loss_mean_return(port_ret: torch.Tensor) -> torch.Tensor:
    """Negative mean (minimize this = maximize mean)."""
    return -port_ret.mean()


def cvar_smooth(port_ret: torch.Tensor, alpha: float = 0.05, temperature: float = 0.1) -> torch.Tensor:
    """
    Differentiable approximation of CVaR (Expected Shortfall) at level alpha.
    Uses soft sorting: softmax weights by (r - quantile); CVaR = weighted avg of worst returns.
    """
    n = port_ret.numel()
    k = max(1, int(alpha * n))
    # Soft top-k: weight each return by how much it is in the "worst" tail
    s, _ = torch.sort(port_ret, dim=0)
    q = s[k - 1] if k <= len(s) else s[0]
    # Soft indicator: weight ~ exp(-(r - q)/temp) for r < q
    diff = port_ret - q
    w = torch.exp(-torch.relu(-diff) / temperature)
    w = w / (w.sum() + 1e-8)
    cvar = (w * port_ret).sum()
    return cvar


def loss_turnover(weights: torch.Tensor, weights_prev: Optional[torch.Tensor]) -> torch.Tensor:
    """Sum of |w_t - w_prev| over assets, averaged over batch."""
    if weights_prev is None:
        return torch.tensor(0.0, device=weights.device, dtype=weights.dtype)
    return torch.abs(weights - weights_prev).sum(dim=-1).mean()


def loss_return_variance(port_ret: torch.Tensor) -> torch.Tensor:
    """Variance of portfolio returns."""
    return port_ret.var(unbiased=False)


def loss_weight_path(weights: torch.Tensor, weights_prev: Optional[torch.Tensor]) -> torch.Tensor:
    """Weight instability: sum of (w_t - w_prev)^2, averaged over batch."""
    if weights_prev is None:
        return torch.tensor(0.0, device=weights.device, dtype=weights.dtype)
    return ((weights - weights_prev) ** 2).sum(dim=-1).mean()


def combined_loss(
    weights: torch.Tensor,
    returns: torch.Tensor,
    weights_prev: Optional[torch.Tensor] = None,
    lambda_cvar: float = 0.5,
    lambda_turnover: float = 0.01,
    lambda_vol: float = 0.5,
    lambda_path: float = 0.01,
    cvar_alpha: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """
    Combined loss L and component dict for logging.

    weights: (B, n_assets), returns: (B, n_assets)
    """
    port_ret = portfolio_returns(weights, returns)
    L_mean = loss_mean_return(port_ret)
    cvar_val = cvar_smooth(port_ret, alpha=cvar_alpha)
    L_cvar = -cvar_val  # penalize when CVaR is negative (bad tail)
    L_turn = loss_turnover(weights, weights_prev)
    L_vol = loss_return_variance(port_ret)
    L_path = loss_weight_path(weights, weights_prev)

    L = (
        L_mean
        + lambda_cvar * L_cvar
        + lambda_turnover * L_turn
        + lambda_vol * L_vol
        + lambda_path * L_path
    )
    components = {
        "mean_return": -L_mean.item(),
        "cvar": cvar_val.item(),
        "turnover": L_turn.item(),
        "volatility": L_vol.item(),
        "weight_path": L_path.item(),
    }
    return L, components
