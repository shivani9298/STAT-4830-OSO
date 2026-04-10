"""
Differentiable loss components for IPO portfolio optimization.

L = mean_return_weight*(-mean_return) + lambda_cvar*CVaR + ... + log_growth_weight*(-mean log(1+r))
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


def loss_log_growth_proxy(port_ret: torch.Tensor) -> torch.Tensor:
    """
    Negative mean log(1 + r). Minimizing ``k * loss_log_growth_proxy`` for k > 0 pushes toward
    higher **geometric** growth within the batch (closer to maximizing compound return than mean alone).
    """
    x = port_ret.clamp(min=-0.999999)
    return -torch.log1p(x).mean()


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


def loss_vol_excess(
    port_ret: torch.Tensor,
    target_vol_annual: float = 0.20,
    ann_factor: float = 252.0,
) -> torch.Tensor:
    """
    Penalize when portfolio volatility exceeds target.
    Uses realized daily std, annualized for comparison with target_vol_annual.
    """
    daily_std = port_ret.std(unbiased=False).clamp(min=1e-8)
    ann_vol = daily_std * (ann_factor ** 0.5)
    return torch.relu(ann_vol - target_vol_annual)


def loss_weight_path(weights: torch.Tensor, weights_prev: Optional[torch.Tensor]) -> torch.Tensor:
    """Weight instability: sum of (w_t - w_prev)^2, averaged over batch."""
    if weights_prev is None:
        return torch.tensor(0.0, device=weights.device, dtype=weights.dtype)
    return ((weights - weights_prev) ** 2).sum(dim=-1).mean()


def loss_diversification_min_weight(weights: torch.Tensor, min_weight: float = 0.1) -> torch.Tensor:
    """
    Penalize when the minimum asset weight is below min_weight.
    Encourages holding both market and IPO instead of going all-in on one.
    """
    w_min = weights.min(dim=-1).values
    return torch.relu(min_weight - w_min).mean()


def combined_loss(
    weights: torch.Tensor,
    returns: torch.Tensor,
    weights_prev: Optional[torch.Tensor] = None,
    lambda_cvar: float = 0.5,
    lambda_turnover: float = 0.01,
    lambda_vol: float = 0.5,
    lambda_path: float = 0.01,
    lambda_vol_excess: float = 0.0,
    target_vol_annual: float = 0.20,
    lambda_diversify: float = 0.0,
    min_weight: float = 0.1,
    cvar_alpha: float = 0.05,
    mean_return_weight: float = 1.0,
    log_growth_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Combined loss L and component dict for logging.

    weights: (B, n_assets), returns: (B, n_assets)

    ``mean_return_weight`` (>0): scales the daily **mean return** term vs risk penalties; use >1 to
    prioritize level returns. ``log_growth_weight`` (>=0): adds ``-mean(log(1+r))``, a differentiable
    proxy for **compound** growth within the batch (often correlates better with total return than mean alone).

    Why 100% IPO before diversify penalty:
    - Loss minimizes -mean_return + risk terms. IPO vastly outperformed market in-sample.
    - No prior penalty for concentration, so optimal allocation = all-in on higher-return asset.
    """
    port_ret = portfolio_returns(weights, returns)
    L_mean = loss_mean_return(port_ret)
    L_log = loss_log_growth_proxy(port_ret)
    cvar_val = cvar_smooth(port_ret, alpha=cvar_alpha)
    L_cvar = -cvar_val  # penalize when CVaR is negative (bad tail)
    L_turn = loss_turnover(weights, weights_prev)
    L_vol = loss_return_variance(port_ret)
    L_vol_excess = loss_vol_excess(port_ret, target_vol_annual=target_vol_annual)
    L_path = loss_weight_path(weights, weights_prev)
    L_div = loss_diversification_min_weight(weights, min_weight=min_weight)

    L = (
        float(mean_return_weight) * L_mean
        + float(log_growth_weight) * L_log
        + lambda_cvar * L_cvar
        + lambda_turnover * L_turn
        + lambda_vol * L_vol
        + lambda_vol_excess * L_vol_excess
        + lambda_path * L_path
        + lambda_diversify * L_div
    )
    components = {
        "mean_return": -L_mean.item(),
        "mean_log1p_return": float(torch.log1p(port_ret.clamp(min=-0.999999)).mean().item()),
        "cvar": cvar_val.item(),
        "turnover": L_turn.item(),
        "volatility": L_vol.item(),
        "vol_excess": L_vol_excess.item(),
        "weight_path": L_path.item(),
        "diversify": L_div.item(),
    }
    return L, components


def combined_loss_sector_heads(
    weights: torch.Tensor,
    returns: torch.Tensor,
    weights_prev: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Mean of ``combined_loss`` over G independent sector heads.

    weights, returns: (B, G, 2)
    weights_prev: (B, G, 2) or None
    """
    n_sec = weights.shape[1]
    if n_sec == 0:
        return torch.tensor(0.0, device=weights.device, dtype=weights.dtype), {}
    total = torch.tensor(0.0, device=weights.device, dtype=weights.dtype)
    agg: dict[str, float] = {}
    for g in range(n_sec):
        wp = weights_prev[:, g] if weights_prev is not None else None
        L, comp = combined_loss(weights[:, g], returns[:, g], weights_prev=wp, **kwargs)
        total = total + L
        for k, v in comp.items():
            agg[k] = agg.get(k, 0.0) + v
    n = float(n_sec)
    agg_mean = {k: v / n for k, v in agg.items()}
    return total / n_sec, agg_mean
