"""
GRU-based portfolio policy for multi-asset allocation over time.
Input: window of past returns [B, W, N]. Output: weights on simplex [B, N].
Fits build_ipo_index.py: use --use-gru to weight index constituents by GRU instead of market-cap or OnlinePortfolioOptimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRUPolicy(nn.Module):
    """
    GRU that maps a window of N-asset returns to portfolio weights (simplex).
    Input x: [B, W, N] (batch, window_length, n_assets).
    Output: [B, N] weights (softmax).
    """

    def __init__(
        self,
        n_assets: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=n_assets,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_assets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, W, N]   (batch, window, assets)
        Returns: [B, N] weights on simplex.
        """
        out, _ = self.gru(x)
        last = out[:, -1, :]
        logits = self.head(last)
        weights = F.softmax(logits, dim=-1)
        return weights


def portfolio_utility(
    portfolio_returns: torch.Tensor,
    weights: torch.Tensor,
    lam_var: float = 0.1,
    lam_tc: float = 1e-3,
) -> torch.Tensor:
    """
    Utility = mean(portfolio_returns) - lam_var * var - lam_tc * turnover.
    portfolio_returns: [T]
    weights: [T, N]
    """
    mean = portfolio_returns.mean()
    var = portfolio_returns.var(unbiased=False)
    turnover = (weights[1:] - weights[:-1]).abs().sum(dim=-1).mean()
    utility = mean - lam_var * var - lam_tc * turnover
    return utility


def build_windows(returns: np.ndarray, window: int) -> tuple:
    """
    Build overlapping windows from return matrix [T, N].
    Returns: (windows [T - window, window, N], next_returns [T - window, N] for targets).
    """
    T, N = returns.shape
    if T <= window:
        return np.zeros((0, window, N), dtype=np.float32), np.zeros((0, N), dtype=np.float32)
    windows = np.stack([returns[i : i + window] for i in range(T - window)], axis=0)
    next_returns = returns[window:]
    return windows, next_returns


def train_gru_on_returns(
    returns: np.ndarray,
    window: int = 21,
    n_epochs: int = 50,
    lr: float = 1e-3,
    hidden_size: int = 128,
    num_layers: int = 2,
    lam_var: float = 0.1,
    lam_tc: float = 1e-3,
    device: torch.device = None,
    seed: int = 0,
) -> tuple:
    """
    Train GRUPolicy on return matrix [T, N] to maximize portfolio_utility.
    Returns: (trained GRUPolicy, list of utility per epoch).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, N = returns.shape
    if T <= window or N == 0:
        raise ValueError("Need T > window and N > 0")

    windows_np, next_returns_np = build_windows(returns, window)
    n_samples = len(windows_np)
    if n_samples == 0:
        raise ValueError("No windows built")

    torch.manual_seed(seed)
    model = GRUPolicy(N, hidden_size=hidden_size, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(n_epochs):
        model.train()
        x = torch.tensor(windows_np, dtype=torch.float32, device=device)
        next_ret = torch.tensor(next_returns_np, dtype=torch.float32, device=device)

        w = model(x)
        portfolio_ret = (w * next_ret).sum(dim=-1)
        loss = -portfolio_utility(portfolio_ret, w, lam_var=lam_var, lam_tc=lam_tc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(-loss.item())
    return model, history
