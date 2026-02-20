"""
Portfolio weight models: MLP, GRU/LSTM, Transformer, Hybrid.

Input: (batch, T, F) — window of past returns/features.
Output: (batch, n_assets) — weights on simplex (non-negative, sum 1).
All models share the same interface: forward(x) -> weights.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Literal, Union

ModuleType = Union["AllocatorNet", "MLPAllocator", "TransformerAllocator", "HybridAllocator"]


class AllocatorNet(nn.Module):
    """
    GRU or LSTM → last hidden → MLP → softmax → weights.
    """

    def __init__(
        self,
        input_size: int,
        n_assets: int = 2,
        hidden_size: int = 64,
        num_layers: int = 1,
        rnn_type: Literal["gru", "lstm"] = "gru",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.input_size = input_size
        self.hidden_size = hidden_size
        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        self.rnn_type = rnn_type
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_assets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        logits = self.mlp(last)
        return torch.softmax(logits, dim=-1)


class MLPAllocator(nn.Module):
    """
    Flatten or aggregate window (mean, std per feature) → MLP → softmax → weights.
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        n_assets: int = 2,
        hidden_size: int = 64,
        aggregate: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets
        if aggregate:
            # mean + std per feature -> 2 * input_size
            in_dim = 2 * input_size
        else:
            in_dim = seq_len * input_size
        self.aggregate = aggregate
        self.input_size = input_size
        self.seq_len = seq_len
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_assets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.aggregate:
            m = x.mean(dim=1)
            s = x.std(dim=1).clamp(min=1e-6)
            feat = torch.cat([m, s], dim=1)
        else:
            feat = x.reshape(x.size(0), -1)
        logits = self.mlp(feat)
        return torch.softmax(logits, dim=-1)


class TransformerAllocator(nn.Module):
    """
    Positional encoding + transformer encoder → pool (last token) → MLP → softmax → weights.
    """

    def __init__(
        self,
        input_size: int,
        n_assets: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_assets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        x = self.proj(x)
        pe = self._positional_encoding(T, x.device)
        x = x + pe
        x = self.transformer(x)
        last = x[:, -1, :]
        logits = self.mlp(last)
        return torch.softmax(logits, dim=-1)

    def _positional_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        pe = torch.zeros(1, seq_len, self.d_model, device=device, dtype=torch.float32)
        for i in range(seq_len):
            for j in range(0, self.d_model, 2):
                pe[0, i, j] = math.sin(i / 10000 ** (j / self.d_model))
                if j + 1 < self.d_model:
                    pe[0, i, j + 1] = math.cos(i / 10000 ** (j / self.d_model))
        return pe


class HybridAllocator(nn.Module):
    """
    Stage 1: GRU/Transformer → state vector. Stage 2: MLP(state) → softmax → weights.
    """

    def __init__(
        self,
        input_size: int,
        n_assets: int = 2,
        hidden_size: int = 64,
        num_layers: int = 1,
        use_transformer: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.hidden_size = hidden_size
        if use_transformer:
            self.proj = nn.Linear(input_size, hidden_size)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=4,
                    dim_feedforward=hidden_size * 2,
                    dropout=dropout,
                    batch_first=True,
                ),
                num_layers=num_layers,
            )
            self._get_state = self._get_state_transformer
        else:
            self.encoder = nn.GRU(
                input_size, hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
            )
            self._get_state = self._get_state_gru
        self.use_transformer = use_transformer
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_assets),
        )

    def _get_state_gru(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.encoder(x)
        return out[:, -1, :]

    def _get_state_transformer(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        x = self.proj(x)
        pe = torch.zeros(1, T, self.hidden_size, device=x.device, dtype=x.dtype)
        for i in range(T):
            for j in range(0, self.hidden_size, 2):
                pe[0, i, j] = math.sin(i / 10000 ** (j / self.hidden_size))
                if j + 1 < self.hidden_size:
                    pe[0, i, j + 1] = math.cos(i / 10000 ** (j / self.hidden_size))
        x = x + pe
        x = self.encoder(x)
        return x[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self._get_state(x)
        logits = self.head(state)
        return torch.softmax(logits, dim=-1)


def build_model(
    n_features: int,
    n_assets: int = 2,
    seq_len: int = 252,
    hidden_size: int = 64,
    num_layers: int = 1,
    model_type: str = "gru",
    rnn_type: str = "gru",
    dropout: float = 0.1,
) -> ModuleType:
    if model_type == "mlp":
        return MLPAllocator(
            input_size=n_features,
            seq_len=seq_len,
            n_assets=n_assets,
            hidden_size=hidden_size,
            dropout=dropout,
        )
    if model_type == "transformer":
        return TransformerAllocator(
            input_size=n_features,
            n_assets=n_assets,
            d_model=hidden_size,
            nhead=4,
            num_layers=min(2, num_layers + 1),
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
        )
    if model_type == "hybrid":
        return HybridAllocator(
            input_size=n_features,
            n_assets=n_assets,
            hidden_size=hidden_size,
            num_layers=num_layers,
            use_transformer=False,
            dropout=dropout,
        )
    rnn = "lstm" if model_type == "lstm" else "gru"
    return AllocatorNet(
        input_size=n_features,
        n_assets=n_assets,
        hidden_size=hidden_size,
        num_layers=num_layers,
        rnn_type=rnn,
        dropout=dropout,
    )
