"""
Transformer-based portfolio allocators: self-attention encoder → weights.

Used when ``model_type == "transformer"`` from :mod:`src.model` builders.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def _init_equal_weight_head(layer: nn.Module) -> None:
    """
    Initialize final linear layer to zero logits so softmax starts uniform.
    """
    if isinstance(layer, nn.Linear):
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)


def _build_sinusoidal_positional_encoding(
    seq_len: int,
    d_model: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Vectorized sinusoidal positional encoding with shape ``(1, seq_len, d_model)``."""
    pe = torch.zeros(1, seq_len, d_model, device=device, dtype=dtype)
    pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / float(d_model))
    )
    phase = pos * div
    pe[0, :, 0::2] = torch.sin(phase)
    cos_cols = pe[0, :, 1::2].shape[-1]
    if cos_cols > 0:
        pe[0, :, 1::2] = torch.cos(phase[:, :cos_cols])
    return pe


class AttentionCapturingEncoderLayer(nn.TransformerEncoderLayer):
    """
    Like ``TransformerEncoderLayer`` but runs self-attention with ``need_weights=True``
    so ``last_attn`` holds averaged head weights (batch, T, T) after each forward.
    """

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        attn_out, attn_w = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
            is_causal=is_causal,
        )
        self.last_attn = attn_w.detach()
        return self.dropout1(attn_out)


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
        encoder_layer = AttentionCapturingEncoderLayer(
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
        _init_equal_weight_head(self.mlp[-1])
        self.register_buffer("_pe_cache", torch.empty(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _b, T, _f = x.shape
        x = self.proj(x)
        pe = self._positional_encoding(T, x.device, x.dtype)
        x = x + pe
        x = self.transformer(x)
        last = x[:, -1, :]
        logits = self.mlp(last)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def attention_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Self-attention weights per encoder layer after projecting + PE.

        Each tensor has shape ``(batch, T, T)`` (query position vs key position),
        heads averaged. Requires disabling the MHA fused fast path (handled here).

        Parameters
        ----------
        x : (B, T, F)
        """
        prev = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            self.eval()
            b, t, _f = x.shape
            h = self.proj(x)
            pe = self._positional_encoding(t, h.device, h.dtype)
            h = h + pe
            _ = self.transformer(h)
            out = []
            for layer in self.transformer.layers:
                la = getattr(layer, "last_attn", None)
                if la is not None:
                    out.append(la)
            return out
        finally:
            torch.backends.mha.set_fastpath_enabled(prev)

    def _positional_encoding(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cache = self._pe_cache
        if (
            cache.numel() == 0
            or cache.device != device
            or cache.dtype != dtype
            or cache.size(1) < seq_len
        ):
            self._pe_cache = _build_sinusoidal_positional_encoding(
                seq_len,
                self.d_model,
                device=device,
                dtype=dtype,
            )
        return self._pe_cache[:, :seq_len, :]


class SectorMultiHeadTransformerAllocator(nn.Module):
    """
    Shared Transformer encoder; one MLP head per IPO sector group.
    Each head outputs a 2-way softmax (market vs that sector's IPO basket).

    Forward: (B, T, F) -> (B, G, 2).
    """

    def __init__(
        self,
        input_size: int,
        n_sectors: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_sectors = n_sectors
        self.d_model = d_model
        self.proj = nn.Linear(input_size, d_model)
        encoder_layer = AttentionCapturingEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 2),
                )
                for _ in range(n_sectors)
            ]
        )
        for head in self.heads:
            _init_equal_weight_head(head[-1])
        self.register_buffer("_pe_cache", torch.empty(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        x = self.proj(x)
        pe = self._positional_encoding(T, x.device, x.dtype)
        x = x + pe
        x = self.transformer(x)
        h = x[:, -1, :]
        w_list = []
        for head in self.heads:
            logits = head(h)
            w_list.append(torch.softmax(logits, dim=-1))
        return torch.stack(w_list, dim=1)

    @torch.no_grad()
    def attention_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Same semantics as :meth:`TransformerAllocator.attention_maps`."""
        prev = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            self.eval()
            t = x.size(1)
            h = self.proj(x)
            pe = self._positional_encoding(t, h.device, h.dtype)
            h = h + pe
            _ = self.transformer(h)
            out = []
            for layer in self.transformer.layers:
                la = getattr(layer, "last_attn", None)
                if la is not None:
                    out.append(la)
            return out
        finally:
            torch.backends.mha.set_fastpath_enabled(prev)

    def _positional_encoding(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cache = self._pe_cache
        if (
            cache.numel() == 0
            or cache.device != device
            or cache.dtype != dtype
            or cache.size(1) < seq_len
        ):
            self._pe_cache = _build_sinusoidal_positional_encoding(
                seq_len,
                self.d_model,
                device=device,
                dtype=dtype,
            )
        return self._pe_cache[:, :seq_len, :]
