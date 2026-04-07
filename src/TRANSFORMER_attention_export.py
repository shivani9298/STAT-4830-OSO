"""
Export Transformer self-attention maps (2-asset or sector multi-head models).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .TRANSFORMER_model import SectorMultiHeadTransformerAllocator, TransformerAllocator


def attention_maps_to_numpy(maps: list[torch.Tensor]) -> dict[str, np.ndarray]:
    """``layer_i`` -> (B, T, T) float32 arrays."""
    return {f"layer_{i}": m.cpu().numpy().astype(np.float32) for i, m in enumerate(maps)}


def save_attention_npz(
    model: nn.Module,
    x: torch.Tensor,
    out_path: str | Path,
    *,
    meta: dict[str, Any] | None = None,
) -> tuple[Path, list[torch.Tensor]]:
    """
    Run ``model.attention_maps(x)`` once and save weights as ``layer_0``, ``layer_1``, ...

    ``x`` must be on the same device as ``model`` before calling (caller moves tensors).

    Parameters
    ----------
    model
        :class:`TransformerAllocator` or :class:`SectorMultiHeadTransformerAllocator`.
    x
        Shape ``(B, T, F)``.
    out_path
        ``.npz`` path.
    meta
        Extra scalars/strings stored as ``meta_*`` keys (string values only for non-arrays).

    Returns
    -------
    path
        Path to the ``.npz`` file.
    maps
        Raw attention tensors (same as ``model.attention_maps(x)``).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, (TransformerAllocator, SectorMultiHeadTransformerAllocator)):
        maps = model.attention_maps(x)
    else:
        raise TypeError(
            "attention export only supports TransformerAllocator and "
            "SectorMultiHeadTransformerAllocator; got "
            f"{type(model).__name__}"
        )

    arrays = attention_maps_to_numpy(maps)
    extra: dict[str, Any] = {}
    if meta:
        for k, v in meta.items():
            key = f"meta_{k}"
            if isinstance(v, (np.ndarray, np.generic)):
                extra[key] = v
            else:
                extra[key] = np.array(str(v))

    np.savez(out_path, **arrays, **extra)
    return out_path, maps


def save_attention_heatmap_png(
    attn: torch.Tensor | np.ndarray,
    out_path: str | Path,
    *,
    title: str = "Self-attention (mean over batch)",
    dpi: int = 150,
) -> Path:
    """
    Save a single heatmap for ``(B, T, T)`` attention averaged over batch dimension.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(attn, torch.Tensor):
        a = attn.float().mean(dim=0).cpu().numpy()
    else:
        a = np.asarray(attn, dtype=np.float32).mean(axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(a, aspect="auto", cmap="viridis")
    ax.set_xlabel("Key position (day in window)")
    ax.set_ylabel("Query position (day in window)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
