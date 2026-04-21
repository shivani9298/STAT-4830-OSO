"""
Training loop for IPO portfolio allocator.

Forward → portfolio returns → L → backward; validate; save best checkpoint.

Learning-rate schedules (dynamic step sizes): ``constant``, ``cosine``,
``warmup_cosine`` (linear warmup then cosine decay),
``plateau`` (ReduceLROnPlateau on validation loss), ``exponential`` (per-epoch decay).
"""
from __future__ import annotations

import math
import torch
from pathlib import Path
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)
from typing import Any, Literal, Optional

from .model import build_model, build_sector_head_model

GradNormMode = Literal["clip", "rescale", "none"]
from .losses import combined_loss, combined_loss_sector_heads

LrSchedule = Literal["constant", "cosine", "warmup_cosine", "plateau", "exponential"]


def _resolve_lr_schedule(
    lr_schedule: str | None,
    cosine_lr: bool,
) -> LrSchedule:
    """Map config + legacy ``cosine_lr`` flag to a schedule name."""
    if lr_schedule:
        s = str(lr_schedule).lower().strip()
        aliases = {
            "none": "constant",
            "fixed": "constant",
            "reduce": "plateau",
            "reduce_on_plateau": "plateau",
            "adaptive": "plateau",
            "dynamic": "plateau",
            "exp": "exponential",
        }
        s = aliases.get(s, s)
        if s not in ("constant", "cosine", "warmup_cosine", "plateau", "exponential"):
            raise ValueError(
                f"lr_schedule must be constant|cosine|warmup_cosine|plateau|exponential; got {lr_schedule!r}"
            )
        return s  # type: ignore[return-value]
    if cosine_lr:
        return "cosine"
    return "constant"


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule: LrSchedule,
    epochs: int,
    *,
    plateau_factor: float,
    plateau_patience: int,
    min_lr: float,
    exponential_gamma: float,
    warmup_epochs: int = 0,
) -> Any:
    """``warmup_epochs`` used only for ``warmup_cosine``: linear ramp then cosine to ``min_lr``."""
    if schedule == "constant":
        return None
    if schedule == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=float(min_lr))
    if schedule == "warmup_cosine":
        w = max(0, int(warmup_epochs))
        if w >= epochs:
            w = max(0, epochs - 1)
        # Short warmup from 0.01 * lr to full lr, then cosine for the rest of training.
        if w <= 0:
            return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=float(min_lr))
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=w,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - w),
            eta_min=float(min_lr),
        )
        return SequentialLR(optimizer, [warmup, cosine], milestones=[w])
    if schedule == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(plateau_factor),
            patience=int(plateau_patience),
            min_lr=float(min_lr),
            threshold=1e-8,
        )
    if schedule == "exponential":
        g = float(exponential_gamma)
        if not (0.0 < g <= 1.0):
            raise ValueError(f"exponential_gamma must be in (0, 1]; got {g}")
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=g)
    return None


def _apply_gradient_normalization(
    model: torch.nn.Module,
    *,
    max_grad_norm: float,
    mode: GradNormMode,
) -> float:
    """
    After ``loss.backward()``:

    - ``clip`` (default): L2 norm clipping — only scales down if norm exceeds ``max_grad_norm``.
    - ``rescale``: always rescale the full gradient so its global L2 norm equals ``max_grad_norm``
      (unless norm is 0). Useful when gradients are often small and you want a fixed step scale.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return 0.0
    mg = float(max_grad_norm)
    if mg <= 0:
        raise ValueError("max_grad_norm must be positive")
    if mode == "none":
        return float(torch.nn.utils.clip_grad_norm_(params, float("inf")))
    if mode == "clip":
        return float(torch.nn.utils.clip_grad_norm_(params, mg))
    if mode == "rescale":
        total_norm = float(torch.nn.utils.clip_grad_norm_(params, float("inf")))
        if total_norm <= 0:
            return 0.0
        scale = mg / total_norm
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)
        return total_norm
    raise ValueError(f"grad_norm_mode must be clip|rescale|none; got {mode!r}")


def _scheduler_step(
    scheduler: Any,
    val_loss: float,
) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


def _accumulate_term_breakdown(
    acc: dict[str, dict[str, float]],
    td: dict[str, dict[str, Any]],
) -> None:
    """In-place sum of raw / weighted across batches (lambda assumed constant)."""
    for name, d in td.items():
        if name not in acc:
            acc[name] = {"raw": 0.0, "weighted": 0.0, "lambda": float(d["lambda"])}
        acc[name]["raw"] += float(d["raw"])
        acc[name]["weighted"] += float(d["weighted"])


def _finalize_term_breakdown(
    acc: dict[str, dict[str, float]],
    n_batches: int,
) -> dict[str, dict[str, float]]:
    """Average batch contributions for logging."""
    if n_batches <= 0:
        return {}
    out: dict[str, dict[str, float]] = {}
    for k, v in acc.items():
        out[k] = {
            "raw": v["raw"] / n_batches,
            "lambda": v["lambda"],
            "weighted": v["weighted"] / n_batches,
        }
    return out


def _format_term_breakdown_table(bd: dict[str, dict[str, float]], prefix: str) -> str:
    if not bd:
        return ""
    lines = [f"  [{prefix} term breakdown]"]
    for name in sorted(bd.keys()):
        d = bd[name]
        lines.append(
            f"    {name:12s}  raw={d['raw']:+.6e}  λ={d['lambda']:.6g}  λ·raw={d['weighted']:+.6e}"
        )
    return "\n".join(lines)


def _nan_inf_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isnan(x).any().item() or torch.isinf(x).any().item())


def _to_float_tensor(x: Any) -> torch.Tensor:
    """
    Convert dataset arrays to float32 tensors once (usually CPU-resident).
    """
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32)


def _slice_batch_tensor(
    x: torch.Tensor | Any,
    start: int,
    end: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """
    Slice a batch and place it on ``device`` with ``float32`` dtype.
    Handles both tensor-backed and numpy-backed datasets.
    """
    non_blocking = device.type != "cpu"
    if isinstance(x, torch.Tensor):
        b = x[start:end]
        if b.device == device and b.dtype == torch.float32:
            return b
        return b.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    return torch.as_tensor(x[start:end], device=device, dtype=torch.float32)


def _weights_prev_consecutive(
    w: torch.Tensor,
    last_w: Optional[torch.Tensor],
    start_idx: int,
) -> torch.Tensor:
    """
    Build ``weights_prev`` aligned with **adjacent calendar rows** (same length as ``w``).

    Row ``k`` compares to the weight at global index ``start_idx + k - 1``. The first row of
    the dataset (``start_idx == 0``) has no predecessor; it is paired with itself (detached)
    so turnover/path add zero for that row. Within a batch, row ``k`` uses ``w[k-1]`` so
    gradients flow to both neighbors. ``last_w`` is ``w`` at ``start_idx - 1`` when
    ``start_idx > 0`` (typically detached from the previous batch).
    """
    b = w.shape[0]
    if b == 0:
        raise ValueError("w must be non-empty")
    prev = torch.empty_like(w)
    if start_idx == 0:
        prev[0] = w[0].detach()
    else:
        if last_w is None:
            prev[0] = w[0].detach()
        else:
            prev[0] = last_w
    if b > 1:
        prev[1:] = w[:-1]
    return prev


def train_epoch(
    model: torch.nn.Module,
    X: torch.Tensor,
    R: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 64,
    lambda_cvar: float = 0.5,
    lambda_turnover: float = 0.01,
    lambda_vol: float = 0.5,
    lambda_path: float = 0.01,
    lambda_diversify: float = 0.0,
    min_weight: float = 0.1,
    lambda_vol_excess: float = 0.0,
    target_vol_annual: float = 0.20,
    mean_return_weight: float = 1.0,
    log_growth_weight: float = 0.0,
    cvar_temperature: float = 0.1,
    mean_loss_mode: str = "linear",
    huber_delta: float = 0.02,
    winsor_abs: Optional[float] = None,
    max_grad_norm: float = 1.0,
    grad_norm_mode: GradNormMode = "clip",
) -> tuple[float, dict, dict]:
    """``mean_return_weight`` / ``log_growth_weight``: see :func:`src.losses.combined_loss`.
    ``grad_norm_mode``: ``clip`` | ``rescale`` | ``none`` (measure-only for ``none``).
    ``winsor_abs``: if set, training-only clamp on daily portfolio returns inside the loss.
    Returns ``(mean_loss, avg_components, diagnostics)``."""
    model.train()
    n = X.shape[0]
    total_loss = 0.0
    n_batches = 0
    acc_components: dict[str, float] = {}
    breakdown_acc: dict[str, dict[str, float]] = {}
    last_w: Optional[torch.Tensor] = None
    grad_norm_pre: list[float] = []
    grad_norm_post: list[float] = []
    nan_loss_batches = 0
    nan_grad_batches = 0
    skipped_nan_batches = 0
    loss_kw = dict(
        lambda_cvar=lambda_cvar,
        lambda_turnover=lambda_turnover,
        lambda_vol=lambda_vol,
        lambda_path=lambda_path,
        lambda_diversify=lambda_diversify,
        min_weight=min_weight,
        lambda_vol_excess=lambda_vol_excess,
        target_vol_annual=target_vol_annual,
        mean_return_weight=mean_return_weight,
        log_growth_weight=log_growth_weight,
        cvar_temperature=cvar_temperature,
        mean_loss_mode=mean_loss_mode,
        huber_delta=huber_delta,
        winsor_abs=winsor_abs,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    # Batches in chronological order (rolling windows by prediction date).
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if end - start < 1:
            continue
        x = _slice_batch_tensor(X, start, end, device=device)
        r = _slice_batch_tensor(R, start, end, device=device)
        w = model(x)
        prev = _weights_prev_consecutive(w, last_w, start_idx=start)
        loss, components = combined_loss(w, r, weights_prev=prev, **loss_kw)
        if _nan_inf_tensor(loss):
            nan_loss_batches += 1
            skipped_nan_batches += 1
            last_w = w[-1].detach()
            continue
        td = components.pop("term_breakdown", {})
        components.pop("mean_loss_mode", None)
        optimizer.zero_grad()
        loss.backward()
        pre_clip = float(torch.nn.utils.clip_grad_norm_(params, float("inf")))
        grad_nan = False
        for p in params:
            if p.grad is not None and _nan_inf_tensor(p.grad):
                grad_nan = True
                break
        if grad_nan:
            nan_grad_batches += 1
            skipped_nan_batches += 1
            optimizer.zero_grad()
            last_w = w[-1].detach()
            continue
        grad_norm_pre.append(pre_clip)
        post_clip = _apply_gradient_normalization(
            model, max_grad_norm=max_grad_norm, mode=grad_norm_mode
        )
        grad_norm_post.append(post_clip)
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
        _accumulate_term_breakdown(breakdown_acc, td)
        for k, v in components.items():
            if isinstance(v, (int, float)):
                acc_components[k] = acc_components.get(k, 0.0) + float(v)
        last_w = w[-1].detach()
    if n_batches == 0:
        return 0.0, {}, {
            "grad_norm_pre_mean": float("nan"),
            "grad_norm_pre_max": float("nan"),
            "grad_norm_post_mean": float("nan"),
            "grad_norm_post_max": float("nan"),
            "nan_loss_batches": nan_loss_batches,
            "nan_grad_batches": nan_grad_batches,
            "skipped_nan_batches": skipped_nan_batches,
            "train_term_breakdown": {},
        }
    avg = {k: v / n_batches for k, v in acc_components.items()}
    avg["term_breakdown"] = _finalize_term_breakdown(breakdown_acc, n_batches)
    gpre = grad_norm_pre
    gpost = grad_norm_post
    diag = {
        "grad_norm_pre_mean": float(sum(gpre) / max(len(gpre), 1)),
        "grad_norm_pre_max": float(max(gpre)) if gpre else float("nan"),
        "grad_norm_post_mean": float(sum(gpost) / max(len(gpost), 1)),
        "grad_norm_post_max": float(max(gpost)) if gpost else float("nan"),
        "nan_loss_batches": nan_loss_batches,
        "nan_grad_batches": nan_grad_batches,
        "skipped_nan_batches": skipped_nan_batches,
        "train_term_breakdown": avg["term_breakdown"],
    }
    return total_loss / n_batches, avg, diag


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    X: torch.Tensor,
    R: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    lambda_cvar: float = 0.5,
    lambda_turnover: float = 0.01,
    lambda_vol: float = 0.5,
    lambda_path: float = 0.01,
    lambda_diversify: float = 0.0,
    min_weight: float = 0.1,
    lambda_vol_excess: float = 0.0,
    target_vol_annual: float = 0.20,
    mean_return_weight: float = 1.0,
    log_growth_weight: float = 0.0,
    cvar_temperature: float = 0.1,
    mean_loss_mode: str = "linear",
    huber_delta: float = 0.02,
) -> tuple[float, dict]:
    """Validation uses the same loss as training but **no** return winsorization (fair val metric)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    acc_components: dict[str, float] = {}
    breakdown_acc: dict[str, dict[str, float]] = {}
    last_w: Optional[torch.Tensor] = None
    loss_kw = dict(
        lambda_cvar=lambda_cvar,
        lambda_turnover=lambda_turnover,
        lambda_vol=lambda_vol,
        lambda_path=lambda_path,
        lambda_diversify=lambda_diversify,
        min_weight=min_weight,
        lambda_vol_excess=lambda_vol_excess,
        target_vol_annual=target_vol_annual,
        mean_return_weight=mean_return_weight,
        log_growth_weight=log_growth_weight,
        cvar_temperature=cvar_temperature,
        mean_loss_mode=mean_loss_mode,
        huber_delta=huber_delta,
        winsor_abs=None,
    )
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        x = _slice_batch_tensor(X, start, end, device=device)
        r = _slice_batch_tensor(R, start, end, device=device)
        w = model(x)
        prev = _weights_prev_consecutive(w, last_w, start_idx=start)
        loss, components = combined_loss(w, r, weights_prev=prev, **loss_kw)
        td = components.pop("term_breakdown", {})
        components.pop("mean_loss_mode", None)
        total_loss += float(loss.item())
        n_batches += 1
        _accumulate_term_breakdown(breakdown_acc, td)
        for k, v in components.items():
            if isinstance(v, (int, float)):
                acc_components[k] = acc_components.get(k, 0.0) + float(v)
        last_w = w[-1].detach()
    if n_batches == 0:
        return 0.0, {}
    avg = {k: v / n_batches for k, v in acc_components.items()}
    avg["term_breakdown"] = _finalize_term_breakdown(breakdown_acc, n_batches)
    return total_loss / n_batches, avg


def train_epoch_sector_heads(
    model: torch.nn.Module,
    X: torch.Tensor,
    R: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 64,
    lambda_cvar: float = 0.5,
    lambda_turnover: float = 0.01,
    lambda_vol: float = 0.5,
    lambda_path: float = 0.01,
    lambda_diversify: float = 0.0,
    min_weight: float = 0.1,
    lambda_vol_excess: float = 0.0,
    target_vol_annual: float = 0.20,
    mean_return_weight: float = 1.0,
    log_growth_weight: float = 0.0,
    cvar_temperature: float = 0.1,
    mean_loss_mode: str = "linear",
    huber_delta: float = 0.02,
    winsor_abs: Optional[float] = None,
    max_grad_norm: float = 1.0,
    grad_norm_mode: GradNormMode = "clip",
) -> tuple[float, dict, dict]:
    """R: (N, G, 2); model(x) -> (B, G, 2). Same diagnostics as :func:`train_epoch`."""
    model.train()
    n = X.shape[0]
    total_loss = 0.0
    n_batches = 0
    acc_components: dict[str, float] = {}
    breakdown_acc: dict[str, dict[str, float]] = {}
    last_w: Optional[torch.Tensor] = None
    grad_norm_pre: list[float] = []
    grad_norm_post: list[float] = []
    nan_loss_batches = 0
    nan_grad_batches = 0
    skipped_nan_batches = 0
    loss_kw = dict(
        lambda_cvar=lambda_cvar,
        lambda_turnover=lambda_turnover,
        lambda_vol=lambda_vol,
        lambda_path=lambda_path,
        lambda_diversify=lambda_diversify,
        min_weight=min_weight,
        lambda_vol_excess=lambda_vol_excess,
        target_vol_annual=target_vol_annual,
        mean_return_weight=mean_return_weight,
        log_growth_weight=log_growth_weight,
        cvar_temperature=cvar_temperature,
        mean_loss_mode=mean_loss_mode,
        huber_delta=huber_delta,
        winsor_abs=winsor_abs,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if end - start < 1:
            continue
        x = _slice_batch_tensor(X, start, end, device=device)
        r = _slice_batch_tensor(R, start, end, device=device)
        w = model(x)
        prev = _weights_prev_consecutive(w, last_w, start_idx=start)
        loss, components = combined_loss_sector_heads(w, r, weights_prev=prev, **loss_kw)
        if _nan_inf_tensor(loss):
            nan_loss_batches += 1
            skipped_nan_batches += 1
            last_w = w[-1].detach()
            continue
        td = components.pop("term_breakdown", {})
        components.pop("mean_loss_mode", None)
        optimizer.zero_grad()
        loss.backward()
        pre_clip = float(torch.nn.utils.clip_grad_norm_(params, float("inf")))
        grad_nan = any(
            p.grad is not None and _nan_inf_tensor(p.grad) for p in params
        )
        if grad_nan:
            nan_grad_batches += 1
            skipped_nan_batches += 1
            optimizer.zero_grad()
            last_w = w[-1].detach()
            continue
        grad_norm_pre.append(pre_clip)
        post_clip = _apply_gradient_normalization(
            model, max_grad_norm=max_grad_norm, mode=grad_norm_mode
        )
        grad_norm_post.append(post_clip)
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
        _accumulate_term_breakdown(breakdown_acc, td)
        for k, v in components.items():
            if isinstance(v, (int, float)):
                acc_components[k] = acc_components.get(k, 0.0) + float(v)
        last_w = w[-1].detach()
    if n_batches == 0:
        return 0.0, {}, {
            "grad_norm_pre_mean": float("nan"),
            "grad_norm_pre_max": float("nan"),
            "grad_norm_post_mean": float("nan"),
            "grad_norm_post_max": float("nan"),
            "nan_loss_batches": nan_loss_batches,
            "nan_grad_batches": nan_grad_batches,
            "skipped_nan_batches": skipped_nan_batches,
            "train_term_breakdown": {},
        }
    avg = {k: v / n_batches for k, v in acc_components.items()}
    avg["term_breakdown"] = _finalize_term_breakdown(breakdown_acc, n_batches)
    gpre, gpost = grad_norm_pre, grad_norm_post
    diag = {
        "grad_norm_pre_mean": float(sum(gpre) / max(len(gpre), 1)),
        "grad_norm_pre_max": float(max(gpre)) if gpre else float("nan"),
        "grad_norm_post_mean": float(sum(gpost) / max(len(gpost), 1)),
        "grad_norm_post_max": float(max(gpost)) if gpost else float("nan"),
        "nan_loss_batches": nan_loss_batches,
        "nan_grad_batches": nan_grad_batches,
        "skipped_nan_batches": skipped_nan_batches,
        "train_term_breakdown": avg["term_breakdown"],
    }
    return total_loss / n_batches, avg, diag


@torch.no_grad()
def validate_sector_heads(
    model: torch.nn.Module,
    X: torch.Tensor,
    R: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    lambda_cvar: float = 0.5,
    lambda_turnover: float = 0.01,
    lambda_vol: float = 0.5,
    lambda_path: float = 0.01,
    lambda_diversify: float = 0.0,
    min_weight: float = 0.1,
    lambda_vol_excess: float = 0.0,
    target_vol_annual: float = 0.20,
    mean_return_weight: float = 1.0,
    log_growth_weight: float = 0.0,
    cvar_temperature: float = 0.1,
    mean_loss_mode: str = "linear",
    huber_delta: float = 0.02,
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    acc_components: dict[str, float] = {}
    breakdown_acc: dict[str, dict[str, float]] = {}
    last_w: Optional[torch.Tensor] = None
    loss_kw = dict(
        lambda_cvar=lambda_cvar,
        lambda_turnover=lambda_turnover,
        lambda_vol=lambda_vol,
        lambda_path=lambda_path,
        lambda_diversify=lambda_diversify,
        min_weight=min_weight,
        lambda_vol_excess=lambda_vol_excess,
        target_vol_annual=target_vol_annual,
        mean_return_weight=mean_return_weight,
        log_growth_weight=log_growth_weight,
        cvar_temperature=cvar_temperature,
        mean_loss_mode=mean_loss_mode,
        huber_delta=huber_delta,
        winsor_abs=None,
    )
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        x = _slice_batch_tensor(X, start, end, device=device)
        r = _slice_batch_tensor(R, start, end, device=device)
        w = model(x)
        prev = _weights_prev_consecutive(w, last_w, start_idx=start)
        loss, components = combined_loss_sector_heads(w, r, weights_prev=prev, **loss_kw)
        td = components.pop("term_breakdown", {})
        components.pop("mean_loss_mode", None)
        total_loss += float(loss.item())
        n_batches += 1
        _accumulate_term_breakdown(breakdown_acc, td)
        for k, v in components.items():
            if isinstance(v, (int, float)):
                acc_components[k] = acc_components.get(k, 0.0) + float(v)
        last_w = w[-1].detach()
    if n_batches == 0:
        return 0.0, {}
    avg = {k: v / n_batches for k, v in acc_components.items()}
    avg["term_breakdown"] = _finalize_term_breakdown(breakdown_acc, n_batches)
    return total_loss / n_batches, avg


def run_training_sector_heads(
    data: dict,
    device: Optional[torch.device] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 10,
    checkpoint_dir: Optional[str] = None,
    lambda_cvar: float = 0.5,
    lambda_turnover: float = 0.01,
    lambda_vol: float = 0.5,
    lambda_path: float = 0.01,
    lambda_diversify: float = 0.0,
    min_weight: float = 0.1,
    lambda_vol_excess: float = 0.0,
    target_vol_annual: float = 0.20,
    mean_return_weight: float = 1.0,
    log_growth_weight: float = 0.0,
    cvar_temperature: float = 0.1,
    mean_loss_mode: str = "linear",
    huber_delta: float = 0.02,
    winsor_abs: Optional[float] = None,
    max_grad_norm: float = 1.0,
    grad_norm_mode: GradNormMode = "clip",
    hidden_size: int = 64,
    num_layers: int = 1,
    model_type: str = "gru",
    verbose: bool = True,
    log_every: int = 1,
    log_loss_term_breakdown: bool = True,
    weight_decay: float = 1e-5,
    dropout: float = 0.1,
    cosine_lr: bool = False,
    lr_schedule: str | None = None,
    lr_decay: float = 0.5,
    plateau_patience: int = 4,
    min_lr: float = 1e-6,
    exponential_gamma: float = 0.99,
    warmup_epochs: int = 0,
) -> tuple[torch.nn.Module, list[dict]]:
    """
    Train sector multi-head model (GRU/LSTM or Transformer). Expects ``data`` with
    ``X_train, R_train, X_val, R_val`` and ``R_*`` shape (N, G, 2).

    ``lr_schedule``: ``constant`` | ``cosine`` | ``warmup_cosine`` | ``plateau`` | ``exponential``.
    Legacy ``cosine_lr=True`` is equivalent to ``lr_schedule="cosine"``.
    For ``plateau``, ``lr_decay`` is the factor applied when validation loss plateaus.
    For ``exponential``, ``exponential_gamma`` is the per-epoch multiplicative decay.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = _to_float_tensor(data["X_train"])
    R_train = _to_float_tensor(data["R_train"])
    X_val = _to_float_tensor(data["X_val"])
    R_val = _to_float_tensor(data["R_val"])
    n_features = X_train.shape[2]
    n_sectors = data["n_sectors"]
    seq_len = X_train.shape[1]

    model = build_sector_head_model(
        n_features=n_features,
        n_sectors=n_sectors,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        model_type=model_type,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched_name = _resolve_lr_schedule(lr_schedule, cosine_lr)
    scheduler = _build_lr_scheduler(
        optimizer,
        sched_name,
        epochs,
        plateau_factor=lr_decay,
        plateau_patience=plateau_patience,
        min_lr=min_lr,
        exponential_gamma=exponential_gamma,
        warmup_epochs=warmup_epochs,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    if verbose:
        n_tr, n_val = X_train.shape[0], X_val.shape[0]
        print(
            f"[IPO] Sector-head training  device={device}  sectors={n_sectors}  "
            f"train_windows={n_tr}  val_windows={n_val}  epochs(max)={epochs}  patience={patience}  "
            f"lr_schedule={sched_name}",
            flush=True,
        )

    for epoch in range(epochs):
        train_loss, train_comp, train_diag = train_epoch_sector_heads(
            model, X_train, R_train, optimizer, device,
            batch_size=batch_size,
            lambda_cvar=lambda_cvar,
            lambda_turnover=lambda_turnover,
            lambda_vol=lambda_vol,
            lambda_path=lambda_path,
            lambda_diversify=lambda_diversify,
            min_weight=min_weight,
            lambda_vol_excess=lambda_vol_excess,
            target_vol_annual=target_vol_annual,
            mean_return_weight=mean_return_weight,
            log_growth_weight=log_growth_weight,
            cvar_temperature=cvar_temperature,
            mean_loss_mode=mean_loss_mode,
            huber_delta=huber_delta,
            winsor_abs=winsor_abs,
            max_grad_norm=max_grad_norm,
            grad_norm_mode=grad_norm_mode,
        )
        val_loss, val_comp = validate_sector_heads(
            model, X_val, R_val, device,
            batch_size=batch_size,
            lambda_cvar=lambda_cvar,
            lambda_turnover=lambda_turnover,
            lambda_vol=lambda_vol,
            lambda_path=lambda_path,
            lambda_diversify=lambda_diversify,
            min_weight=min_weight,
            lambda_vol_excess=lambda_vol_excess,
            target_vol_annual=target_vol_annual,
            mean_return_weight=mean_return_weight,
            log_growth_weight=log_growth_weight,
            cvar_temperature=cvar_temperature,
            mean_loss_mode=mean_loss_mode,
            huber_delta=huber_delta,
        )
        train_tb = train_comp.pop("term_breakdown", {})
        val_tb = val_comp.pop("term_breakdown", {})
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_comp.items()},
            **{f"val_{k}": v for k, v in val_comp.items()},
            **{
                f"train_diag_{k}": v
                for k, v in train_diag.items()
                if k != "train_term_breakdown" and isinstance(v, (int, float))
            },
            "train_term_breakdown": train_tb,
            "val_term_breakdown": val_tb,
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        _scheduler_step(scheduler, val_loss)
        row["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(row)

        if verbose and log_every > 0:
            ep = epoch + 1
            if ep == 1 or ep % log_every == 0 or ep == epochs:
                gd = train_diag
                print(
                    f"  [IPO] epoch {ep}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                    f"lr={row['lr']:.2e}  |grad|_pre μ={gd['grad_norm_pre_mean']:.3e} max={gd['grad_norm_pre_max']:.3e}  "
                    f"nan_batches(loss/grad)={gd['nan_loss_batches']}/{gd['nan_grad_batches']}",
                    flush=True,
                )
                if log_loss_term_breakdown and train_tb:
                    print(_format_term_breakdown_table(train_tb, "train"), flush=True)
                    print(_format_term_breakdown_table(val_tb, "val"), flush=True)

        if epochs_no_improve >= patience:
            if verbose:
                print(
                    f"[IPO] Early stopping at epoch {epoch + 1} (no val improvement for {patience} epochs).",
                    flush=True,
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    if checkpoint_dir:
        ckpt = Path(checkpoint_dir)
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "history": history}, ckpt / "best_sector_heads.pt")

    if verbose:
        print(
            f"[IPO] Sector-head training finished  epochs={len(history)}  best_val_loss={best_val_loss:.6f}",
            flush=True,
        )

    return model, history


def run_training(
    data: dict,
    device: Optional[torch.device] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 10,
    checkpoint_dir: Optional[str] = None,
    lambda_cvar: float = 0.5,
    lambda_turnover: float = 0.01,
    lambda_vol: float = 0.5,
    lambda_path: float = 0.01,
    lambda_diversify: float = 0.0,
    min_weight: float = 0.1,
    lambda_vol_excess: float = 0.0,
    target_vol_annual: float = 0.20,
    mean_return_weight: float = 1.0,
    log_growth_weight: float = 0.0,
    cvar_temperature: float = 0.1,
    mean_loss_mode: str = "linear",
    huber_delta: float = 0.02,
    winsor_abs: Optional[float] = None,
    max_grad_norm: float = 1.0,
    grad_norm_mode: GradNormMode = "clip",
    hidden_size: int = 64,
    num_layers: int = 1,
    model_type: str = "gru",
    verbose: bool = True,
    log_every: int = 1,
    log_loss_term_breakdown: bool = True,
    weight_decay: float = 1e-5,
    dropout: float = 0.1,
    cosine_lr: bool = False,
    lr_schedule: str | None = None,
    lr_decay: float = 0.5,
    plateau_patience: int = 4,
    min_lr: float = 1e-6,
    exponential_gamma: float = 0.99,
    warmup_epochs: int = 0,
) -> tuple[torch.nn.Module, list[dict]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = _to_float_tensor(data["X_train"])
    R_train = _to_float_tensor(data["R_train"])
    X_val = _to_float_tensor(data["X_val"])
    R_val = _to_float_tensor(data["R_val"])
    n_features = X_train.shape[2]
    n_assets = data["n_assets"]
    seq_len = X_train.shape[1]

    model = build_model(
        n_features=n_features,
        n_assets=n_assets,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        model_type=model_type,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched_name = _resolve_lr_schedule(lr_schedule, cosine_lr)
    scheduler = _build_lr_scheduler(
        optimizer,
        sched_name,
        epochs,
        plateau_factor=lr_decay,
        plateau_patience=plateau_patience,
        min_lr=min_lr,
        exponential_gamma=exponential_gamma,
        warmup_epochs=warmup_epochs,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    if verbose:
        n_tr, n_val = X_train.shape[0], X_val.shape[0]
        print(
            f"[IPO] Training running  device={device}  train_windows={n_tr}  val_windows={n_val}  "
            f"epochs(max)={epochs}  patience={patience}  lr_schedule={sched_name}",
            flush=True,
        )

    for epoch in range(epochs):
        train_loss, train_comp, train_diag = train_epoch(
            model, X_train, R_train, optimizer, device,
            batch_size=batch_size,
            lambda_cvar=lambda_cvar,
            lambda_turnover=lambda_turnover,
            lambda_vol=lambda_vol,
            lambda_path=lambda_path,
            lambda_diversify=lambda_diversify,
            min_weight=min_weight,
            lambda_vol_excess=lambda_vol_excess,
            target_vol_annual=target_vol_annual,
            mean_return_weight=mean_return_weight,
            log_growth_weight=log_growth_weight,
            cvar_temperature=cvar_temperature,
            mean_loss_mode=mean_loss_mode,
            huber_delta=huber_delta,
            winsor_abs=winsor_abs,
            max_grad_norm=max_grad_norm,
            grad_norm_mode=grad_norm_mode,
        )
        val_loss, val_comp = validate(
            model, X_val, R_val, device,
            batch_size=batch_size,
            lambda_cvar=lambda_cvar,
            lambda_turnover=lambda_turnover,
            lambda_vol=lambda_vol,
            lambda_path=lambda_path,
            lambda_diversify=lambda_diversify,
            min_weight=min_weight,
            lambda_vol_excess=lambda_vol_excess,
            target_vol_annual=target_vol_annual,
            mean_return_weight=mean_return_weight,
            log_growth_weight=log_growth_weight,
            cvar_temperature=cvar_temperature,
            mean_loss_mode=mean_loss_mode,
            huber_delta=huber_delta,
        )
        train_tb = train_comp.pop("term_breakdown", {})
        val_tb = val_comp.pop("term_breakdown", {})
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_comp.items()},
            **{f"val_{k}": v for k, v in val_comp.items()},
            **{
                f"train_diag_{k}": v
                for k, v in train_diag.items()
                if k != "train_term_breakdown" and isinstance(v, (int, float))
            },
            "train_term_breakdown": train_tb,
            "val_term_breakdown": val_tb,
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        _scheduler_step(scheduler, val_loss)
        row["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(row)

        if verbose and log_every > 0:
            ep = epoch + 1
            if ep == 1 or ep % log_every == 0 or ep == epochs:
                gd = train_diag
                print(
                    f"  [IPO] epoch {ep}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                    f"lr={row['lr']:.2e}  |grad|_pre μ={gd['grad_norm_pre_mean']:.3e} max={gd['grad_norm_pre_max']:.3e}  "
                    f"nan_batches(loss/grad)={gd['nan_loss_batches']}/{gd['nan_grad_batches']}",
                    flush=True,
                )
                if log_loss_term_breakdown and train_tb:
                    print(_format_term_breakdown_table(train_tb, "train"), flush=True)
                    print(_format_term_breakdown_table(val_tb, "val"), flush=True)

        if epochs_no_improve >= patience:
            if verbose:
                print(
                    f"[IPO] Early stopping at epoch {epoch + 1} (no val improvement for {patience} epochs).",
                    flush=True,
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    if checkpoint_dir:
        ckpt = Path(checkpoint_dir)
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "history": history}, ckpt / "best.pt")

    if verbose:
        print(
            f"[IPO] Training finished  epochs_completed={len(history)}  best_val_loss={best_val_loss:.6f}",
            flush=True,
        )

    return model, history
