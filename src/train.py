"""
Training loop for IPO portfolio allocator.

Forward → portfolio returns → L → backward; validate; save best checkpoint.

Learning-rate schedules (dynamic step sizes): ``constant``, ``cosine``,
``plateau`` (ReduceLROnPlateau on validation loss), ``exponential`` (per-epoch decay).
"""
from __future__ import annotations

import torch
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any, Literal, Optional

from .model import build_model, build_sector_head_model

GradNormMode = Literal["clip", "rescale"]
from .losses import combined_loss, combined_loss_sector_heads

LrSchedule = Literal["constant", "cosine", "plateau", "exponential"]


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
        if s not in ("constant", "cosine", "plateau", "exponential"):
            raise ValueError(
                f"lr_schedule must be constant|cosine|plateau|exponential; got {lr_schedule!r}"
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
) -> Any:
    if schedule == "constant":
        return None
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
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
    raise ValueError(f"grad_norm_mode must be clip|rescale; got {mode!r}")


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
    max_grad_norm: float = 1.0,
    grad_norm_mode: GradNormMode = "clip",
) -> tuple[float, dict]:
    """``mean_return_weight`` / ``log_growth_weight``: see :func:`src.losses.combined_loss`.
    ``grad_norm_mode``: ``clip`` = PyTorch norm clip; ``rescale`` = fix global L2 norm to ``max_grad_norm``."""
    model.train()
    n = X.shape[0]
    total_loss = 0.0
    n_batches = 0
    acc_components = {}
    last_w: Optional[torch.Tensor] = None
    # Batches in chronological order (rolling windows by prediction date).
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if end - start < 1:
            continue
        x = torch.as_tensor(X[start:end], device=device, dtype=torch.float32)
        r = torch.as_tensor(R[start:end], device=device, dtype=torch.float32)
        w = model(x)
        prev = _weights_prev_consecutive(w, last_w, start_idx=start)
        loss, components = combined_loss(
            w, r,
            weights_prev=prev,
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
        )
        optimizer.zero_grad()
        loss.backward()
        _apply_gradient_normalization(model, max_grad_norm=max_grad_norm, mode=grad_norm_mode)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        for k, v in components.items():
            acc_components[k] = acc_components.get(k, 0.0) + v
        last_w = w[-1].detach()
    if n_batches == 0:
        return 0.0, {}
    avg = {k: v / n_batches for k, v in acc_components.items()}
    return total_loss / n_batches, avg


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
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    acc_components = {}
    last_w: Optional[torch.Tensor] = None
    for start in range(0, X.shape[0], batch_size):
        x = torch.as_tensor(X[start : start + batch_size], device=device, dtype=torch.float32)
        r = torch.as_tensor(R[start : start + batch_size], device=device, dtype=torch.float32)
        w = model(x)
        prev = _weights_prev_consecutive(w, last_w, start_idx=start)
        loss, components = combined_loss(
            w, r,
            weights_prev=prev,
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
        )
        total_loss += loss.item()
        n_batches += 1
        for k, v in components.items():
            acc_components[k] = acc_components.get(k, 0.0) + v
        last_w = w[-1].detach()
    if n_batches == 0:
        return 0.0, {}
    avg = {k: v / n_batches for k, v in acc_components.items()}
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
    max_grad_norm: float = 1.0,
    grad_norm_mode: GradNormMode = "clip",
) -> tuple[float, dict]:
    """R: (N, G, 2); model(x) -> (B, G, 2)."""
    model.train()
    n = X.shape[0]
    total_loss = 0.0
    n_batches = 0
    acc_components = {}
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
    )
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if end - start < 1:
            continue
        x = torch.as_tensor(X[start:end], device=device, dtype=torch.float32)
        r = torch.as_tensor(R[start:end], device=device, dtype=torch.float32)
        w = model(x)
        prev = _weights_prev_consecutive(w, last_w, start_idx=start)
        loss, components = combined_loss_sector_heads(w, r, weights_prev=prev, **loss_kw)
        optimizer.zero_grad()
        loss.backward()
        _apply_gradient_normalization(model, max_grad_norm=max_grad_norm, mode=grad_norm_mode)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        for k, v in components.items():
            acc_components[k] = acc_components.get(k, 0.0) + v
        last_w = w[-1].detach()
    if n_batches == 0:
        return 0.0, {}
    avg = {k: v / n_batches for k, v in acc_components.items()}
    return total_loss / n_batches, avg


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
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    acc_components = {}
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
    )
    for start in range(0, X.shape[0], batch_size):
        x = torch.as_tensor(X[start : start + batch_size], device=device, dtype=torch.float32)
        r = torch.as_tensor(R[start : start + batch_size], device=device, dtype=torch.float32)
        w = model(x)
        prev = _weights_prev_consecutive(w, last_w, start_idx=start)
        loss, components = combined_loss_sector_heads(w, r, weights_prev=prev, **loss_kw)
        total_loss += loss.item()
        n_batches += 1
        for k, v in components.items():
            acc_components[k] = acc_components.get(k, 0.0) + v
        last_w = w[-1].detach()
    if n_batches == 0:
        return 0.0, {}
    avg = {k: v / n_batches for k, v in acc_components.items()}
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
    max_grad_norm: float = 1.0,
    grad_norm_mode: GradNormMode = "clip",
    hidden_size: int = 64,
    num_layers: int = 1,
    model_type: str = "gru",
    verbose: bool = True,
    log_every: int = 1,
    weight_decay: float = 1e-5,
    dropout: float = 0.1,
    cosine_lr: bool = False,
    lr_schedule: str | None = None,
    lr_decay: float = 0.5,
    plateau_patience: int = 4,
    min_lr: float = 1e-6,
    exponential_gamma: float = 0.99,
) -> tuple[torch.nn.Module, list[dict]]:
    """
    Train sector multi-head model (GRU/LSTM or Transformer). Expects ``data`` with
    ``X_train, R_train, X_val, R_val`` and ``R_*`` shape (N, G, 2).

    ``lr_schedule``: ``constant`` | ``cosine`` | ``plateau`` | ``exponential``.
    Legacy ``cosine_lr=True`` is equivalent to ``lr_schedule="cosine"``.
    For ``plateau``, ``lr_decay`` is the factor applied when validation loss plateaus.
    For ``exponential``, ``exponential_gamma`` is the per-epoch multiplicative decay.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = data["X_train"]
    R_train = data["R_train"]
    X_val = data["X_val"]
    R_val = data["R_val"]
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
        train_loss, train_comp = train_epoch_sector_heads(
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
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_comp.items()},
            **{f"val_{k}": v for k, v in val_comp.items()},
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
                print(
                    f"  [IPO] epoch {ep}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                    f"lr={row['lr']:.2e}",
                    flush=True,
                )

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
    max_grad_norm: float = 1.0,
    grad_norm_mode: GradNormMode = "clip",
    hidden_size: int = 64,
    num_layers: int = 1,
    model_type: str = "gru",
    verbose: bool = True,
    log_every: int = 1,
    weight_decay: float = 1e-5,
    dropout: float = 0.1,
    cosine_lr: bool = False,
    lr_schedule: str | None = None,
    lr_decay: float = 0.5,
    plateau_patience: int = 4,
    min_lr: float = 1e-6,
    exponential_gamma: float = 0.99,
) -> tuple[torch.nn.Module, list[dict]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = data["X_train"]
    R_train = data["R_train"]
    X_val = data["X_val"]
    R_val = data["R_val"]
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
        train_loss, train_comp = train_epoch(
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
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_comp.items()},
            **{f"val_{k}": v for k, v in val_comp.items()},
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
                print(
                    f"  [IPO] epoch {ep}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                    f"lr={row['lr']:.2e}",
                    flush=True,
                )

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
