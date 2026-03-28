"""
Training loop for IPO portfolio allocator.

Forward → portfolio returns → L → backward; validate; save best checkpoint.
"""
from __future__ import annotations

import torch
from pathlib import Path
from typing import Optional

from .model import build_model, build_sector_head_model
from .losses import combined_loss, combined_loss_sector_heads


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
) -> tuple[float, dict]:
    model.train()
    n = X.shape[0]
    total_loss = 0.0
    n_batches = 0
    acc_components = {}
    weights_prev = None
    # Batches in chronological order (rolling windows by prediction date).
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if end - start < 2:
            continue
        x = torch.as_tensor(X[start:end], device=device, dtype=torch.float32)
        r = torch.as_tensor(R[start:end], device=device, dtype=torch.float32)
        w = model(x)
        prev = weights_prev if (weights_prev is not None and weights_prev.shape[0] == w.shape[0]) else None
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
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        for k, v in components.items():
            acc_components[k] = acc_components.get(k, 0.0) + v
        weights_prev = w.detach()
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
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    acc_components = {}
    weights_prev = None
    for start in range(0, X.shape[0], batch_size):
        x = torch.as_tensor(X[start : start + batch_size], device=device, dtype=torch.float32)
        r = torch.as_tensor(R[start : start + batch_size], device=device, dtype=torch.float32)
        w = model(x)
        prev = weights_prev if (weights_prev is not None and weights_prev.shape[0] == w.shape[0]) else None
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
        )
        total_loss += loss.item()
        n_batches += 1
        for k, v in components.items():
            acc_components[k] = acc_components.get(k, 0.0) + v
        weights_prev = w
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
) -> tuple[float, dict]:
    """R: (N, G, 2); model(x) -> (B, G, 2)."""
    model.train()
    n = X.shape[0]
    total_loss = 0.0
    n_batches = 0
    acc_components = {}
    weights_prev = None
    loss_kw = dict(
        lambda_cvar=lambda_cvar,
        lambda_turnover=lambda_turnover,
        lambda_vol=lambda_vol,
        lambda_path=lambda_path,
        lambda_diversify=lambda_diversify,
        min_weight=min_weight,
        lambda_vol_excess=lambda_vol_excess,
        target_vol_annual=target_vol_annual,
    )
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if end - start < 2:
            continue
        x = torch.as_tensor(X[start:end], device=device, dtype=torch.float32)
        r = torch.as_tensor(R[start:end], device=device, dtype=torch.float32)
        w = model(x)
        prev = weights_prev if (weights_prev is not None and weights_prev.shape[0] == w.shape[0]) else None
        loss, components = combined_loss_sector_heads(w, r, weights_prev=prev, **loss_kw)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        for k, v in components.items():
            acc_components[k] = acc_components.get(k, 0.0) + v
        weights_prev = w.detach()
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
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    acc_components = {}
    weights_prev = None
    loss_kw = dict(
        lambda_cvar=lambda_cvar,
        lambda_turnover=lambda_turnover,
        lambda_vol=lambda_vol,
        lambda_path=lambda_path,
        lambda_diversify=lambda_diversify,
        min_weight=min_weight,
        lambda_vol_excess=lambda_vol_excess,
        target_vol_annual=target_vol_annual,
    )
    for start in range(0, X.shape[0], batch_size):
        x = torch.as_tensor(X[start : start + batch_size], device=device, dtype=torch.float32)
        r = torch.as_tensor(R[start : start + batch_size], device=device, dtype=torch.float32)
        w = model(x)
        prev = weights_prev if (weights_prev is not None and weights_prev.shape[0] == w.shape[0]) else None
        loss, components = combined_loss_sector_heads(w, r, weights_prev=prev, **loss_kw)
        total_loss += loss.item()
        n_batches += 1
        for k, v in components.items():
            acc_components[k] = acc_components.get(k, 0.0) + v
        weights_prev = w
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
    hidden_size: int = 64,
    num_layers: int = 1,
    model_type: str = "gru",
    verbose: bool = True,
    log_every: int = 1,
) -> tuple[torch.nn.Module, list[dict]]:
    """
    Train sector multi-head model (GRU/LSTM or Transformer). Expects ``data`` with
    ``X_train, R_train, X_val, R_val`` and ``R_*`` shape (N, G, 2).
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
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    if verbose:
        n_tr, n_val = X_train.shape[0], X_val.shape[0]
        print(
            f"[IPO] Sector-head training  device={device}  sectors={n_sectors}  "
            f"train_windows={n_tr}  val_windows={n_val}  epochs(max)={epochs}  patience={patience}",
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
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_comp.items()},
            **{f"val_{k}": v for k, v in val_comp.items()},
        })
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and log_every > 0:
            ep = epoch + 1
            if ep == 1 or ep % log_every == 0 or ep == epochs:
                print(
                    f"  [IPO] epoch {ep}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}",
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
    hidden_size: int = 64,
    num_layers: int = 1,
    model_type: str = "gru",
    verbose: bool = True,
    log_every: int = 1,
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
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    if verbose:
        n_tr, n_val = X_train.shape[0], X_val.shape[0]
        print(
            f"[IPO] Training running  device={device}  train_windows={n_tr}  val_windows={n_val}  "
            f"epochs(max)={epochs}  patience={patience}",
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
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_comp.items()},
            **{f"val_{k}": v for k, v in val_comp.items()},
        })
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and log_every > 0:
            ep = epoch + 1
            if ep == 1 or ep % log_every == 0 or ep == epochs:
                print(
                    f"  [IPO] epoch {ep}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}",
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
