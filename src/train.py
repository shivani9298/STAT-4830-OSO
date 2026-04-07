"""
Training loop for IPO portfolio allocator.

Forward → portfolio returns → L → backward; validate; save best checkpoint.
"""
from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from typing import Optional

from .model import build_model
from .losses import combined_loss


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(0, dtype=np.float64)
    if window <= 1 or len(x) < window:
        return np.array([float(np.sum(x))], dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64)
    return np.convolve(x, kernel, mode="valid")


def rolling_tail_excess_objective(
    weights: np.ndarray,
    returns: np.ndarray,
    window: int = 21,
    tail_quantile: float = 0.10,
    drawdown_penalty: float = 0.0,
) -> tuple[float, dict]:
    """
    Objective based on lower tail of rolling excess return vs equal-weight benchmark.

    Lower is better for optimization:
      objective = -quantile_q( rolling_sum( excess_vs_ew ) ) + penalty*|max_drawdown|
    """
    if len(weights) == 0 or len(returns) == 0:
        return 0.0, {
            "tail_q_excess": 0.0,
            "mean_excess": 0.0,
            "max_drawdown": 0.0,
            "rolling_window": float(window),
        }

    port_ret = (weights * returns).sum(axis=1)
    ew_ret = returns.mean(axis=1)
    excess = port_ret - ew_ret
    rolling_excess = _rolling_sum(excess, window=window)
    tail_q = float(np.quantile(rolling_excess, tail_quantile))
    mean_excess = float(np.mean(excess))

    wealth = np.cumprod(1.0 + port_ret)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth / np.clip(peak, 1e-12, None)) - 1.0
    max_dd = float(np.min(dd)) if len(dd) else 0.0

    objective = -tail_q + drawdown_penalty * abs(min(max_dd, 0.0))
    return objective, {
        "tail_q_excess": tail_q,
        "mean_excess": mean_excess,
        "max_drawdown": max_dd,
        "rolling_window": float(window),
    }


def _predict_weights_np(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    if X.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            x = torch.as_tensor(X[start : start + batch_size], device=device, dtype=torch.float32)
            w = model(x)
            out.append(w.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def _grad_l2_and_near_zero_fraction(model: torch.nn.Module, eps: float = 1e-10) -> tuple[float, float]:
    sq_sum = 0.0
    n = 0
    n_near_zero = 0
    for p in model.parameters():
        g = p.grad
        if g is None:
            continue
        g = g.detach()
        sq_sum += float(torch.sum(g * g).item())
        n += g.numel()
        n_near_zero += int((g.abs() <= eps).sum().item())
    if n == 0:
        return 0.0, 1.0
    return float(np.sqrt(sq_sum)), float(n_near_zero / n)


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
    lambda_vs_ew: float = 0.0,
) -> tuple[float, dict]:
    model.train()
    n = X.shape[0]
    perm = np.random.permutation(n)
    total_loss = 0.0
    n_batches = 0
    acc_components = {}
    weights_prev = None
    grad_l2_sum = 0.0
    grad_zero_frac_sum = 0.0
    ipo_std_sum = 0.0
    shift_5050_sum = 0.0
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        if len(idx) < 2:
            continue
        x = torch.as_tensor(X[idx], device=device, dtype=torch.float32)
        r = torch.as_tensor(R[idx], device=device, dtype=torch.float32)
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
            lambda_vs_ew=lambda_vs_ew,
        )
        optimizer.zero_grad()
        loss.backward()
        grad_l2, grad_zero_frac = _grad_l2_and_near_zero_fraction(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        grad_l2_sum += grad_l2
        grad_zero_frac_sum += grad_zero_frac
        if w.shape[1] > 1:
            ipo_w = w[:, 1].detach()
            ipo_std_sum += float(ipo_w.std(unbiased=False).item())
            shift_5050_sum += float(torch.mean(torch.abs(ipo_w - 0.5)).item())
        for k, v in components.items():
            acc_components[k] = acc_components.get(k, 0.0) + v
        weights_prev = w.detach()
    if n_batches == 0:
        return 0.0, {}
    avg = {k: v / n_batches for k, v in acc_components.items()}
    avg["diag_grad_l2"] = grad_l2_sum / n_batches
    avg["diag_grad_near_zero_frac"] = grad_zero_frac_sum / n_batches
    avg["diag_ipo_weight_std"] = ipo_std_sum / n_batches if ipo_std_sum > 0 else 0.0
    avg["diag_shift_from_5050"] = shift_5050_sum / n_batches if shift_5050_sum > 0 else 0.0
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
    lambda_vs_ew: float = 0.0,
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    acc_components = {}
    weights_prev = None
    ipo_std_sum = 0.0
    shift_5050_sum = 0.0
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
            lambda_vs_ew=lambda_vs_ew,
        )
        total_loss += loss.item()
        n_batches += 1
        if w.shape[1] > 1:
            ipo_w = w[:, 1]
            ipo_std_sum += float(ipo_w.std(unbiased=False).item())
            shift_5050_sum += float(torch.mean(torch.abs(ipo_w - 0.5)).item())
        for k, v in components.items():
            acc_components[k] = acc_components.get(k, 0.0) + v
        weights_prev = w
    if n_batches == 0:
        return 0.0, {}
    avg = {k: v / n_batches for k, v in acc_components.items()}
    avg["diag_ipo_weight_std"] = ipo_std_sum / n_batches if ipo_std_sum > 0 else 0.0
    avg["diag_shift_from_5050"] = shift_5050_sum / n_batches if shift_5050_sum > 0 else 0.0
    return total_loss / n_batches, avg


def run_training(
    data: dict,
    device: Optional[torch.device] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    lr_decay: float = 0.1,
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
    lambda_vs_ew: float = 0.0,
    hidden_size: int = 64,
    num_layers: int = 1,
    model_type: str = "gru",
    selection_metric: str = "val_loss",
    rolling_window: int = 21,
    rolling_tail_quantile: float = 0.10,
    selection_drawdown_penalty: float = 0.0,
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
    # Drop LR by lr_decay after the first epoch to escape the noise floor,
    # then hold steady for the remainder of training.
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=lr_decay)

    best_selection_objective = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

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
            lambda_vs_ew=lambda_vs_ew,
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
            lambda_vs_ew=lambda_vs_ew,
        )
        selection_obj = val_loss
        selection_metrics = {}
        if selection_metric == "rolling_tail_excess":
            val_w = _predict_weights_np(model, X_val, device=device, batch_size=batch_size)
            selection_obj, selection_metrics = rolling_tail_excess_objective(
                val_w,
                R_val,
                window=rolling_window,
                tail_quantile=rolling_tail_quantile,
                drawdown_penalty=selection_drawdown_penalty,
            )
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "selection_objective": selection_obj,
            "lr": current_lr,
            **{f"train_{k}": v for k, v in train_comp.items()},
            **{f"val_{k}": v for k, v in val_comp.items()},
            **{f"val_sel_{k}": v for k, v in selection_metrics.items()},
        })
        msg = (
            f"[epoch {epoch + 1:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"sel_obj={selection_obj:.6f} "
            f"lr={current_lr:.2e}"
        )
        if "tail_q_excess" in selection_metrics:
            msg += f" tail_q_excess={selection_metrics['tail_q_excess']:.6f}"
        if "diag_grad_l2" in train_comp:
            msg += f" grad_l2={train_comp['diag_grad_l2']:.3e}"
        if "diag_shift_from_5050" in val_comp:
            msg += f" val_|ipo-0.5|={val_comp['diag_shift_from_5050']:.4f}"
        print(msg, flush=True)
        if selection_obj < best_selection_objective:
            best_selection_objective = selection_obj
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(
                f"[early_stop] no improvement for {patience} epochs; "
                f"best_selection_objective={best_selection_objective:.6f}",
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

    return model, history
