"""
Training loop for IPO portfolio allocator.

Forward → portfolio returns → L → backward; validate; save best checkpoint.
"""
from __future__ import annotations

import random

import numpy as np
import torch
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


def mean_excess_vs_ew_selection_objective(
    weights: np.ndarray,
    returns: np.ndarray,
) -> tuple[float, dict]:
    """
    Validation metric for checkpointing: maximize average daily excess vs equal-weight
    (same definition as training: ew return = row-wise mean of asset returns).

    Lower is better for the optimizer loop: objective = -mean_excess.
    """
    if len(weights) == 0 or len(returns) == 0:
        return 0.0, {"mean_excess_vs_ew": 0.0}
    port_ret = (weights * returns).sum(axis=1)
    ew_ret = returns.mean(axis=1)
    mean_excess = float(np.mean(port_ret - ew_ret))
    return -mean_excess, {"mean_excess_vs_ew": mean_excess}


def path_metrics_numpy(port_ret: np.ndarray, ann_factor: float = 252.0) -> dict[str, float]:
    """
    Chronological portfolio simple returns → compound return, Sharpe, Sortino, max drawdown.
    """
    r = np.asarray(port_ret, dtype=np.float64)
    if len(r) == 0:
        return {
            "compound_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
        }
    mu = float(np.mean(r))
    sd = float(np.std(r))
    sharpe = (mu / sd * np.sqrt(ann_factor)) if sd > 1e-12 else 0.0
    downside = r[r < 0.0]
    dsd = float(np.std(downside)) if len(downside) > 1 else 0.0
    sortino = (mu / dsd * np.sqrt(ann_factor)) if dsd > 1e-12 else sharpe
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / np.clip(peak, 1e-12, None) - 1.0
    max_dd = float(np.min(dd))
    compound_return = float(np.prod(1.0 + r) - 1.0)
    return {
        "compound_return": compound_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
    }


def path_selection_objective(
    weights: np.ndarray,
    returns: np.ndarray,
    metric: str,
    selection_drawdown_penalty: float = 0.0,
) -> tuple[float, dict]:
    """
    Map validation path statistics to a scalar for early stopping (lower is better).

    Metrics val_compound_return, val_sharpe, val_sortino use negated headline stats.
    val_max_drawdown uses -max_drawdown so that shallower drawdowns win.
    val_retail_composite blends Sharpe, Sortino, and an optional drawdown penalty.
    """
    port_ret = (weights * returns).sum(axis=1)
    m = path_metrics_numpy(port_ret)
    out = {**m, "selection_metric_name": metric}
    if metric == "val_compound_return":
        return -m["compound_return"], out
    if metric == "val_sharpe":
        return -m["sharpe"], out
    if metric == "val_sortino":
        return -m["sortino"], out
    if metric == "val_max_drawdown":
        # max_drawdown is ≤ 0; we minimize -max_dd so -0.2 beats -0.5
        return -m["max_drawdown"], out
    if metric == "val_retail_composite":
        dd_pen = abs(min(m["max_drawdown"], 0.0))
        obj = -(m["sharpe"] + 0.5 * m["sortino"]) + selection_drawdown_penalty * dd_pen
        return obj, out
    raise ValueError(f"unknown path selection metric: {metric}")


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
    lambda_log_return: float = 0.0,
    train_segment_len: int = 0,
    lambda_segment_log: float = 0.0,
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
            lambda_log_return=lambda_log_return,
        )
        if lambda_segment_log > 0.0 and train_segment_len >= 2 and n >= train_segment_len:
            ss = int(random.randint(0, n - train_segment_len))
            xseg = torch.as_tensor(
                X[ss : ss + train_segment_len], device=device, dtype=torch.float32
            )
            rseg = torch.as_tensor(
                R[ss : ss + train_segment_len], device=device, dtype=torch.float32
            )
            wseg = model(xseg)
            pseg = (wseg * rseg).sum(dim=-1)
            loss_seg = -torch.log1p(pseg.clamp(min=-0.999)).mean()
            loss = loss + lambda_segment_log * loss_seg
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
    lambda_log_return: float = 0.0,
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
            lambda_log_return=lambda_log_return,
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
    lambda_log_return: float = 0.0,
    train_segment_len: int = 0,
    lambda_segment_log: float = 0.0,
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
            lambda_log_return=lambda_log_return,
            train_segment_len=train_segment_len,
            lambda_segment_log=lambda_segment_log,
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
            lambda_log_return=lambda_log_return,
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
        elif selection_metric == "mean_excess_vs_ew":
            val_w = _predict_weights_np(model, X_val, device=device, batch_size=batch_size)
            selection_obj, selection_metrics = mean_excess_vs_ew_selection_objective(val_w, R_val)
        elif selection_metric in (
            "val_compound_return",
            "val_sharpe",
            "val_sortino",
            "val_max_drawdown",
            "val_retail_composite",
        ):
            val_w = _predict_weights_np(model, X_val, device=device, batch_size=batch_size)
            selection_obj, selection_metrics = path_selection_objective(
                val_w,
                R_val,
                metric=selection_metric,
                selection_drawdown_penalty=selection_drawdown_penalty,
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
        if "mean_excess_vs_ew" in selection_metrics:
            msg += f" val_mean_excess_vs_ew={selection_metrics['mean_excess_vs_ew']:.6f}"
        if "sharpe" in selection_metrics and "compound_return" in selection_metrics:
            msg += (
                f" val_path Sharpe={selection_metrics['sharpe']:.3f} "
                f"Sortino={selection_metrics['sortino']:.3f} "
                f"compound={selection_metrics['compound_return']:.4f} "
                f"maxDD={selection_metrics['max_drawdown']:.4f}"
            )
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
