"""
REINFORCE training for IPO policy network (course: SGD, Adam, step-size schedules, validation).
Objective: maximize E[reward] via policy gradient; backtest is non-differentiable.
"""

import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch

from src.data import Episode
from src.features import episodes_to_tensor
from src.policy_network import IPOPolicyNetwork, sample_and_log_prob
from src.backtest import backtest_all_with_decisions
from src.objective import score


def train_reinforce(
    episodes: List[Episode],
    val_episodes: Optional[List[Episode]] = None,
    *,
    n_epochs: int = 50,
    lr: float = 1e-3,
    lr_schedule: str = "constant",  # "constant" | "cosine" | "step"
    cost_bps: float = 10.0,
    lam: float = 1.0,
    alpha: float = 0.9,
    kappa: float = 1.0,
    mu: float = 1.0,
    batch_size: int = 32,
    entropy_coef: float = 0.01,
    seed: int = 0,
    device: Optional[torch.device] = None,
    out_dir: Optional[Path] = None,
    reward_type: str = "net_ret",  # "net_ret" = per-episode return (default); "score" = full fitness per batch
) -> dict:
    """
    REINFORCE training (course: stochastic optimization, Adam, validation).
    
    Reward per episode = net_ret (or risk-adjusted score over batch).
    Loss = - (log_prob * reward).mean() - entropy_coef * entropy.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    policy = IPOPolicyNetwork().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)  # course: Adaptive methods
    
    n = len(episodes)
    if val_episodes is None:
        val_episodes = []
    
    def _reward_from_decisions(ep_list: List[Episode], decisions: List[dict]) -> torch.Tensor:
        res_df, equity = backtest_all_with_decisions(ep_list, decisions, cost_bps)
        sc, _ = score(res_df, equity, lam=lam, alpha=alpha, kappa=kappa, mu=mu)
        if reward_type == "score":
            # Use full fitness score as reward for every episode in batch (same scalar)
            B = len(ep_list)
            return torch.full((B,), float(sc), dtype=torch.float32, device=device)
        # Default: per-episode net return
        net_rets = res_df["net_ret"].values
        return torch.tensor(net_rets, dtype=torch.float32, device=device)
    
    history = {"train_score": [], "val_score": [], "loss": []}
    score_baseline = None  # running baseline for reward_type=="score"
    baseline_momentum = 0.99
    
    for epoch in range(n_epochs):
        # Step-size schedule (course: step-size tuning)
        if lr_schedule == "cosine":
            t = (epoch + 1) / n_epochs
            lr_cur = lr * 0.5 * (1 + np.cos(np.pi * t))
            for g in optimizer.param_groups:
                g["lr"] = lr_cur
        elif lr_schedule == "step" and (epoch + 1) % 20 == 0:
            lr_cur = lr * (0.5 ** ((epoch + 1) // 20))
            for g in optimizer.param_groups:
                g["lr"] = lr_cur
        
        policy.train()
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) == 0:
                continue
            batch_episodes = [episodes[i] for i in idx]
            x = episodes_to_tensor(batch_episodes, device)
            
            decisions, log_prob = sample_and_log_prob(policy, x, batch_episodes, generator)
            rewards = _reward_from_decisions(batch_episodes, decisions)
            # REINFORCE: baseline for variance reduction
            if reward_type == "score":
                # Single scalar reward per batch; use running baseline
                sc_batch = rewards[0].item()
                if score_baseline is None:
                    score_baseline = sc_batch
                else:
                    score_baseline = baseline_momentum * score_baseline + (1 - baseline_momentum) * sc_batch
                baseline = torch.full_like(rewards, score_baseline, device=device)
            else:
                baseline = rewards.mean().detach()
            advantage = rewards - baseline
            loss = -(log_prob * advantage).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        history["loss"].append(avg_loss)
        
        # Training score (full train set, no grad)
        policy.eval()
        with torch.no_grad():
            decs, _ = policy.sample_actions(
                episodes_to_tensor(episodes, device), episodes, generator
            )
            res_df, equity = backtest_all_with_decisions(episodes, decs, cost_bps)
            sc, _ = score(res_df, equity, lam=lam, alpha=alpha, kappa=kappa, mu=mu)
            history["train_score"].append(sc)
        
        val_sc = None
        if val_episodes:
            with torch.no_grad():
                decs_val, _ = policy.sample_actions(
                    episodes_to_tensor(val_episodes, device), val_episodes, generator
                )
                res_val, eq_val = backtest_all_with_decisions(val_episodes, decs_val, cost_bps)
                val_sc, _ = score(res_val, eq_val, lam=lam, alpha=alpha, kappa=kappa, mu=mu)
                history["val_score"].append(val_sc)
        
        log_every = 1 if n_epochs <= 20 else 10
        if (epoch + 1) % log_every == 0 or epoch == 0:
            msg = f"Epoch {epoch+1}/{n_epochs} loss={avg_loss:.6f} train_score={sc:.6f}"
            if val_sc is not None:
                msg += f" val_score={val_sc:.6f}"
            print(msg)
    
    # Build readable summary and interpretation
    epochs_list = [
        {
            "epoch": e + 1,
            "loss": round(history["loss"][e], 6),
            "train_score": round(history["train_score"][e], 6),
            "val_score": round(history["val_score"][e], 6) if history["val_score"] else None,
        }
        for e in range(n_epochs)
    ]
    best_epoch_val = (
        int(np.argmax(history["val_score"]) + 1) if history["val_score"] else None
    )
    best_val_score = (
        max(history["val_score"]) if history["val_score"] else None
    )
    summary = {
        "config": {
            "n_epochs": n_epochs,
            "n_train": n,
            "n_val": len(val_episodes),
            "lr": lr,
            "lr_schedule": lr_schedule,
            "batch_size": batch_size,
            "seed": seed,
        },
        "best_epoch_by_val": best_epoch_val,
        "best_val_score": round(best_val_score, 6) if best_val_score is not None else None,
        "final": {
            "epoch": n_epochs,
            "loss": round(history["loss"][-1], 6),
            "train_score": round(history["train_score"][-1], 6),
            "val_score": round(history["val_score"][-1], 6) if history["val_score"] else None,
        },
        "epochs": epochs_list,
        "interpretation": _interpret_training(history, n_epochs, best_epoch_val, best_val_score),
    }
    
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), out_dir / "policy_network.pt")
        with open(out_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    
    # Print readable summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Train episodes: {n}  |  Val episodes: {len(val_episodes)}")
    if best_val_score is not None:
        print(f"  Best epoch (by val score): {best_epoch_val}  |  Best val score: {best_val_score:.6f}")
    else:
        print("  (no validation set)")
    if history["val_score"]:
        print(f"  Final train score: {history['train_score'][-1]:.6f}  |  Final val score: {history['val_score'][-1]:.6f}")
    else:
        print(f"  Final train score: {history['train_score'][-1]:.6f}")
    print(f"  Final loss: {history['loss'][-1]:.6f}")
    print("=" * 60)
    print("\nInterpretation:")
    for line in summary["interpretation"].split("\n"):
        print(f"  {line}")
    print("=" * 60)
    if out_dir:
        print(f"\nResults saved to: {out_dir}/")
        print(f"  - policy_network.pt")
        print(f"  - training_summary.json")
    
    return {
        "policy": policy,
        "history": history,
        "summary": summary,
        "device": device,
    }


def _interpret_training(
    history: dict,
    n_epochs: int,
    best_epoch_val: Optional[int],
    best_val_score: Optional[float],
) -> str:
    """Short interpretation of training run for JSON and console."""
    lines = []
    final_train = history["train_score"][-1]
    final_val = history["val_score"][-1] if history["val_score"] else None
    
    if final_train > 0:
        lines.append("Train score is positive: policy is profitable on average on training data.")
    else:
        lines.append("Train score is negative: costs/risk penalties outweigh returns on training data (common with small data or strong penalties).")
    
    if final_val is not None:
        if best_val_score and best_val_score > 0:
            lines.append(f"Best validation score ({best_val_score:.4f}) at epoch {best_epoch_val} is positive.")
        else:
            lines.append(f"Validation score stays negative; best epoch by val score: {best_epoch_val}.")
        if len(history["val_score"]) > 1:
            val_std = np.std(history["val_score"])
            if val_std > 0.1:
                lines.append("High validation variance across epochs: consider more epochs or more validation data.")
            elif best_epoch_val and best_epoch_val < n_epochs * 0.5:
                lines.append("Best val score early in training; later epochs may overfit (check epochs curve).")
    lines.append("Score = E[R] - λ*CVaR - κ*Cost - μ*MDD; negative means risk/costs dominate.")
    return "\n".join(lines)
