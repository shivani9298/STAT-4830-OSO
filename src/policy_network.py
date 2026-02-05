"""
PyTorch policy network for IPO trading (course: How to compute gradients, Autodiff).
Contextual bandit: features -> participate, entry bucket, exit bucket, sizing.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np

from src.data import Episode
from src.features import N_FEATURES, episodes_to_tensor


# Action space (daily MVP)
N_ENTRY_CHOICES = 2   # 0 or 1
N_HOLD_CHOICES = 9    # 1..9 days
W_MAX = 1.0


class IPOPolicyNetwork(nn.Module):
    """
    Maps episode features to action distribution (course: neural net as function, autodiff).
    Outputs: participate logits, entry_day logits, hold_k logits, weight (sigmoid -> [0, w_max]).
    """
    
    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden: int = 32,
        n_entry: int = N_ENTRY_CHOICES,
        n_hold: int = N_HOLD_CHOICES,
        w_max: float = W_MAX,
    ):
        super().__init__()
        self.n_entry = n_entry
        self.n_hold = n_hold
        self.w_max = w_max
        
        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.head_participate = nn.Linear(hidden, 1)
        self.head_entry = nn.Linear(hidden, n_entry)
        self.head_hold = nn.Linear(hidden, n_hold)
        self.head_weight = nn.Linear(hidden, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, n_features). Returns logits for participate, entry, hold; weight in [0, w_max].
        """
        h = self.mlp(x)
        logit_participate = self.head_participate(h).squeeze(-1)   # (B,)
        logit_entry = self.head_entry(h)                           # (B, n_entry)
        logit_hold = self.head_hold(h)                             # (B, n_hold)
        weight = torch.sigmoid(self.head_weight(h).squeeze(-1)) * self.w_max  # (B,)
        return logit_participate, logit_entry, logit_hold, weight
    
    def sample_actions(
        self,
        x: torch.Tensor,
        episodes: List[Episode],
        generator: torch.Generator,
    ) -> Tuple[List[dict], torch.Tensor]:
        """
        Sample actions for each episode; return list of decision dicts and log_prob (for REINFORCE).
        """
        B = x.size(0)
        logit_p, logit_e, logit_h, weight = self.forward(x)
        
        # Sample participate (Bernoulli)
        probs_p = torch.sigmoid(logit_p)
        u = torch.rand(B, device=x.device, generator=generator)
        participate = (u < probs_p).long()  # (B,)
        log_prob_p = torch.where(
            participate == 1,
            torch.log(probs_p.clamp(min=1e-6)),
            torch.log((1 - probs_p).clamp(min=1e-6)),
        )
        
        # Sample entry_day (Categorical 0 or 1)
        probs_e = torch.softmax(logit_e, dim=-1)
        entry_day = torch.multinomial(probs_e, 1, generator=generator).squeeze(-1)  # (B,)
        log_prob_e = torch.log(probs_e.gather(1, entry_day.unsqueeze(1)).squeeze(1).clamp(min=1e-6))
        
        # Sample hold_k (Categorical 1..n_hold)
        probs_h = torch.softmax(logit_h, dim=-1)
        hold_k_idx = torch.multinomial(probs_h, 1, generator=generator).squeeze(-1)  # 0..n_hold-1
        hold_k = hold_k_idx + 1  # 1..n_hold
        log_prob_h = torch.log(probs_h.gather(1, hold_k_idx.unsqueeze(1)).squeeze(1).clamp(min=1e-6))
        
        # Weight is deterministic given state (no sampling) for lower variance; optional: add noise
        w = weight
        
        # Build decision dicts and total log_prob
        decisions = []
        for i in range(B):
            ep = episodes[i]
            N = len(ep.df)
            entry_d = min(int(entry_day[i].item()), N - 1)
            exit_d = min(entry_d + int(hold_k[i].item()), N - 1)
            if exit_d <= entry_d:
                exit_d = min(entry_d + 1, N - 1)
            part = bool(participate[i].item())
            decisions.append({
                "participate": part,
                "entry_day": entry_d,
                "exit_day": exit_d if part else 0,
                "weight": float(w[i].item()) if part else 0.0,
            })
        
        log_prob = log_prob_p + log_prob_e + log_prob_h  # (B,)
        return decisions, log_prob


def sample_and_log_prob(
    policy: IPOPolicyNetwork,
    x: torch.Tensor,
    episodes: List[Episode],
    generator: torch.Generator,
) -> Tuple[List[dict], torch.Tensor]:
    """
    Forward pass (with grad), sample actions, return decisions and differentiable log_prob (REINFORCE).
    """
    logit_p, logit_e, logit_h, weight = policy(x)
    B = x.size(0)
    
    probs_p = torch.sigmoid(logit_p)
    u = torch.rand(B, device=x.device, generator=generator)
    participate = (u < probs_p).long()
    log_prob_p = torch.where(
        participate == 1,
        torch.log(probs_p.clamp(min=1e-6)),
        torch.log((1 - probs_p).clamp(min=1e-6)),
    )
    
    probs_e = torch.softmax(logit_e, dim=-1)
    entry_day = torch.multinomial(probs_e, 1, generator=generator).squeeze(-1)
    log_prob_e = torch.log(probs_e.gather(1, entry_day.unsqueeze(1)).squeeze(1).clamp(min=1e-6))
    
    probs_h = torch.softmax(logit_h, dim=-1)
    hold_k_idx = torch.multinomial(probs_h, 1, generator=generator).squeeze(-1)
    hold_k = hold_k_idx + 1
    log_prob_h = torch.log(probs_h.gather(1, hold_k_idx.unsqueeze(1)).squeeze(1).clamp(min=1e-6))
    
    w = weight
    decisions = []
    for i in range(B):
        ep = episodes[i]
        N = len(ep.df)
        entry_d = min(int(entry_day[i].item()), N - 1)
        exit_d = min(entry_d + int(hold_k[i].item()), N - 1)
        if exit_d <= entry_d:
            exit_d = min(entry_d + 1, N - 1)
        part = bool(participate[i].item())
        decisions.append({
            "participate": part,
            "entry_day": entry_d,
            "exit_day": exit_d if part else 0,
            "weight": float(w[i].item()) if part else 0.0,
        })
    log_prob = log_prob_p + log_prob_e + log_prob_h
    return decisions, log_prob


def policy_network_to_decision_list(
    policy: IPOPolicyNetwork,
    episodes: List[Episode],
    device: torch.device,
    generator: torch.Generator,
) -> Tuple[List[dict], torch.Tensor]:
    """
    Run policy on episodes and sample actions (course: forward pass, sampling).
    """
    x = episodes_to_tensor(episodes, device)
    with torch.no_grad():
        return policy.sample_actions(x, episodes, generator)
