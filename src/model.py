"""
Core optimization interface (Week 4: model.py).

Re-exports policy network, objective scoring, and training so that
notebooks and scripts can use a single entry point for "the model."
"""

from src.policy_network import IPOPolicyNetwork, sample_and_log_prob
from src.objective import score
from src.train_policy import train_reinforce
from src.backtest import backtest_all, backtest_all_with_decisions
from src.policy import PolicyParams, decide_trade

__all__ = [
    "IPOPolicyNetwork",
    "sample_and_log_prob",
    "score",
    "train_reinforce",
    "backtest_all",
    "backtest_all_with_decisions",
    "PolicyParams",
    "decide_trade",
]
