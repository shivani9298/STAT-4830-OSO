"""
Optimizer module - random search and optimization logic.
Owned by Person D.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from src.data import Episode
from src.policy import PolicyParams, sample_params
from src.backtest import backtest_all
from src.objective import score
from src.logging_utils import log_trial


def random_search(
    episodes: List[Episode],
    n_trials: int,
    seed: int,
    objective_kwargs: Dict[str, Any],
    out_dir: Path
) -> Dict[str, Any]:
    """
    Run random search optimization.
    
    Args:
        episodes: List of trading episodes
        n_trials: Number of trials to run
        seed: Random seed
        objective_kwargs: Keyword arguments for score function
        out_dir: Output directory for artifacts
        
    Returns:
        Dict with best params and results
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(seed)
    trials_path = out_dir / "trials.jsonl"
    
    # Leaderboard (top-K trials)
    leaderboard = []
    
    for trial_idx in range(n_trials):
        # Sample parameters
        params = sample_params(rng)
        
        # Backtest
        results_df, equity_curve = backtest_all(episodes, params, objective_kwargs.get("cost_bps", 10.0))
        
        # Score
        trial_score, metrics = score(
            results_df,
            equity_curve,
            **{k: v for k, v in objective_kwargs.items() if k != "cost_bps"}
        )
        
        # Create trial record (convert numpy types to native Python types)
        trial_record = {
            "trial": int(trial_idx),
            "seed": int(seed),
            "params": {
                "participate_threshold": float(params.participate_threshold),
                "entry_day": int(params.entry_day),
                "hold_k": int(params.hold_k),
                "w_max": float(params.w_max),
                "raw_weight": float(params.raw_weight),
                "use_volume_cap": bool(params.use_volume_cap),
                "vol_cap_mult": float(params.vol_cap_mult)
            },
            "metrics": {k: float(v) for k, v in metrics.items()},
            "score": float(trial_score)
        }
        
        # Log trial
        log_trial(trials_path, trial_record)
        
        # Update leaderboard
        leaderboard.append((trial_score, trial_record))
        leaderboard.sort(key=lambda x: x[0], reverse=True)  # Sort by score descending
        leaderboard = leaderboard[:5]  # Keep top 5
    
    # Get best trial
    best_score, best_record = leaderboard[0]
    
    # Save best params
    best_params_path = out_dir / "best_params.json"
    with open(best_params_path, 'w') as f:
        json.dump(best_record["params"], f, indent=2)
    
    # Re-run best params to get results_df
    best_params = PolicyParams(**best_record["params"])
    best_results_df, _ = backtest_all(episodes, best_params, objective_kwargs.get("cost_bps", 10.0))
    
    # Save best results
    results_path = out_dir / "results.csv"
    best_results_df.to_csv(results_path, index=False)
    
    return {
        "best_params": best_record["params"],
        "best_score": best_score,
        "best_metrics": best_record["metrics"],
        "leaderboard": [record for _, record in leaderboard]
    }


def baseline_always_skip(episodes: List[Episode]) -> Dict[str, Any]:
    """Baseline: always skip all IPOs."""
    params = PolicyParams(participate_threshold=1.0)  # Never participate
    results_df, equity_curve = backtest_all(episodes, params, cost_bps=10.0)
    score_value, metrics = score(results_df, equity_curve)
    
    return {
        "name": "always_skip",
        "params": {"participate_threshold": 1.0},
        "score": score_value,
        "metrics": metrics
    }


def baseline_always_participate(episodes: List[Episode], weight: float = 0.1) -> Dict[str, Any]:
    """Baseline: always participate with fixed weight."""
    params = PolicyParams(participate_threshold=0.0, raw_weight=weight)
    results_df, equity_curve = backtest_all(episodes, params, cost_bps=10.0)
    score_value, metrics = score(results_df, equity_curve)
    
    return {
        "name": "always_participate",
        "params": {"participate_threshold": 0.0, "raw_weight": weight},
        "score": score_value,
        "metrics": metrics
    }


def baseline_fixed_hold_k(episodes: List[Episode], hold_k: int = 5, weight: float = 0.1) -> Dict[str, Any]:
    """Baseline: fixed hold period."""
    params = PolicyParams(participate_threshold=0.0, hold_k=hold_k, raw_weight=weight)
    results_df, equity_curve = backtest_all(episodes, params, cost_bps=10.0)
    score_value, metrics = score(results_df, equity_curve)
    
    return {
        "name": f"fixed_hold_{hold_k}",
        "params": {"participate_threshold": 0.0, "hold_k": hold_k, "raw_weight": weight},
        "score": score_value,
        "metrics": metrics
    }
