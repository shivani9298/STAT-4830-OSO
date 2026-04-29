"""Sanity checks for smooth CVaR tail sensitivity."""

import torch

from src.losses import cvar_smooth


def test_cvar_smooth_emphasizes_left_tail():
    r = torch.tensor([-0.20, -0.10, 0.00, 0.10, 0.20], dtype=torch.float32)
    base = float(cvar_smooth(r, alpha=0.2, temperature=0.05).item())
    eps = 1e-3

    r_worst = r.clone()
    r_worst[0] -= eps
    cvar_worst = float(cvar_smooth(r_worst, alpha=0.2, temperature=0.05).item())

    r_best = r.clone()
    r_best[-1] -= eps
    cvar_best = float(cvar_smooth(r_best, alpha=0.2, temperature=0.05).item())

    delta_worst = abs(cvar_worst - base)
    delta_best = abs(cvar_best - base)

    # A left-tail metric should react more to worsening the worst return
    # than to an equal perturbation in the best return.
    assert delta_worst > delta_best
