# Week 4: AI / LLM Exploration Log

Conversations with AI tools (Cursor, etc.) used to explore:

- Objective formulation (E[R] - λ·CVaR - κ·Cost - μ·MDD)
- REINFORCE and policy network design
- Validation and test strategy
- Repository structure and report outline

---

**2026-02-04: Objective and fitness** — Explored whether the reported fitness score (E[R] − λ·CVaR − κ·Cost − μ·MDD) was the same as the REINFORCE loss. Clarified that the code optimizes per-episode net return (reward) while the full score is logged for monitoring; added optional `reward_type="score"` in train_policy so the batch fitness can be used as the reward.

**2026-02-04: build_ipo_index and gradient descent** — Confirmed that the default market-cap index does not use gradient descent; added `OnlinePortfolioOptimizer` (gradient ascent + simplex projection) and `--use-optimizer` so the IPO index builder can optionally use gradient-based weight updates aligned with the course.

**2026-02-10: Walk-forward validation design** — Asked: "What's the right way to do train/test splits for time-series trading data?" AI clarified that random splits leak future information into training; correct approach is chronological split (train on earlier dates, test on later dates). For IPO data with only a few years, a single cutoff (e.g. 80/20 by date) is reasonable; multi-fold walk-forward (train 2021→test 2022, train 2021-22→test 2023, …) requires more data. Implemented single time-based split in notebook's walk-forward section.

**2026-02-10: Sharpe term and objective consistency** — Asked: "Should I add Sharpe ratio to the objective, and how does it interact with CVaR?" AI explained: Sharpe and CVaR are partially redundant (both penalize variance) but CVaR is more sensitive to tail risk. Adding β·Sharpe on top of −λ·CVaR double-penalizes variance; better to pick one or tune β to be small. Decision: keep CVaR as the primary risk term; note Sharpe as a future extension. This resolves the report/code inconsistency where the headline formula included β·Sharpe but `objective.score()` did not.

**2026-02-12: Hyperparameter sensitivity (λ, κ, μ)** — Asked: "How sensitive is the policy to the penalty weights?" AI suggested: run a small grid (e.g. λ ∈ {0.5, 1.0, 2.0}, μ ∈ {0.5, 1.0}) on the validation set and pick the combo with the best OOS score. Noted that with only ~16 OOS episodes the grid will be noisy; a fixed "risk budget" (λ=1, κ=1, μ=1) is defensible for an MVP. Current code already supports overriding via `--lam`, `--kappa`, `--mu` flags in `run_pytorch.py`.

**2026-02-14: REINFORCE variance and batch size** — Asked: "Why are REINFORCE training scores so noisy epoch-to-epoch?" AI explained three sources: (1) stochastic action sampling per batch, (2) small batch sizes → high-variance gradient estimates, (3) MDD is non-smooth over small episode sets. Mitigations implemented: mean-reward baseline in `train_policy.py`, gradient clipping (norm 1.0), and entropy bonus (coef=0.01). Larger batch sizes help most; with 64 episodes and batch_size=64 (full-batch), variance dropped noticeably in test runs.
