# Week 4: AI / LLM Exploration Log

Conversations with AI tools (Cursor, etc.) used to explore:

- Objective formulation (E[R] - λ·CVaR - κ·Cost - μ·MDD)
- REINFORCE and policy network design
- Validation and test strategy
- Repository structure and report outline

---

**2026-02-04: Objective and fitness** — Explored whether the reported fitness score (E[R] − λ·CVaR − κ·Cost − μ·MDD) was the same as the REINFORCE loss. Clarified that the code optimizes per-episode net return (reward) while the full score is logged for monitoring; added optional `reward_type="score"` in train_policy so the batch fitness can be used as the reward.

**2026-02-04: build_ipo_index and gradient descent** — Confirmed that the default market-cap index does not use gradient descent; added `OnlinePortfolioOptimizer` (gradient ascent + simplex projection) and `--use-optimizer` so the IPO index builder can optionally use gradient-based weight updates aligned with the course.
