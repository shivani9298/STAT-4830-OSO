# Development Log

## Week 3: Initial Implementation

### 2025-02-05: Project Setup and Core Implementation

#### Goals
- Define optimization problem mathematically
- Implement Online Gradient Descent allocator
- Build IPO 180-day index with market-cap weighting
- Run walk-forward backtest

#### Decisions Made

**1. Choice of OGD over other methods**
- Considered: Mean-variance optimization, Black-Litterman, RL (PPO/SAC)
- Chose OGD because:
  - Online learning adapts to regime changes
  - Simple to implement with PyTorch autograd
  - No distributional assumptions required
  - Interpretable fitness function

**2. IPO holding period: 180 days**
- Research suggests IPO "honeymoon period" is 3-6 months
- 180 trading days ≈ 9 calendar months
- Captures initial price discovery while avoiding long-term underperformance

**3. Market-cap weighting vs equal-weight**
- Chose market-cap weighting to:
  - Reduce impact of illiquid small IPOs
  - Better reflect investable universe
  - Mirror institutional construction

**4. Fitness function components**
- Mean return: Primary objective
- Variance penalty (λ₁=20): Risk control
- Drawdown penalty (λ₂=8): Tail risk protection
- Turnover penalty (λ₃=0.15): Transaction cost proxy

#### Implementation Notes

**Data Pipeline:**
```
Yahoo Finance API → Price DataFrame → Returns → IPO Index + SPY → Combined Returns
```

Issues encountered:
- Timezone mismatch between yfinance and pandas (fixed with tz_localize)
- Some tickers delisted or renamed (handled with try/except)
- Shares outstanding only available as current value (known limitation)

**Optimization Loop:**
```
For t in [window, T]:
    1. Extract R[t-W:t]
    2. Compute fitness gradient via autograd
    3. Update weights: w = w - lr * grad
    4. Project to simplex
    5. Store w[t]
```

#### Results Summary

| Metric | OGD | SPY Only | IPO Only |
|--------|-----|----------|----------|
| Total Return | 193% | 86% | 1699% |
| Sharpe | 1.42 | 0.97 | 1.56 |
| Max Drawdown | -26% | -25% | -73% |

Key observation: OGD sacrifices upside for better drawdown control.

#### Open Questions
1. Is 180-day holding period optimal?
2. How sensitive are results to λ parameters?
3. Would adding a third asset (bonds/cash) improve risk-adjusted returns?

---

### Next Steps (Week 4)

1. [ ] Implement historical shares outstanding lookup
2. [ ] Add hyperparameter sensitivity analysis
3. [ ] Create train/validation/test split
4. [ ] Expand IPO universe
5. [ ] Add bootstrap confidence intervals

---

## Code Quality Checklist

- [x] Code runs end-to-end without errors
- [x] All functions have docstrings
- [x] Unit tests for core functions
- [x] Results are reproducible (seed set)
- [ ] Code profiled for performance
- [ ] Memory usage monitored
- [ ] Edge cases handled

---

## Resources Used

**Data:**
- Yahoo Finance (via yfinance library)
- IPO dates from NYSE/NASDAQ public calendars

**References:**
- Zinkevich (2003): Online Convex Programming
- Hazan (2016): Introduction to Online Convex Optimization
- Ritter & Welch (2002): A Review of IPO Activity, Pricing, and Allocations

**AI Tools:**
- Claude: Code generation, debugging, documentation
- See `docs/llm_exploration/week3_log.md` for conversation logs
