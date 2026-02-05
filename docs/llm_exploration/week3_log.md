# Week 3 LLM Exploration Log

## Session: 2025-02-05

### Topic: Building IPO Index and Portfolio Optimization

#### Questions Asked

**Q1: How to create an IPO index with 180-day holding period?**

Approach discussed:
- Track IPO dates for each ticker
- Count trading days since IPO
- Include in index only if days_since_ipo < 180
- Weight by market cap (price × shares outstanding)

Key insight: Need to track trading days, not calendar days, to account for weekends/holidays.

**Q2: Market-cap weighting vs equal-weighting?**

Trade-offs:
- Market-cap: Larger companies have more influence, more stable, lower turnover
- Equal-weight: Higher exposure to small caps, potentially higher returns, more rebalancing

Decision: Use market-cap weighting for institutional relevance.

**Q3: What fitness function for portfolio optimization?**

Discussed options:
1. Sharpe ratio (mean/std)
2. Mean-variance utility
3. Custom fitness with multiple penalties

Chose custom fitness:
```
F = mean - λ₁*var + λ₂*mdd - λ₃*turnover
```

Rationale: Can tune trade-offs between objectives.

**Q4: How to handle Yahoo Finance data issues?**

Issues identified:
- Timezone-aware timestamps
- Missing data for delisted stocks
- Shares outstanding only current value

Solutions implemented:
- `tz_localize(None)` for timezone
- Try/except for missing tickers
- Document limitation for shares data

#### Code Generated

1. `project_to_simplex()` - Euclidean projection onto simplex
2. `max_drawdown_from_returns()` - Differentiable max drawdown
3. `OnlineOGDAllocator` class - Main optimization logic
4. `build_ipo_index_mcap()` - IPO index construction

#### Insights Gained

1. **Data quality is critical**: The biggest limitation is using current shares outstanding for historical market caps. This is a form of look-ahead bias.

2. **Hyperparameters matter**: The penalty coefficients significantly affect behavior. Need systematic tuning.

3. **Regime changes are real**: The model shifts to 100% SPY in 2024-2025, suggesting it learned regime-specific patterns.

#### Follow-up Questions for Next Session

1. How to get historical shares outstanding data?
2. Best practices for hyperparameter optimization in time series?
3. How to test for statistical significance of Sharpe ratio differences?

---

## Session Notes Template

For future sessions, document:
- Questions asked
- Approaches considered
- Decisions made (with rationale)
- Code generated
- Insights gained
- Follow-up questions
