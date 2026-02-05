# Self-Critique: Week 3 Deliverable

## OBSERVE: Initial Assessment

After reviewing my report and implementation:
- The code runs end-to-end and produces reasonable results
- Mathematical formulation is present but could be more rigorous
- Data sources are real (Yahoo Finance) but have limitations
- Performance metrics show promising results (Sharpe 1.42)

---

## ORIENT: Analysis

### Strengths (3 key points)

1. **Working Implementation**: The OGD optimizer runs successfully with walk-forward backtesting, producing valid portfolio weights that sum to 1 and are non-negative.

2. **Real Data Pipeline**: Uses actual market data from Yahoo Finance rather than synthetic data, making results more credible and reproducible.

3. **Clear Risk Management**: The fitness function explicitly penalizes variance, drawdown, and turnover - addressing multiple dimensions of portfolio risk.

### Areas for Improvement (3 key points)

1. **Data Quality Issues**: Using current shares outstanding for all historical dates introduces look-ahead bias. The IPO universe is limited (~40 stocks) and suffers from survivorship bias (excludes delisted IPOs).

2. **Hyperparameter Justification**: The penalty coefficients (λ₁=20, λ₂=8, λ₃=0.15) were chosen heuristically without systematic optimization or sensitivity analysis.

3. **Validation Gaps**: No out-of-sample test period, no statistical significance tests for performance differences, no comparison to standard benchmarks (e.g., 60/40 portfolio).

### Critical Risks/Assumptions

The biggest risk is **overfitting to the 2020-2021 IPO boom**. This period had unprecedented IPO activity and returns (RIVN, ABNB, COIN) that may not repeat. The model's shift to 100% SPY in 2024-2025 suggests it learned regime-specific patterns rather than generalizable allocation rules. Additionally, the assumption that market-cap weighting with current shares is valid for historical periods could significantly bias market cap calculations.

---

## DECIDE: Concrete Next Actions

1. **Fix Data Pipeline** (Week 4): Implement historical shares outstanding lookup using quarterly SEC filings (10-Q) or find a data provider with point-in-time market caps. This directly addresses the look-ahead bias.

2. **Add Hyperparameter Optimization** (Week 4-5): Implement grid search or Bayesian optimization for λ values using a validation set (e.g., 2020-2022 train, 2023 validate, 2024+ test). Document sensitivity of results to parameter choices.

3. **Expand Validation** (Week 5): Add statistical tests (bootstrap confidence intervals for Sharpe ratio), include traditional benchmarks (60/40, risk parity), and create a proper train/validation/test split to detect overfitting.

---

## ACT: Resource Needs

**Tools needed:**
- SEC EDGAR API access for quarterly filings (free)
- Optuna or scikit-optimize for hyperparameter tuning
- scipy.stats for bootstrap confidence intervals

**Knowledge gaps:**
- Need to learn how to parse SEC filings for shares outstanding
- Should review literature on online convex optimization for better theoretical grounding

**Potential blockers:**
- SEC filing parsing may be complex; may need to use a paid data provider (Polygon, EOD Historical) if free approach fails
- Running hyperparameter search over 5 years of daily data may be slow; may need to optimize code or use cloud compute

---

## Summary

The current implementation is a solid foundation but needs work on data quality and validation rigor before the results can be trusted. The immediate priority is fixing the shares outstanding data to eliminate look-ahead bias, followed by proper hyperparameter selection and out-of-sample testing.
