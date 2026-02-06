# Self-Critique: Week 3 Deliverable

## OBSERVE: Initial Assessment

After reviewing our report and implementation:
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

1. **Update Model Architecture** (Week 5): Add a model architecture such as a GRU to predict future model weights/investment strategies

2. **Add Hyperparameter Optimization** (Week 5): Implement grid search or Bayesian optimization for λ values using a validation set (e.g., 2020-2022 train, 2023 validate, 2024+ test). Document sensitivity of results to parameter choices.

3. **Expand Validation** (Week 5): Add statistical tests, include traditional benchmarks, and create a proper train/validation/test split to detect overfitting.

---

## ACT: Resource Needs
We plan to use Scikit-optimize for Bayesian hyperparameter tuning and expand the dataset to include more assets, longer history, and macro features to reduce overfitting. Key knowledge gaps include implementing GRU sequence models correctly in PyTorch and selecting appropriate time-series statistical tests (forecast comparison and Sharpe significance). Major blockers are limited or slow access to WRDS data and long runtimes for hyperparameter search over multi-year daily data, which may require GPU use and search-space reduction. 

We plan on referencing this article: https://sharmasaravanan.medium.com/time-series-forecasting-using-gru-a-step-by-step-guide-b537dc8dcfba
https://wrds-www.wharton.upenn.edu/pages/classroom/accessing-data-via-the-wrds-api-and-excel/

We would also need a CRSP subscription as well to use the WRDS api.



