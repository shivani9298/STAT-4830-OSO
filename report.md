# Online Portfolio Optimization: S&P 500 vs IPO 180-Day Index

## Problem Statement

### What We're Optimizing
We optimize **daily portfolio weights** between two asset classes:
1. **S&P 500** (SPY ETF) - Established large-cap US equities
2. **IPO 180-Day Index** - A custom market-cap weighted index of recent IPOs where each stock exits after 180 trading days

The goal is to maximize a **risk-adjusted fitness score** that balances returns against volatility, drawdown, and transaction costs.

### Why This Problem Matters
IPOs represent a unique asset class with distinct return characteristics:
- **High volatility**: IPO stocks exhibit 3-4x the volatility of the S&P 500
- **Mean reversion**: Many IPOs experience significant price changes in their first 6 months
- **Information asymmetry**: Early price discovery creates potential alpha opportunities

A systematic approach to allocating between stable market exposure (SPY) and high-risk/high-reward IPO exposure could capture upside while managing downside risk.

### Success Metrics
| Metric | Target | Rationale |
|--------|--------|-----------|
| Sharpe Ratio | > 1.0 | Risk-adjusted return above market |
| Max Drawdown | < 30% | Capital preservation |
| Annualized Return | > 15% | Beat SPY baseline |
| Calmar Ratio | > 0.8 | Return per unit drawdown |

### Constraints
- **Long-only**: Weights ∈ [0, 1], no shorting
- **Fully invested**: Weights sum to 1 (simplex constraint)
- **Daily rebalancing**: Practical for retail/institutional investors
- **Transaction costs**: Implicit via turnover penalty

### Data Requirements
- **Price data**: Daily adjusted close prices from Yahoo Finance (yfinance API)
- **Shares outstanding**: For market-cap weighting (yfinance API)
- **IPO dates**: Curated list of ~80 major IPOs from 2020-2024
- **Time horizon**: 2020-01-01 to 2025-01-14 (~1,200 trading days)

### What Could Go Wrong
1. **Survivorship bias**: Only including IPOs that still trade (excluding delistings)
2. **Look-ahead bias**: Using current shares outstanding for historical market caps
3. **Data quality**: Yahoo Finance data gaps or errors
4. **Overfitting**: Hyperparameters tuned to historical data may not generalize

---

## Technical Approach

### Mathematical Formulation

**Objective Function (Fitness Score):**

<img width="356" height="65" alt="image" src="https://github.com/user-attachments/assets/2d33d72f-cf65-4cf6-81b0-6fa08feef8ce" />


Where:
- $w \in \mathbb{R}^2$ = portfolio weights [SPY, IPO_INDEX]
- $\mu_p = \frac{1}{T}\sum_{t=1}^{T} r_t^\top w$ = mean portfolio return over window
- $\sigma_p^2 = \text{Var}(r^\top w)$ = portfolio variance
- <img width="461" height="68" alt="image" src="https://github.com/user-attachments/assets/900185b2-6846-4284-be5b-1f0f29ac3030" />
- $\lambda_1, \lambda_2, \lambda_3$ = penalty coefficients (hyperparameters)

**Constraints:**
$$w_i \geq 0, \quad \sum_i w_i = 1 \quad \text{(probability simplex)}$$

### Algorithm: Online Gradient Descent (OGD)

We use **projected gradient descent** with adaptive learning rates:

```
For each day t:
    1. Extract trailing window of returns R[t-W:t]
    2. Compute gradient of fitness w.r.t. weights
    3. Update: w_new = w - lr * grad
    4. Project onto simplex: w = proj_simplex(w_new)
    5. Apply weights to next day's returns
```

**Why OGD?**
- Online learning adapts to regime changes (bull/bear markets)
- No assumptions about return distributions
- Differentiable objective enables PyTorch autograd
- Simplex projection ensures valid portfolio weights

### PyTorch Implementation Strategy
- **Autograd**: Compute gradients automatically via `loss.backward()`
- **Tensor operations**: Vectorized return calculations
- **Custom projection**: Euclidean projection onto simplex (O(n log n))

### Validation Methods
1. **Walk-forward testing**: Train on [t-W, t], test on t+1 (no look-ahead)
2. **Benchmark comparison**: OGD vs Equal Weight vs SPY-only
3. **Rolling metrics**: 126-day rolling Sharpe, volatility, drawdown
4. **Sensitivity analysis**: Vary hyperparameters (λ₁, λ₂, λ₃, W, lr)

---

## Initial Results

### Evidence Implementation Works

**IPO Index Construction:**
- Successfully built market-cap weighted index with ~80 IPO tickers
- Average 8.5 stocks in index at any time
- Average total market cap: $150.8B
- 1,208 valid trading days with both assets

**OGD Optimization:**
- Walk-forward optimization completed over full period
- Weights evolve smoothly (no erratic behavior)
- Simplex constraint satisfied at all times

### Performance Metrics (2020-2025)

| Strategy | Total Return | Ann. Return | Ann. Vol | Sharpe | Max DD | Calmar |
|----------|-------------|-------------|----------|--------|--------|--------|
| **OGD Portfolio** | **193.5%** | **28.5%** | 20.1% | **1.42** | **-26.2%** | 1.09 |
| Equal Weight | 577.7% | 56.2% | 35.0% | 1.60 | -51.3% | 1.09 |
| S&P 500 Only | 86.1% | 15.6% | 16.1% | 0.97 | -24.5% | 0.64 |
| IPO Index Only | 1699.3% | 96.0% | 61.4% | 1.56 | -73.1% | 1.31 |

**Key Finding**: OGD achieves **Sharpe 1.42** with only **-26% max drawdown**, trading off some upside for significantly better risk control compared to IPO-only (-73% drawdown).

### Current Limitations
1. **Shares outstanding approximation**: Using current shares for all historical dates
2. **Limited IPO universe**: ~80 tickers vs hundreds of actual IPOs
3. **No transaction costs**: Turnover penalty is proxy, not actual costs
4. **Single hyperparameter set**: Not yet optimized across regimes

### Resource Usage
- **Runtime**: ~3 minutes for full backtest (fetching data + optimization)
- **Memory**: <1GB (pandas DataFrames fit comfortably)
- **API calls**: ~85 Yahoo Finance requests (one per ticker)

---

## Next Steps

### Immediate Improvements
1. **Expand IPO universe**: Scrape comprehensive IPO data from SEC EDGAR or Nasdaq
2. **Historical shares outstanding**: Use quarterly filings for accurate market caps
3. **Add transaction costs**: Model bid-ask spreads and market impact
4. **Add neural network**: Add a model architecture such as a GRU to predict future model weights/investment strategies 

### Technical Challenges
1. **Survivorship bias correction**: Include delisted IPOs (requires premium data)
2. **Hyperparameter optimization**: Grid search or Bayesian optimization for λ values
3. **Regime detection**: Adapt parameters for bull/bear/volatile markets

### Questions Needing Help
1. Is there a free data source for historical IPO filings with shares outstanding?
2. How to properly handle delisted stocks in backtesting?
3. Should we add a third asset class (bonds/cash) for risk-off periods?

### Alternative Approaches to Try
1. **Reinforcement Learning**: Replace OGD with PPO/SAC for weight allocation
2. **Factor-based IPO selection**: Use momentum/value/quality factors within IPO universe
3. **Dynamic holding period**: Optimize exit timing instead of fixed 180 days

### What We've Learned
1. IPOs are extremely volatile (3-4x SPY) but offer high returns in aggregate
2. Market-cap weighting reduces impact of small, illiquid names
3. OGD effectively shifts to defensive (SPY) during IPO drawdowns
4. The 180-day holding period captures most of the "IPO pop" while avoiding long-term underperformance
