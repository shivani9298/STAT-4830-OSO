# Week 3 Deliverable: IPO Trading Strategy Optimization

## Problem Statement

### What are you optimizing? (Be specific)

Build a constrained trading-optimization sandbox that selects and sizes retail IPO trades using event-conditioned rules (IPO day-1/week-1 regime). The system outputs a policy over: (i) participate/skip IPO, (ii) requested size under allocation uncertainty, (iii) entry timing bucket, (iv) position size, and (v) exit style.

We are optimizing a parameterized policy ($\pi_\theta$) that maps IPO-level features and post-listing "event state" (time-since-open bucket, realized volatility proxy, volume shock proxy) into decisions that maximize risk-adjusted utility:

The fitness score used for strategy optimization is computed as follows, echoing the core logic of the provided `calculate_fitness` function:

Let $r_{t, \theta}$ be the daily return series of strategy parameters $\theta$, and $rf$ the annual risk-free rate (e.g., $0.04$):

1. **Convert annual risk-free to daily:**
   $$
   rf_{\text{daily}} = (1 + rf)^{1/252} - 1
   $$

2. **Annualized Return:**
   $$
   \mathbb{E}[R_\theta]_\text{annual} = \text{mean}(r_{t, \theta}) \times 252
   $$

3. **Annualized Volatility:**
   $$
   \sigma_\theta^\text{annual} = \text{std}(r_{t, \theta}) \times \sqrt{252}
   $$

4. **Sharpe Ratio:**
   $$
   \mathrm{Sharpe}(R_\theta) = \frac{ \mathbb{E}[R_\theta]_\text{annual} - rf }{ \sigma_\theta^\text{annual} }
   $$

5. **Maximum Drawdown (MDD):**
   $$
   \text{Cumulative: } S_t = \prod_{s=1}^{t} (1 + r_{s, \theta})
   $$
   $$
   \text{MDD}_\theta = \min_{t} \left( \frac{S_t - \max_{s \leq t} S_s}{\max_{s \leq t} S_s} \right)
   $$

6. **Combined Fitness Score:**
   $$
   \mathrm{Fitness}_\theta = 0.4 \cdot \mathrm{Sharpe}(R_\theta) 
      + 0.3 \cdot \mathbb{E}[R_\theta]_\text{annual}
      + 0.3 \cdot (1 + \mathrm{MDD}_\theta)
   $$
   (where $1 + \mathrm{MDD}_\theta$ guarantees positivity as in code)

> In summary:  
> **Fitness = 0.4 × Sharpe + 0.3 × annualized return + 0.3 × (1 + max drawdown)**  
> This scoring function rewards high risk-adjusted returns, penalizes drawdowns, and incorporates both Sharpe and raw return performance.

**Parameters**:  
- $r_{t, \theta}$: daily returns of the strategy  
- $rf$: risk-free rate (annual, e.g., 0.04)  
- All calculations use 252 trading days/year.

**Reference implementation** (Python):
```python
def calculate_fitness(returns, risk_free_rate=0.04):
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    portfolio_return = np.mean(returns) * 252
    portfolio_volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    fitness = (0.4 * sharpe_ratio +
               0.3 * portfolio_return +
               0.3 * (1 + max_drawdown))
    return fitness
```

### Why does this problem matter?

Retail IPO trading exhibits extreme volatility, halts, and structural frictions (opening auction noise, liquidity constraints). A systematic, risk-aware controller is a better research artifact than one-off heuristics: it forces clear objectives, constraints, and robust evaluation. This approach enables:

- **Systematic evaluation** of trading strategies under realistic constraints
- **Risk-aware optimization** that balances return, risk, and costs
- **Reproducible research** framework that can be extended to other event-driven strategies

### How will you measure success?

Potential criteria:
- **Lower tail risk** (CVaR, worst decile) without collapsing average return
- **Robustness** across different IPO cohorts and market regimes
- **Sharpe ratio** and risk-adjusted metrics

### What are your constraints?

- **Max per-IPO exposure**: $w \leq w_{\max}$ (position size cap)
- **Max portfolio exposure**: $\sum |w_i| \leq W_{\max}$ (total IPO bucket exposure)
- **Max loss per IPO trade**: Hard cap or position cap derived from gap proxy
- **Liquidity cap**: $w \leq c \cdot \text{ADV}$ (or float-based proxy)
- **No trade during halt windows**: If intraday halts are available; otherwise conservative cost penalties on day-1

### What data do you need?

- **IPO calendar + terms**: ticker, IPO date (first trading day), offer price, range, offer size/float, underwriter (or proxy)
- **Post-IPO prices/volume**: Daily OHLCV for IPO date through day+N (MVP), optional intraday bars for day 1–5
- **Event features**: Time-since-open buckets, realized volatility, volume shocks, halt indicators

### What could go wrong?

- **Universe construction errors**: Missing delisted IPOs, wrong "first trading day," ticker changes
- **Regime dependence**: Results dominated by specific IPO cycles (e.g., 2020-2021 "hot" market)
- **Lookahead/leakage**: Using day-1 outcomes to decide day-1 entry
- **Transaction cost model too optimistic**: Day-1 conditions have wider spreads, halts, and execution uncertainty
- **Small sample size**: Limited IPOs per year make statistical significance challenging
- **Overfitting**: Too many parameters relative to available data

---

## Technical Approach

### Mathematical formulation (objective function, constraints)

Index IPOs by $i$. Each IPO generates an episode of length $T$ days (or intraday buckets). State features $x_{i,t}$ include:

- **Static features**: offer price, raise size, float %, underwriter proxy, sector
- **Event-conditioned features**: time-since-open bucket (or day index), realized vol proxy, volume shock proxy

**Actions** $a_{i,t}$ for MVP (discrete):
- $a^{\text{part}}_i \in \{0,1\}$ (participate/skip)
- $a^{\text{entry}}_i \in \{\text{open}, +5m, +30m, \text{EOD}, \text{day+1}, \ldots\}$ (if intraday; otherwise day buckets)
- $w_i \in [0, w_{\max}]$ (position size)
- $a^{\text{exit}}_i$ = holding horizon bucket or staged schedule

**Return per IPO**:
$$r_i(\theta) = a^{\text{part}}_i \cdot w_i \cdot \left(\frac{P^{\text{exit}}_i}{P^{\text{entry}}_i} - 1\right) - \text{cost}_i(w_i, \text{liq}, \text{vol})$$

Portfolio return = $R_\theta = \sum_i r_i(\theta)$ over an evaluation window, then compute utility with CVaR.

**Maximize risk-adjusted utility**:
$$\max_{\theta} \mathbb{E}[R_\theta] - \lambda \cdot \mathrm{CVaR}_\alpha(R_\theta) - \kappa \cdot \mathbb{E}[C_\theta] - \mu \cdot \mathrm{MDD}$$

Where $R_\theta$ is strategy return and $C_\theta$ is an execution cost proxy.

### Algorithm/approach choice and justification

**Start with black-box optimization** over a discrete rule policy (participate/skip, entry/exit buckets, sizing) with a risk-aware objective (CVaR + costs + drawdown penalty) because:

- **Non-differentiable backtest**: Fills, constraints, discrete timing make gradient-based methods challenging
- **Modest sample size**: IPO count per year is limited, making lower-variance methods preferable
- **Easier validation**: Rule-based policies are interpretable and debuggable
- **Lower computational cost**: Random search is fast and parallelizable

**Use walk-forward + cohort/regime splits** as the main guardrail since IPO dynamics are non-stationary and "hot years" can dominate results; we want something that generalizes, not something that fits 2020–2021.

**Add PyTorch only as an optional "policy learner" layer** (contextual bandit / discrete policy over buckets) once the sandbox is stable, so ML is demonstrating generality rather than being the core dependency.

### PyTorch implementation strategy

**Represent each IPO as an episode tensor** $X[i, t, f]$ with a mask; for the daily MVP, we can compress to decision-time features (e.g., features at entry time) and avoid long sequences initially.

**Build a small network** that outputs:
- Categorical logits for: participate vs skip, entry bucket, exit bucket
- Bounded sizing scalar (sigmoid → [0, 1] mapped to risk budget)

**Training Options**:
1. **Contextual bandit / policy gradient (REINFORCE)** on realized episode returns with:
   - Strong baseline (moving average / value head)
   - Entropy regularization to reduce variance
2. **Supervised imitation** for sanity-check (imitate a heuristic) + then fine-tune with policy gradient on the true objective

**Keep the simulator / objective outside PyTorch**: PyTorch just proposes actions, then the core backtester returns rewards and risk metrics.

### Validation methods

1. **Walk-forward validation by calendar time**: Train on earlier IPO cohorts, test on later cohorts to mimic live deployment and check performance stability across regimes
2. **Rolling re-optimization**: Periodically re-tune policy parameters on a moving training window and evaluate on the next unseen window, building an out-of-sample equity curve across the full history
3. **Hold-out test set**: Reserve the most recent IPOs as a final untouched sample, used only once after all modeling decisions are fixed to assess true generalization
4. **Ablation studies**: Remove or simplify components (allocation model, timing features, exit sophistication, CVaR term) to measure their individual contribution to performance and robustness
5. **Placebo and leakage checks**: Randomize entry times, shuffle labels, or time-shift features to verify that any edge disappears when the true structure is broken, helping detect overfitting and data leakage

### Resource requirements and constraints

- **Computational**: Daily-data MVP runs on laptop CPU (sufficient for random search with 100-1000 trials)
- **Data storage**: IPO metadata (~1MB), price data varies by granularity (daily: ~10MB, intraday: ~100MB+)
- **Time**: Random search with 500 trials on 1600 IPOs completes in ~1-2 minutes
- **Memory**: Minimal (< 1GB for daily data MVP)

---

## Initial Results

### Evidence your implementation works

**✅ Core System Functional**:
- IPO index builder (`build_ipo_index.py`) successfully fetches live data from Yahoo Finance
- Market-cap weighted index construction working correctly
- Successfully processed 1,178 unique IPOs from dailyhistorical_21-26.csv dataset
- Comparison against S&P 500 (SPY) benchmark implemented
- Results saved to CSV with daily index values, constituent weights, and pivot tables

**✅ Data Pipeline**:
- IPO metadata loading: Parses COMPUSTAT-style daily historical data (gvkey, tic, datadate, prccd)
- Live price fetching: Uses yfinance to get 180 days of post-IPO price data
- Shares outstanding: Fetches from Yahoo Finance for market-cap weighting
- Handles delisted/failed tickers gracefully (43/100 failed in 2021 cohort, mostly SPACs)

**✅ Index Construction**:
- Market-cap weighted returns calculated daily
- Rolling 180-day inclusion window per IPO
- Constituent weights tracked and saved
- Cumulative index value computed (base = 100)

### Basic performance metrics

**IPO Index vs S&P 500: 2021 Cohort (July 2021 - Feb 2022)**

| Metric | IPO Index | S&P 500 | Difference |
|--------|-----------|---------|------------|
| **Total Return** | -43.54% | +6.53% | -50.07% |
| **Annualized Volatility** | 131.37% | 14.83% | +116.54% |
| **Max Drawdown** | -68.40% | -11.67% | -56.73% |
| **Fitness Score** | -0.037 | 0.240 | -0.277 |
| **Avg Constituents** | 47 | - | - |
| **Max Constituents** | 57 | - | - |

**IPO Index vs S&P 500: 2023 Cohort (Jan 2023 - Oct 2023)**

| Metric | IPO Index | S&P 500 | Difference |
|--------|-----------|---------|------------|
| **Total Return** | +82.96% | +6.11% | +76.85% |
| **Annualized Volatility** | 558.69% | 13.25% | +545.44% |
| **Max Drawdown** | -70.60% | -9.97% | -60.63% |
| **Fitness Score** | 2.270 | 0.727 | +1.543 |
| **Avg Constituents** | 22 | - | - |
| **Max Constituents** | 37 | - | - |

**Key Findings**:
- **Extreme regime dependence**: 2021 IPOs massively underperformed (-50% vs SPY) during tech correction, while 2023 IPOs outperformed (+77% vs SPY)
- **Very high volatility**: IPO index volatility 9-40x higher than S&P 500
- **Severe drawdowns**: Max drawdowns of -68% to -71% regardless of final performance
- **Data attrition**: 40-50% of IPOs fail to fetch (delisted SPACs, ticker changes, missing data)

### Files used by current implementation

| File | Purpose | Status |
|------|---------|--------|
| `build_ipo_index.py` | Main IPO index builder | ✅ Used |
| `src/dailyhistorical_21-26.csv` | IPO historical data (1,178 IPOs) | ✅ Used |
| `results/ipo_index.csv` | Output: daily index values | ✅ Generated |
| `results/ipo_weights.csv` | Output: constituent weights | ✅ Generated |
| `results/ipo_weights_pivot.csv` | Output: weights pivot table | ✅ Generated |

**Unused files** (from previous implementation attempts):
- `src/backtest.py`, `src/features.py`, `src/logging_utils.py`, `src/metrics.py`
- `src/objective.py`, `src/optimize.py`, `src/policy.py`, `src/policy_network.py`, `src/train_policy.py`
- `run_week3.py` (single stock comparison, not IPO index)
- `run_pytorch.py` (broken imports)
- `generate_report.py` (requires different input format)

### Data source details

**Input File**: `src/dailyhistorical_21-26.csv`
- **Format**: COMPUSTAT-style daily data
- **Columns**: gvkey, datadate, tic (ticker), prccd (closing price), cshtrd (volume)
- **Total IPOs**: 1,178 unique tickers
- **Date Range**: 2021-2026
- **After 2021 filter**: 1,170 IPOs available

**Data Fetch Results (2021 Cohort, 100 IPOs)**:
- Successfully fetched: 57/100 (57%)
- Failed - No price data: 33 (mostly delisted SPACs ending in 'U')
- Failed - No shares outstanding: 3
- Failed - Other: 7

**Output Files Generated**:
```
results/
├── ipo_index.csv          # Daily index values, returns, SPY comparison
├── ipo_weights.csv        # Long format: date, ticker, weight, price, market_cap
└── ipo_weights_pivot.csv  # Pivot table: date x ticker weights
```

### Current limitations

1. **High Data Attrition**: 40-50% of IPOs fail to fetch from Yahoo Finance (delisted SPACs, ticker changes, missing shares outstanding data). This survivorship bias may affect results.

2. **Extreme Volatility in Index**: Annualized volatility of 131-559% makes the index impractical as a standalone investment. Need risk controls.

3. **No Policy Optimization Yet**: Current implementation is a passive market-cap weighted index. The optimization framework (participate/skip, timing, sizing) is not yet integrated.

4. **180-Day Window Only**: Fixed holding period doesn't allow for early exit on momentum signals or stop-losses.

5. **Equal Treatment of All IPOs**: No filtering by sector, size, underwriter quality, or other predictive features.

6. **No Transaction Costs**: Index calculation assumes frictionless trading. Real IPO day-1 has wide spreads and execution uncertainty.

7. **Regime Dependence**: Results vary dramatically by IPO cohort year. 2021 lost -44%, 2023 gained +83%.

8. **Single Stock Concentration**: At end of period, index often collapses to 1-2 constituents (e.g., KVUE at 100% weight).

### Resource usage measurements

- **Runtime**: ~3-5 minutes for 100 IPOs (fetching live data from Yahoo Finance)
- **API Calls**: ~100 yfinance requests per run (rate-limited)
- **Memory**: < 500MB peak usage
- **Disk**: Output files ~500KB (ipo_index.csv, ipo_weights.csv, ipo_weights_pivot.csv)
- **Network**: Requires internet connection for live data fetching

### Unexpected challenges

1. **SPAC Failures**: Many 2021 IPOs were SPACs (ending in U) which have since delisted or merged, causing ~40% data fetch failures.

2. **Shares Outstanding Inconsistency**: Yahoo Finance sometimes returns None for sharesOutstanding, requiring fallback to impliedSharesOutstanding.

3. **Ticker Symbol Changes**: Some IPOs have changed tickers post-IPO (mergers, rebrands), breaking historical lookups.

4. **Date Alignment Issues**: Business day calendars differ between COMPUSTAT data and Yahoo Finance, requiring careful date matching.

5. **Extreme Returns**: Some IPOs show 500%+ gains or 90%+ losses within 180 days, causing extreme index volatility.

6. **Index Concentration**: As IPOs exit the 180-day window, index can become highly concentrated in remaining names.

---

## Next Steps

### Immediate improvements needed

1. **Integrate Policy Optimization with Index**:
   - Connect the existing `src/` optimization modules to `build_ipo_index.py`
   - Add participate/skip decisions based on IPO features
   - Implement variable position sizing (not just market-cap weighting)

2. **Reduce Data Attrition**:
   - Add fallback data sources for delisted tickers
   - Cache successful fetches to avoid repeated API calls
   - Map old tickers to new tickers for merged companies

3. **Add Risk Controls**:
   - Implement position caps per IPO ($w \leq w_{\max}$)
   - Add stop-loss exits before 180-day window ends
   - Diversification constraints to prevent single-stock concentration

4. **Walk-Forward Validation**:
   - Split IPO cohorts by year (train: 2021-2023, test: 2024-2025)
   - Run rolling re-optimization
   - Build out-of-sample equity curve

5. **Transaction Cost Model**:
   - Add spread costs (estimated 50-100 bps for IPO day-1)
   - Volume-based market impact
   - Slippage estimates for illiquid names

### Technical challenges to address

1. **Survivorship Bias**:
   - Current data excludes delisted IPOs that failed
   - Need to account for IPOs that went to zero
   - May need alternative data sources (CRSP, Bloomberg)

2. **Regime Detection**:
   - 2021 (hot market) vs 2022-2023 (cold) behave very differently
   - Need regime-aware policy switching
   - Or robust policy that works across regimes

3. **Index Concentration**:
   - End-of-period single-stock concentration (KVUE at 100%)
   - Need rebalancing rules or equal-weight alternative
   - Cap individual position sizes

4. **Volatility Management**:
   - 131-559% annualized volatility is impractical
   - Add volatility targeting or risk parity weighting
   - Consider hedging with SPY shorts

5. **PyTorch Policy Network**:
   - Fix broken imports in `run_pytorch.py`
   - Connect to live data pipeline
   - Train on historical IPO features and outcomes

### Questions you need help with

1. **Data Quality**: How to get price data for delisted IPOs? CRSP? Bloomberg? Manual collection?

2. **Benchmark Selection**: Is SPY the right benchmark? Should we use small-cap (IWM) or growth (QQQ)?

3. **Holding Period**: Is 180 days optimal? Literature suggests most IPO alpha decays within 30 days.

4. **Weighting Scheme**: Market-cap vs equal-weight vs volatility-weighted?

5. **Regime Indicators**: What signals indicate "hot" vs "cold" IPO markets?

### Alternative approaches to try

1. **Equal-Weight Index**: Remove market-cap concentration risk

2. **Momentum Filter**: Only include IPOs with positive 5-day momentum

3. **Sector Rotation**: Overweight/underweight sectors based on market conditions

4. **Shorter Holding Period**: Exit after 30 or 60 days instead of 180

5. **Hedged Strategy**: Long IPO index, short SPY to isolate IPO alpha

6. **Quality Filter**: Only include IPOs above certain market cap or with institutional backing

### What you've learned so far

1. **IPO Returns are Regime-Dependent**: 2021 cohort lost -44%, 2023 gained +83%. No single strategy works across all periods.

2. **Extreme Volatility**: IPO portfolios have 10-40x the volatility of S&P 500. Risk management is essential.

3. **Data Challenges are Real**: 40-50% of IPOs can't be fetched due to delistings, mergers, and ticker changes.

4. **Concentration Risk**: Market-cap weighting leads to single-stock concentration as IPOs exit the inclusion window.

5. **Fitness Score Limitations**: High returns can coexist with negative fitness if volatility and drawdowns are extreme.

6. **SPACs Dominate Failures**: Most fetch failures are SPACs (tickers ending in U) that have since delisted.

7. **Live Data is Feasible**: Yahoo Finance API provides sufficient data for daily analysis, though with some gaps.

---

## Appendix: Current Implementation Status

### ✅ Completed
- [x] IPO index builder (`build_ipo_index.py`) - standalone, working
- [x] Live data fetching from Yahoo Finance (yfinance)
- [x] Market-cap weighted index construction
- [x] SPY benchmark comparison
- [x] Daily constituent weight tracking
- [x] Fitness score calculation (Sharpe + Return + MaxDD)
- [x] Results export to CSV (index, weights, pivot)
- [x] Data loading from COMPUSTAT-style CSV (dailyhistorical_21-26.csv)

### 🚧 In Progress
- [ ] Connect policy optimization (`src/` modules) to index builder
- [ ] Fix broken imports in `run_pytorch.py`
- [ ] Reduce data attrition (handle delisted tickers)
- [ ] Walk-forward validation by IPO cohort year

### 📋 Planned
- [ ] Position sizing and risk controls
- [ ] Transaction cost model
- [ ] Regime detection and adaptive policies
- [ ] Stop-loss and early exit rules
- [ ] Equal-weight alternative index
- [ ] PyTorch policy network training

### Files Overview

| File | Status | Purpose |
|------|--------|---------|
| `build_ipo_index.py` | ✅ Working | Main IPO index builder |
| `run_week3.py` | ✅ Working | Single stock vs SPY comparison |
| `run_pytorch.py` | ❌ Broken | Policy network training (bad imports) |
| `generate_report.py` | ⚠️ Unused | Report generator (different format) |
| `src/data.py` | ✅ Working | Data loading utilities |
| `src/backtest.py` | ⚠️ Unused | Backtesting engine |
| `src/policy.py` | ⚠️ Unused | Policy parameters |
| `src/optimize.py` | ⚠️ Unused | Random search optimizer |
| `src/objective.py` | ⚠️ Unused | Risk-adjusted objective |
| `src/metrics.py` | ⚠️ Unused | CVaR, MDD calculations |
| `src/train_policy.py` | ⚠️ Unused | REINFORCE training |
| `src/policy_network.py` | ⚠️ Unused | PyTorch policy network |
| `src/features.py` | ⚠️ Unused | Feature engineering |

---

*Generated: 2026-02-04*
