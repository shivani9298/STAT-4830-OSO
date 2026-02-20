# IPO Portfolio Optimization with GRU-Based Allocation

## Problem Statement (~1/2 page)

### What We Are Optimizing

We optimize **daily portfolio weights** between two asset classes:
1. **Market** – S&P 500 (82%) + Dow Jones (18%) proxy from CRSP (SPY/DIA)
2. **IPO Index** – Market-cap weighted index of recent IPOs, each stock held for 180 trading days post-IPO

The model outputs weights \(w \in [0,1]^2\) with \(w_1 + w_2 = 1\) via a **GRU neural network** that maps rolling windows of past returns to allocations. We minimize a differentiable loss that balances return, risk, and stability.

### Why This Problem Matters

IPOs exhibit distinct return patterns: higher volatility (3–4× the market), mean reversion in the first 6 months, and information asymmetry. A systematic allocator that adjusts exposure based on recent history can improve risk-adjusted returns compared to static benchmarks. Retail and institutional investors can use this as a tilt signal (how much IPO vs market exposure).

### Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Sharpe Ratio | > 1.5 | Beat market-only risk-adjusted return |
| Max Drawdown | < 15% | Capital preservation |
| Annualized Return | > Market | Outperform SPY baseline |
| Turnover | Low | Minimize transaction costs |

### Constraints

- **Long-only, fully invested**: \(w_i \geq 0\), \(\sum_i w_i = 1\) (softmax output)
- **No shorting or leverage**
- **Daily rebalancing**: Weights produced daily for next-day execution

### Data Requirements

- **IPO prices & shares**: CRSP daily stock file (`crsp.dsf`) – split-adjusted prices, `shrout` for shares
- **IPO dates**: SDC Platinum (`sdc.wrds_ni_details`) – `ipodate` for each IPO
- **Market returns**: CRSP SPY and DIA
- **Date range**: 2020-01-01 to 2024-12-31 (limited by CRSP lag)

### What Could Go Wrong

1. **Survivorship bias**: IPO index only includes stocks still in CRSP; delistings excluded
2. **Look-ahead**: Validation/test split is temporal; no future data in training
3. **Overfitting**: Hyperparameters tuned on validation may not generalize
4. **Regime shift**: 2020–2024 includes strong IPO performance; future regimes may differ

---

## Technical Approach (~1/2 page)

### Mathematical Formulation

**Objective (minimize loss):**
<img width="471" height="37" alt="image" src="https://github.com/user-attachments/assets/03fffce2-ed4b-461a-9df5-bf4285e0b8ad" />


Where:
<img width="410" height="188" alt="image" src="https://github.com/user-attachments/assets/7943f4b1-6444-4581-814d-03c758be62c7" />


**Constraints:** Implicit via softmax (non-negative, sum to 1).

### Algorithm: GRU-Based Differentiable Optimization

<img width="378" height="154" alt="image" src="https://github.com/user-attachments/assets/33755d0a-e779-4ee0-a192-30170040c1d2" />


**Why GRU?** Sequence modeling captures temporal dependence in returns; the model can adapt allocations to recent momentum/volatility regimes. Differentiable end-to-end enables gradient-based tuning.

### PyTorch Implementation
<img width="415" height="138" alt="image" src="https://github.com/user-attachments/assets/0979c8b9-3951-4fdb-8c6c-ae84821e369f" />

### Validation Methods

- **Time-series split**: Last 20% of dates for validation
- **Metrics**: Validation Sharpe, max drawdown, avg IPO weight
- **Baselines**: Market-only, IPO-only, Equal 50/50

### Resource Requirements

- **WRDS subscription** (CRSP, SDC)
- **Runtime**: ~2–3 min per training run; ~2–3 hr for 32-config grid search
- **Memory**: <2 GB

---

## Initial Results (~1/2 page)

### Evidence Implementation Works

- Data pipeline: SDC + CRSP loads 770K rows, 1,136 IPO tickers, 1,248 days of IPO index returns
- Training converges in 11–14 epochs with early stopping
- Weights satisfy simplex constraint; no NaN/Inf
- Loss curve (train/val) saved to `figures/ipo_optimizer_loss.png`

### Performance Metrics (Validation Period)

| Strategy | Total Return | Ann. Return | Ann. Vol | Sharpe | Max DD |
|----------|-------------|-------------|----------|--------|--------|
| **Model Portfolio** | **34.39%** | **39.44%** | 13.52% | **2.53** | **-7.66%** |
| Market only | 17.26% | 19.62% | 12.22% | 1.53 | -7.89% |
| IPO only | 166.59% | 201.35% | 30.68% | 3.75 | -10.08% |
| Equal 50/50 | 78.16% | 91.50% | 19.26% | 3.47 | -7.20% |

**Summary**: Model allocates ~16% to IPO on average, achieving higher Sharpe (2.53) and lower drawdown (-7.66%) than market-only, while reducing volatility vs IPO-only. Volatility penalty successfully limits IPO tilt.

### Test Case Results

- **Hyperparameter tuning**: 32 configs; best: window_len=126, \(\lambda_{\text{cvar}}=1.0\), \(\lambda_{\text{vol\_excess}}=0.5\), target_vol=0.25
- **Avg turnover**: ~1.3e-5 (negligible; model outputs near-constant weights)
- **Policy output**: "Model suggests moderate or low IPO allocation"; position scale 0.32

### Current Limitations

1. **Near-static weights**: Model learns ~84% market / 16% IPO with minimal day-to-day variation; limited tactical tilting
2. **No transaction costs**: Turnover is tiny; real costs not modeled
3. **Validation only**: No true out-of-sample test period (e.g., 2025)
4. **Display**: Avg turnover rounds to 0.0000 (format precision)

### Resource Usage

- Single run: ~2.5 min (WRDS + training)
- Tuning (32 configs): ~2–3 hr
- Memory: <2 GB

### Unexpected Challenges

- Without volatility/diversification penalties, model went 100% IPO (in-sample optimal); required \(\lambda_{\text{vol\_excess}}\) to get balanced allocation
- WRDS connection and data volume add latency vs local CSV
- Some datasets (e.g., Compustat) replaced by CRSP for consistency

---

## Next Steps (~1/2 page)

### Immediate Improvements

1. **True out-of-sample test**: Reserve final months (e.g., 2025) for test; report metrics only on that period
2. **Display turnover**: Use `.6f` or scientific notation so tiny turnover is visible
3. **Transaction costs**: Add explicit cost term (e.g., bps per unit turnover) to loss

### Technical Challenges

1. **Dynamic allocation**: Increase model capacity or features so weights respond more to regime; current GRU may be underfitting to time variation
2. **Survivorship bias**: Include delisted IPOs if data allows; otherwise document limitation
3. **Hyperparameter robustness**: Sensitivity analysis; avoid overfitting to validation

### Questions Needing Help

1. Best practice for temporal train/val/test splits with limited data (2020–2024)?
2. How to model transaction costs in a differentiable way for gradient-based optimization?
3. Should we add regime indicators (VIX, momentum) as features?

### Alternative Approaches

1. **Reinforcement learning**: PPO/SAC for sequential allocation
2. **Ensemble**: Combine GRU with simple rule (e.g., vol target)
3. **Larger IPO universe**: Broader SDC filter; factor-based IPO selection
4. **Online learning**: Update model incrementally as new data arrives

### What We've Learned

1. Institutional data (WRDS) enables realistic IPO index construction at scale
2. Pure return maximization leads to 100% IPO; risk penalties are essential for diversification
3. GRU produces stable weights; may need architectural changes for more tactical tilting
4. Volatility penalty (target vol cap) effectively limits IPO exposure and improves drawdown
