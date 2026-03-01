# IPO Portfolio Optimization with GRU-Based Allocation

## Problem Statement

### What We Are Optimizing

We optimize **daily portfolio weights** between two asset classes: (1) a **market index** composed of S&P 500 (82%) and Dow Jones (18%) proxied via CRSP daily returns for SPY and DIA, and (2) a **custom IPO index** — a market-cap-weighted basket of recent IPOs sourced from SDC Platinum, where each stock is held for 180 trading days post-IPO. The model outputs weights $w = [w_{\text{market}}, w_{\text{IPO}}]$ with $w_i \geq 0$, $\sum w_i = 1$, produced by a GRU neural network that ingests rolling windows of past returns, volatility, and VIX levels.

### Why This Problem Matters

IPOs exhibit return dynamics distinct from broad equities: higher volatility (3–4× the market), mean reversion in the first six months, and information asymmetry between institutional and retail participants. A static allocation ignores regime shifts — the COVID crash of March 2020, the IPO boom of 2021, and the correction of 2022 all demanded different exposures. A systematic, data-driven allocator that adjusts IPO tilt based on recent market conditions can improve risk-adjusted returns and provide actionable signals for both retail and institutional investors.

### How We Measure Success

| Metric | Target | Achieved |
|--------|--------|----------|
| Annualized Sharpe Ratio | > 1.5 | **3.27** |
| Max Drawdown | < 15% | **-7.47%** |
| Annualized Return | > Market (SPY) | **65.27%** vs 21.24% |
| Avg Turnover | Low (< 0.01/day) | **0.0031** |

### Constraints

- Long-only, fully invested (softmax output guarantees non-negative weights summing to 1)
- No shorting or leverage
- Daily rebalancing frequency; weights produced for next-day execution
- Data limited to 2020-01-01 through 2024-12-31 (CRSP publication lag)

### Data Requirements

| Source | Content | Records |
|--------|---------|---------|
| SDC Platinum (`sdc.wrds_ni_details`) | IPO dates for each issuer | 1,136 tickers |
| CRSP Daily Stock File (`crsp.dsf`) | Split-adjusted prices, shares outstanding | 770,418 rows |
| CRSP SPY + DIA | Market returns (82%/18% blend) | 1,258 trading days |
| CBOE via WRDS (`cboe.cboe`) | Daily VIX closing level | 1,535 days |

### What Could Go Wrong

1. **Survivorship bias**: The IPO index only includes stocks still in CRSP; delistings and failures are excluded, inflating index returns.
2. **Regime dependence**: 2020–2024 includes an unprecedented IPO boom followed by a correction — future regimes may differ substantially.
3. **Overfitting to validation**: Hyperparameters are tuned to maximize validation Sharpe; without a true out-of-sample test period, reported metrics may overstate performance.
4. **Look-ahead risk**: Although the train/validation split is strictly temporal, the IPO index construction uses shares outstanding data that may be backfill-adjusted.

---

## Technical Approach

### Mathematical Formulation

The model minimizes a composite differentiable loss over mini-batches of rolling windows:

<img width="616" height="81" alt="image" src="https://github.com/user-attachments/assets/e019b53a-dc1d-4dbb-aacd-fcbca45ad6f7" />


Where $r_p = w \cdot r$ is the portfolio return, $L_{\text{cvar}}$ is a soft-sorted approximation of CVaR at the 5% level using temperature-scaled exponential weighting, and $\hat{\sigma}_{\text{ann}} = \text{std}(r_p) \cdot \sqrt{252}$. Constraints are implicit: the softmax output layer guarantees $w_i \geq 0$ and $\sum w_i = 1$.

### Algorithm Choice and Justification

We use a **GRU (Gated Recurrent Unit)** recurrent neural network. The GRU processes an 84-day rolling window of 4 features (market return, IPO return, 21-day rolling volatility, VIX level), extracts the final hidden state, and passes it through a 2-layer MLP with softmax output. GRUs are chosen over LSTMs for fewer parameters and faster training on our small dataset (~1,100 samples). The entire pipeline is differentiable, enabling end-to-end gradient-based optimization of portfolio allocation directly through the loss function — no labels or supervised targets required.

### PyTorch Implementation Strategy

- **`src/model.py`**: `AllocatorNet` — GRU(input=4, hidden=64, layers=1) → MLP(64→64→2) → softmax
- **`src/losses.py`**: Modular loss components (`loss_mean_return`, `cvar_smooth`, `loss_turnover`, `loss_return_variance`, `loss_vol_excess`, `loss_weight_path`) composed into `combined_loss`
- **`src/train.py`**: Mini-batch training loop with early stopping (patience=10), gradient clipping (max norm 1.0), Adam optimizer with weight decay 1e-5
- **`src/wrds_data.py`**: WRDS data loading (SDC IPO dates, CRSP prices, CBOE VIX)
- **`src/data_layer.py`**: Rolling window construction, feature engineering, temporal train/val split

### Validation Methods

- **Temporal split**: First 80% of dates for training, last 20% for validation (no shuffle — strict time ordering)
- **Grid search**: 288 hyperparameter configurations searched over `lambda_vol`, `lambda_cvar`, `lambda_turnover`, `lambda_path`, `lambda_vol_excess`, `target_vol_annual`, and `window_len`, optimizing validation Sharpe
- **Baselines**: Market-only, IPO-only, and equal 50/50 portfolios evaluated on the same validation period

### Resource Requirements

- WRDS subscription (CRSP, SDC, CBOE)
- Single training run: ~1 minute on CPU
- Full 288-config grid search: ~19 minutes
- Memory: < 2 GB

---

## Initial Results

### Evidence the Implementation Works

- Data pipeline loads 770,418 rows across 1,136 IPO tickers with 1,248 days of valid IPO index returns
- Real VIX data (1,535 days from CBOE, range 11.86–82.69) replaces the previous constant placeholder
- Training converges in 11 epochs with early stopping; loss goes negative (return term dominates penalties)
- Weights satisfy the simplex constraint on every prediction; no NaN/Inf values
- Loss curves (train and validation) saved to `figures/ipo_optimizer_loss_semilog.png`

### Performance Metrics (Validation Period: Jan–Dec 2024)

| Strategy | Total Return | Ann. Return | Ann. Vol | Sharpe | Max DD |
|----------|-------------|-------------|----------|--------|--------|
| **Model Portfolio** | **58.81%** | **65.27%** | **15.74%** | **3.27** | **-7.47%** |
| Market only | 19.40% | 21.24% | 12.25% | 1.63 | -7.89% |
| IPO only | 192.78% | 221.19% | 31.03% | 3.92 | -10.08% |
| Equal 50/50 | 88.52% | 99.11% | 19.39% | 3.65 | -7.20% |

The model allocates ~69% market / 31% IPO on average, achieving a Sharpe of 3.27 with tighter drawdown (-7.47%) than IPO-only (-10.08%) and higher return than market-only (65% vs 21%).

### Hyperparameter Tuning Results

The 288-config grid search found the optimal penalty weights:

| Parameter | Previous (hardcoded) | Tuned (best) |
|-----------|---------------------|--------------|
| `lambda_turnover` | 0.01 | **0.0025** |
| `lambda_path` | 0.01 | **0.01** |
| `lambda_vol` | 0.5 | **1.0** |
| `lambda_cvar` | 0.5–1.0 | **1.0** |
| `window_len` | 126 | **84** |

Reducing the turnover penalty from 0.01 to 0.0025 allowed the model to rebalance more actively (avg turnover increased from ~1e-5 to 0.0031), while heavier risk penalties (`lambda_vol=1.0`, `lambda_cvar=1.0`) kept volatility controlled. The shorter 84-day lookback outperformed 126 days.

### Current Limitations

1. **No true out-of-sample test**: All metrics are on the validation set (last 20% of 2020–2024). No held-out 2025 test period exists.
2. **Survivorship bias**: The IPO index excludes delisted or failed IPOs, potentially inflating returns.
3. **Regime dependence**: The 2020–2024 window includes strong IPO performance; model behavior in a bear IPO market is untested.
4. **Limited tactical variation**: While IPO weight now varies (31% average vs the prior near-constant 16%), day-to-day changes remain modest.

### Resource Usage

- Full pipeline (WRDS data load + training + inference): ~65 seconds
- 288-config grid search: ~19 minutes
- Peak memory: < 2 GB

### Unexpected Challenges

- **Turnover and path penalties were never tuned** in the original implementation — both were hardcoded at 0.01. Adding them to the grid search improved Sharpe from 3.07 to 3.21.
- **VIX was a constant placeholder** (20.0 on every day), providing zero information to the model. Replacing it with real CBOE VIX data (which spiked to 82.69 during COVID) improved Sharpe further to 3.27.
- **Training loss goes negative**, which the original semi-log plot masked by taking `abs()`. A symmetric log scale was needed to properly visualize convergence.
- Without volatility penalties, the model allocated 100% to IPOs (the in-sample optimal). Risk penalties are essential for producing balanced allocations.

---

## Next Steps

### Immediate Improvements

1. **True out-of-sample test**: Reserve the final 3 months of data as a never-touched test set, or use 2025 data from an alternative source (e.g., yfinance for SPY/IPO tickers) to bridge the CRSP lag.
2. **Transaction cost modeling**: The current turnover penalty is a regularizer, not a cost model. Adding an explicit cost term (e.g., 5–10 bps per unit turnover) to the loss would make the model directly optimize net-of-cost returns.
3. **Feature expansion**: Add momentum indicators (e.g., 20-day vs 60-day return spread), IPO-specific signals (days since IPO, number of active IPOs), or credit spreads as additional input features.

### Technical Challenges

1. **Weight dynamics**: The model still produces relatively smooth allocations. Increasing model capacity (deeper GRU, attention layers) or reducing stability penalties further may enable more responsive tactical tilting — but risks overfitting.
2. **Survivorship bias correction**: Include delisted IPO stocks with their terminal returns (CRSP delisting returns) to avoid upward bias in the IPO index.
3. **Hyperparameter robustness**: Run sensitivity analysis around the best config to confirm the Sharpe improvement is stable, not an artifact of a single lucky seed.

### Questions Needing Help

1. How should we construct a fair out-of-sample test when CRSP data lags by several months? Is bridging with yfinance data methodologically sound?
2. What is the best practice for differentiable transaction cost modeling in end-to-end portfolio optimization?
3. Should the model architecture be modified (e.g., attention mechanism, larger hidden size) given only ~1,100 training samples, or does the small dataset favor simplicity?

### Alternative Approaches to Try

1. **Reinforcement learning** (PPO/SAC): Treat allocation as a sequential decision problem with explicit reward shaping for risk-adjusted returns.
2. **Ensemble methods**: Combine GRU predictions with a simple vol-targeting rule or momentum signal.
3. **Online/incremental learning**: Update model weights as new data arrives rather than full retraining.
4. **Broader IPO universe**: Expand beyond SDC to include SPAC IPOs or international listings.

### What We've Learned

1. **Penalty tuning matters**: Hardcoded regularization weights left performance on the table. Systematic grid search over `lambda_turnover` and `lambda_path` yielded a meaningful Sharpe improvement (3.07 → 3.27).
2. **Feature quality matters**: Replacing a constant VIX placeholder with real CBOE data gave the model actual volatility regime information, improving both return and risk metrics.
3. **Risk penalties are essential**: Pure return maximization drives 100% IPO allocation. The interplay between `lambda_vol`, `lambda_cvar`, and `lambda_vol_excess` is what produces a balanced, investable portfolio.
4. **Institutional data at scale works**: WRDS (SDC + CRSP + CBOE) provides a rigorous, reproducible data foundation with 1,136 IPO tickers and 1,535 days of VIX — far superior to yfinance-based prototyping.
5. **Simple models converge fast**: The GRU learns in ~4 effective epochs on this dataset. The bottleneck is data quantity, not model capacity.
