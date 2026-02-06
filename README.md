# Online Portfolio Optimization: S&P 500 vs IPO 180-Day Index

**STAT 4830 | Spring 2026 | University of Pennsylvania**

This project uses **Online Gradient Descent (OGD)** to dynamically allocate a portfolio between the S&P 500 (SPY ETF) and a custom market-cap weighted index of recent IPOs. The optimizer maximizes a risk-adjusted fitness score that balances returns against volatility, drawdown, and transaction costs.

## Key Results (2020--2025 Backtest)

| Strategy | Total Return | Ann. Return | Ann. Vol | Sharpe | Max Drawdown | Calmar |
|----------|-------------|-------------|----------|--------|--------------|--------|
| **OGD Portfolio** | **193.5%** | **28.5%** | 20.1% | **1.42** | **-26.2%** | **1.09** |
| Equal Weight | 577.7% | 56.2% | 35.0% | 1.60 | -51.3% | 1.09 |
| S&P 500 Only | 86.1% | 15.6% | 16.1% | 0.97 | -24.5% | 0.64 |
| IPO Index Only | 1699.3% | 96.0% | 61.4% | 1.56 | -73.1% | 1.31 |

OGD achieves a Sharpe ratio of 1.42 with only -26% max drawdown, significantly improving risk control compared to IPO-only exposure (-73% drawdown).

---

## Replication Guide

### Prerequisites

- Python 3.10+
- pip
- Git
- Internet connection (for Yahoo Finance data)

### 1. Clone the Repository

```bash
git clone https://github.com/shivani9298/STAT-4830-OSO.git
cd STAT-4830-OSO
```

### 2. Install Dependencies

```bash
pip install torch numpy pandas yfinance matplotlib pytest
```

Optional (for future hyperparameter optimization work):
```bash
pip install scipy scikit-optimize optuna
```

### 3. Run the Main Notebook

Open and run the primary implementation notebook:

```bash
jupyter notebook notebooks/week3_implementation.ipynb
```

Run all cells in order. The notebook will:
1. Fetch daily price data and shares outstanding for ~80 IPO tickers and SPY from Yahoo Finance (~1-2 min)
2. Construct the market-cap weighted 180-day IPO index
3. Run the OGD walk-forward backtest over ~1,200 trading days (~3 min)
4. Print performance metrics and generate visualizations

**Total runtime**: ~5 minutes on a standard laptop. No GPU required.

### 4. Run Tests

```bash
pytest tests/test_basic.py -v
```

This runs 21 unit tests covering simplex projection, max drawdown computation, the OGD allocator, and fitness score behavior.

### 5. Explore Pre-Computed Results

If you want to inspect results without re-running the notebook, pre-computed outputs are in `results/`:

| File | Description |
|------|-------------|
| `week3_metrics.csv` | Performance metrics for all strategies |
| `week3_returns.csv` | Daily portfolio returns |
| `week3_weights.csv` | Daily OGD allocation weights (SPY vs IPO) |

---

## Technical Approach

### Objective Function

The OGD optimizer maximizes:

```
F(w) = mean_return - lambda_1 * variance + lambda_2 * max_drawdown - lambda_3 * turnover
```

Where:
- `w` = portfolio weights [SPY, IPO_INDEX], constrained to the probability simplex (long-only, fully invested)
- `lambda_1 = 20.0` (risk aversion), `lambda_2 = 8.0` (drawdown penalty), `lambda_3 = 0.15` (turnover penalty)

### Algorithm

```
For each day t:
    1. Extract trailing 126-day window of returns
    2. Compute fitness gradient via PyTorch autograd
    3. Gradient ascent: w_new = w + lr * gradient
    4. Project onto simplex (Euclidean projection, O(n log n))
    5. Apply weights to next day's returns (walk-forward, no look-ahead)
    6. Decay learning rate (lr *= 0.999)
```

### IPO Index Construction

- ~80 IPOs from 2020--2024 with a 180 trading-day holding period per stock
- Market-cap weighted (price x shares outstanding)
- Average ~8.5 constituents at any given time

---

## Repository Structure

```
.
├── README.md                          # This file
├── report.md                          # Project report (problem, approach, results)
├── self_critique.md                   # OODA self-assessment
├── notebooks/
│   └── week3_implementation.ipynb     # Main replication notebook
├── src/
│   ├── __init__.py                    # Package exports
│   ├── model.py                       # OGD allocator, simplex projection, max drawdown
│   └── utils.py                       # Data fetching, IPO index, metrics, backtest
├── tests/
│   └── test_basic.py                  # Unit tests (21 tests)
├── results/                           # Pre-computed outputs (CSV)
├── figures/                           # Visualization outputs
└── docs/
    ├── development_log.md             # Design decisions and progress
    ├── llm_exploration/               # AI collaboration logs
    └── assignments/                   # Course assignment specs
```

### Source Code Overview

| Module | Key Contents |
|--------|-------------|
| `src/model.py` | `OnlineOGDAllocator` class, `project_to_simplex()`, `max_drawdown_from_returns()` |
| `src/utils.py` | `fetch_price_and_shares()`, `build_ipo_index()`, `calculate_metrics()`, `run_backtest()` |

---

## Data

All data is fetched live from [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` Python package. No local data files need to be downloaded manually.

- **S&P 500 proxy**: SPY ETF
- **IPO tickers**: ~80 major US IPOs from 2020--2024 (hardcoded in the notebook)
- **Date range**: 2020-01-01 to 2025-01-14

**Note**: Results may vary slightly across runs due to Yahoo Finance data updates and retroactive price adjustments.

---

## Known Limitations

1. **Look-ahead bias in market caps**: Uses current shares outstanding for all historical dates (should use quarterly SEC filings)
2. **Survivorship bias**: Only includes IPOs that still trade; excludes delisted/acquired companies
3. **No real transaction costs**: Turnover penalty is a proxy, not actual bid-ask spreads
4. **Heuristic hyperparameters**: Penalty coefficients not yet systematically optimized

See `self_critique.md` for a detailed discussion.

---

## Reproducing Specific Figures

The notebook generates two key visualizations saved to `figures/`:
- **Cumulative returns** comparing OGD, equal-weight, SPY-only, and IPO-only strategies
- **Weight evolution** showing how OGD shifts between SPY and IPO exposure over time

These are produced in the final cells of `notebooks/week3_implementation.ipynb`.

---

## License

This project was developed for STAT 4830 at the University of Pennsylvania.
