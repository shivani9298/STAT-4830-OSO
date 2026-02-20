# STAT-4830-OSO: IPO Portfolio Optimizer

This project implements an **IPO portfolio optimizer** that allocates between the broad market (e.g. SPY) and an **IPO index** using gradient descent. The model (GRU + MLP + softmax) is trained to maximize mean return while penalizing tail risk (CVaR), transaction costs (turnover), return volatility, and weight path instability.

## Project Structure

```
STAT-4830-OSO/
├── src/
│   ├── data_layer.py      # Load market/IPO returns, align, rolling windows, train/val split
│   ├── losses.py          # Differentiable loss: mean, CVaR, turnover, variance, path
│   ├── model.py           # AllocatorNet: GRU/LSTM + MLP + softmax
│   ├── train.py           # Training loop, validation, checkpointing
│   ├── export.py          # Predict weights, portfolio stats, export CSV and summary
│   ├── policy_layer.py    # Optional: map IPO weight to position scale / rule
│   └── wrds_data.py       # WRDS connection and CRSP stock/index data
├── scripts/
│   └── run_ipo_optimizer.py   # End-to-end: data → train → export weights & summary
├── output/                    # Created on run: daily_weights.csv, summary.txt
├── pyproject.toml        # Dependencies (uv/pip)
└── requirements.txt      # Same deps for pip
```

## Quick Start

1. **Install dependencies (use [uv](https://github.com/astral-sh/uv), or pip):**
   ```bash
   uv sync
   ```
   Or with pip: `pip install -r requirements.txt`

2. **Run the optimizer** (with uv: `uv run python scripts/run_ipo_optimizer.py ...` so the venv is used automatically): (uses synthetic IPO returns if no IPO CSV is provided):
   ```bash
   python scripts/run_ipo_optimizer.py --model gru --epochs 50 --weights-csv output/daily_weights.csv --summary-txt output/summary.txt
   ```
   **Model choices:** `--model mlp | gru | lstm | transformer | hybrid` to test different architectures.
   With your own IPO index returns CSV (columns: `date`, `ipo_return`):
   ```bash
   python scripts/run_ipo_optimizer.py --ipo-csv path/to/ipo_returns.csv --model gru --epochs 50
   ```
   See **docs/IMPLEMENTATION_STEPS.md** for the full design, data sources, and a step-by-step implementation checklist.

## Data Layer

- **Market returns:** Fetched via yfinance (SPY) in `data_layer.load_market_returns()`, or from WRDS CRSP index in `wrds_data.load_market_returns_wrds()`. Use `get_data(..., market_source="wrds", wrds_conn=conn)` to use WRDS.
- **WRDS (CRSP):** Use `src.wrds_data`: `get_connection()` then `load_market_returns_wrds(conn, start, end)` for index returns, or `load_stock_prices_wrds(conn, start, end, tickers=[...])` / `load_stock_returns_wrds(...)` for stock-level data. Requires a WRDS account and, for remote access, two-factor auth. Optional: set `WRDS_USERNAME` in the environment.
- **IPO index:** Either from a CSV (`load_ipo_index_from_csv`) or built from a list of (ticker, ipo_date) via `build_ipo_index_from_tickers()` using yfinance. If neither is provided, synthetic IPO returns (correlated with market + noise) are used.
- **Alignment:** Market and IPO series are merged on date; optional features (rolling vol, VIX) can be added. Rolling windows (e.g. T=252) are built for each date; train/validation split is by time.

## Loss

Minimize:
`L = −mean_return + λ₁·CVaR + λ₂·turnover + λ₃·volatility + λ₄·weight_path`

All terms are differentiable (CVaR uses a smooth approximation) so the full pipeline is trained with gradient descent.

## Outputs for a Retail Trader

- **daily_weights.csv:** Columns `date`, `weight_market`, `weight_IPO`.
- **summary.txt:** Mean return, volatility, Sharpe, max drawdown, average turnover, average IPO weight, % days IPO weight > 20%.
- **Policy:** The script prints a simple rule (e.g. “Consider increasing IPO exposure”) and a suggested position scale for the next IPO based on the current model IPO weight.

## Optional Policy Layer

`policy_layer.ipo_tilt_to_position_scale(ipo_weight)` maps the current IPO sleeve weight to a position scale for the next IPO. `policy_rule(ipo_weight, threshold)` returns a short text recommendation.
