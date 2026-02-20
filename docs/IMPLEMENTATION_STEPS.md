# IPO Portfolio Optimizer: Design Approach and Implementation Steps

## 1. Design Approach

**Goal:** Decide whether and how much to allocate to IPOs as an asset class vs the broad market (e.g. S&P 500), and output daily portfolio weights or a policy useful for a retail trader.

**Inputs:** Historical IPO index returns, market returns (e.g. SPY), and optionally other data (VIX, sector indices, rolling volatility).

**Outputs:** Daily weights (market vs IPO sleeve), summary stats, and an optional policy rule (e.g. “consider increasing IPO exposure” or a position scale for the next IPO).

**Objective (loss):** Minimize  
`L = −mean_return + λ₁·CVaR + λ₂·turnover + λ₃·volatility + λ₄·weight_path`  
so that we maximize mean return while penalizing tail risk, transaction costs, and path instability. All terms are differentiable for gradient descent.

**Model architectures to test:**

| Architecture | Description | When to use |
|-------------|-------------|-------------|
| **MLP** | Flatten or aggregate (mean/std) the window → MLP → softmax | Fast baseline, no temporal structure |
| **GRU / LSTM** | Sequence in → last hidden → MLP → softmax | Good balance of temporal structure and simplicity |
| **Transformer** | Positional encoding + encoder → pool → MLP → softmax | Long-range dependencies, more data |
| **Hybrid** | GRU or Transformer → state vector → small MLP → softmax | Interpretable “predict then allocate” |

All produce weights on the simplex (non-negative, sum to 1) via a final softmax.

---

## 2. Data: What to Gather, Where, How to Pull

### 2.1 Required series

- **Market returns**  
  Daily total or price returns for a broad market proxy (e.g. S&P 500).  
  **Where:** Yahoo Finance (SPY or ^GSPC), Alpha Vantage, FRED, or WRDS (CRSP).  
  **How to pull:**  
  - **yfinance:** `yf.download("SPY", start=..., end=...)` → Adj Close → `pct_change()`.  
  - Save as a series: date index, one column `market_return`.

- **IPO index returns**  
  A single daily return series for “IPO as an asset class” (e.g. equal-weight average of returns of stocks in their first N days after IPO).  
  **Where:** No single ticker; you must build it.  
  **How to pull:**  
  - **Option A (WRDS):** Get IPO dates (e.g. Compustat/WRDS IPO table), link to CRSP; for each date, take equal-weight average of daily returns of names in “first N days after IPO” → one `ipo_return` per date.  
  - **Option B (free):** Get an IPO list with dates (e.g. IPO calendar CSV, Ritter list). For each IPO ticker, pull first N days of prices via yfinance; build panel (date, ticker, return); for each calendar date, average return across tickers in their first N days → `ipo_return`.  
  - **Synthetic (testing):** Correlated with market + noise, e.g. `market * 1.2 + 0.01 * randn()`.

### 2.2 Optional features

- **VIX**  
  **Where:** Yahoo Finance (`^VIX`), FRED (`VIXCLS`).  
  **How:** `yf.download("^VIX", ...)` → Close; align to market dates (forward-fill if needed).

- **Rolling realized volatility**  
  From market returns: 21-day rolling std; align by date.

- **Other indices**  
  Sector ETFs (e.g. XLK, XLF) or FRED series → returns; align by date.

### 2.3 Alignment and windows

- Merge market and IPO returns (and optional features) on date; drop or fill NaNs consistently.
- Build **rolling windows:** for each date `t`, input = past `T` days (e.g. T=252), output = weights for date `t`. Store realized returns at `t` for the loss.
- **Train / validation / test:** Split by time (e.g. 2010–2018 train, 2019–2020 val, 2021–2023 test). No shuffling.

**Deliverable:** Scripts/module that produce aligned DataFrame and `(X_train, R_train, dates_train)`, `(X_val, R_val, dates_val)`, and optionally test set. `X` shape `(N, T, F)`; `R` shape `(N, n_assets)`.

---

## 3. Implementation Steps (Checklist)

### Phase 1: Data

1. **Pull market returns**  
   Implement or use `load_market_returns(start, end, ticker="SPY")` via yfinance (or WRDS). Output: pandas Series with date index.

2. **Obtain or build IPO index**  
   - Either: load from CSV with columns `date`, `ipo_return`.  
   - Or: from IPO list (ticker, ipo_date), for each ticker pull first N days of prices (yfinance), build panel, then for each date average return across names in first N days.  
   - Or: use synthetic series for testing.

3. **Align and add optional features**  
   Merge market and IPO on date; optionally add VIX, rolling vol, other indices. One DataFrame, no NaNs in required columns (or documented fill/drop).

4. **Build rolling windows and splits**  
   For each `t` in range `[T, len(df)]`, form input window `X[t] = df[t-T:t]` and realized return `R[t] = df.loc[t, ['market_return','ipo_return']]`. Split by time into train/val (and test). Deliver `X_train`, `R_train`, `dates_train`, and same for val/test.

### Phase 2: Loss

5. **Implement differentiable loss components**  
   - Mean return: `−mean(portfolio_returns)`.  
   - CVaR: differentiable approximation (e.g. soft quantile / smooth tail average).  
   - Turnover: `mean(|w_t − w_prev|)`.  
   - Return variance: `var(portfolio_returns)`.  
   - Weight path: e.g. `mean((w_t − w_prev)²)`.  
   Combine: `L = −μ + λ₁·CVaR_term + λ₂·turnover + λ₃·var + λ₄·path` with configurable λs.

### Phase 3: Models

6. **Implement MLP allocator**  
   Input: flattened window or aggregated (mean, std per feature). MLP → logits → softmax. Same interface: `forward(x) -> weights`, `x` (B, T, F).

7. **Implement GRU/LSTM allocator**  
   RNN → last hidden state → MLP → softmax. Input (B, T, F), output (B, n_assets).

8. **Implement Transformer allocator**  
   Linear project to d_model, add positional encoding, transformer encoder, pool (e.g. last token), MLP → softmax.

9. **Implement Hybrid allocator**  
   Stage 1: GRU or Transformer → state vector. Stage 2: MLP(state) → softmax. Train end-to-end.

### Phase 4: Training

10. **Single training loop**  
    For each batch: `w = model(X)`, portfolio return = `w * R`, compute `L`, backward, optimizer step. Track previous weights for turnover/path. Validation loop (no grad) for early stopping.

11. **Checkpointing**  
    Save best model (by validation loss) and training history.

### Phase 5: Evaluation and export

12. **Run inference**  
    For validation or test set: `weights = model(X)` (batched). Compute portfolio stats: mean return, vol, Sharpe, max drawdown, average turnover.

13. **Export daily weights**  
    CSV with columns `date`, `weight_market`, `weight_IPO` (and optional asset names).

14. **Export summary**  
    Text file: mean return, volatility, Sharpe, max drawdown, avg turnover, average IPO weight, % days IPO weight > 20%.

### Phase 6: Policy and comparison

15. **Optional policy layer**  
    Map current IPO weight (or tilt) to a position scale for the next IPO and a short rule (e.g. “consider increasing IPO exposure when weight > 20%”). Print or document for the trader.

16. **Compare architectures**  
    Run training with `--model mlp`, `--model gru`, `--model lstm`, `--model transformer`, `--model hybrid`. Compare validation loss and test-set metrics (Sharpe, drawdown, turnover) and choose the best for production or further tuning.

---

## 4. How to Run (This Repo)

- **One architecture:**  
  `python scripts/run_ipo_optimizer.py --model gru --epochs 50 --weights-csv output/daily_weights.csv`

- **All architectures (for testing):**  
  ```bash
  for m in mlp gru lstm transformer hybrid; do
    python scripts/run_ipo_optimizer.py --model $m --epochs 50 --checkpoint-dir output/ckpt_$m --weights-csv output/weights_$m.csv
  done
  ```

- **With your own IPO index CSV:**  
  `python scripts/run_ipo_optimizer.py --ipo-csv path/to/ipo_returns.csv --model gru`

Data is loaded via `src.data_layer.get_data()` (market from yfinance; IPO from CSV, from ticker list, or synthetic). Training uses `src.train.run_training()` with the chosen `model_type`. Weights and summary are written by `src.export`.
