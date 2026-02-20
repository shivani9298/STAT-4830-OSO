# IPO Portfolio Optimizer – GRU-Based Allocation

**STAT 4830 | Spring 2026 | University of Pennsylvania**

A **GRU neural network** allocates daily portfolio weights between a **market index** (S&P 500 + Dow proxy) and a **custom IPO index** (market-cap weighted, 180-day post-IPO). The model minimizes a differentiable loss that balances return, volatility, CVaR, and turnover. Data is pulled from **WRDS** (SDC + CRSP).

## Key Results (Validation Period, 2020–2024)

| Strategy      | Total Return | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|--------------|--------------|-------------|----------|--------|--------------|
| **Model**    | **34.39%**   | **39.44%**  | 13.52%   | **2.53** | **-7.66%** |
| Market only  | 17.26%       | 19.62%      | 12.22%   | 1.53   | -7.89%       |
| IPO only     | 166.59%      | 201.35%     | 30.68%   | 3.75   | -10.08%      |
| Equal 50/50  | 78.16%       | 91.50%      | 19.26%   | 3.47   | -7.20%       |

The model allocates ~16% to the IPO index on average, achieving **Sharpe 2.53** and **Max DD -7.66%**, outperforming market-only (Sharpe 1.53) with controlled volatility.

---

## Quick Start

### Prerequisites

- Python 3.10+
- **WRDS account** (CRSP + SDC)
- `WRDS_USERNAME` and `WRDS_PASSWORD` environment variables (or interactive login)

### 1. Clone and Install

```bash
git clone https://github.com/shivani9298/STAT-4830-OSO.git
cd STAT-4830-OSO
pip install torch numpy pandas matplotlib wrds
```

### 2. Run the Optimizer

```bash
python run_ipo_optimizer_wrds.py
```

This script:

1. Connects to WRDS and loads IPO data from SDC + CRSP
2. Builds the IPO index and market returns
3. Trains the GRU allocator with the best hyperparameters (or defaults)
4. Exports weights to `results/ipo_optimizer_weights.csv` and a summary to `results/ipo_optimizer_summary.txt`
5. Saves a loss plot to `figures/ipo_optimizer_loss.png`

**Runtime**: ~2–3 minutes.

### 3. Hyperparameter Tuning (Optional)

```bash
python tune_hyperparameters_wrds.py
```

Grid search over window length, volatility penalties, CVaR, etc. Saves the best config to `results/ipo_optimizer_best_config.json`; `run_ipo_optimizer_wrds.py` will use it on the next run.

**Runtime**: ~1–3 hours depending on grid size.

### 4. Jupyter Notebook

```bash
jupyter notebook notebooks/week4_implementation.ipynb
```

Step-by-step notebook with problem setup, implementation, and validation.

---

## Technical Approach

### Model Architecture

- **Input**: Rolling window of past returns (e.g., 126 days × features)
- **GRU** → last hidden state → **MLP** → **softmax** → weights on [market, IPO]
- Output satisfies long-only, fully invested (simplex)

### Objective (Loss to Minimize)

$$
\mathcal{L} = -\mu_p + \lambda_{\text{cvar}} L_{\text{cvar}} + \lambda_{\text{vol}} \sigma_p^2 + \lambda_{\text{vol\_excess}} \max(0, \sigma_{\text{ann}} - \tau) + \lambda_{\text{turn}} \cdot \text{turnover} + \lambda_{\text{path}} \|w - w_{\text{prev}}\|^2
$$

- \(\mu_p\): mean portfolio return (maximize)
- \(L_{\text{cvar}}\): smooth CVaR (tail risk)
- \(\sigma_p^2\): variance
- **Vol excess**: penalty when annualized vol exceeds target \(\tau\) (e.g., 25%)
- **Turnover**: day-over-day weight changes
- **Path**: weight stability

### Data Sources

| Source        | Content                                   |
|---------------|-------------------------------------------|
| **SDC**       | IPO dates (`sdc.wrds_ni_details`)         |
| **CRSP**      | Daily prices, shares (split-adjusted)     |
| **CRSP SPY/DIA** | Market returns (82% / 18%)            |

- IPO index: market-cap weighted, 180 trading days per IPO
- Date range: 2020–2024 (CRSP lag)

---

## Repository Structure

```
.
├── README.md                         # This file
├── report.md                         # Week 4 report (problem, approach, results)
├── self_critique.md                 # OODA self-assessment
├── run_ipo_optimizer_wrds.py        # Main run script (WRDS)
├── tune_hyperparameters_wrds.py    # Hyperparameter grid search
├── notebooks/
│   ├── week4_implementation.ipynb   # Main notebook
│   ├── ipo_optimizer_2025_wrds.ipynb
│   └── test_wrds.ipynb
├── src/
│   ├── model.py                     # GRU/MLP allocator
│   ├── losses.py                    # Differentiable loss components
│   ├── train.py                     # Training loop
│   ├── export.py                    # Predict, stats, export
│   ├── data_layer.py                # Rolling windows, splits
│   ├── wrds_data.py                 # WRDS data loading
│   └── policy_layer.py              # Position scaling, policy rules
├── results/
│   ├── ipo_optimizer_weights.csv    # Daily weights
│   ├── ipo_optimizer_summary.txt    # Performance summary
│   └── ipo_optimizer_best_config.json  # Best tuning config
├── figures/
│   └── ipo_optimizer_loss.png       # Train/val loss curve
├── docs/
│   └── ipo_concentration_diagnosis.md
└── tests/
    └── test_basic.py
```

---

## Key Parameters

| Parameter            | Default | Description                          |
|----------------------|---------|--------------------------------------|
| `window_len`         | 126     | Days of history per prediction        |
| `val_frac`           | 0.2     | Fraction of dates for validation     |
| `lambda_vol_excess`  | 1.0     | Penalty when vol exceeds target       |
| `target_vol_annual`  | 0.25    | Target max annual vol (25%)           |
| `lambda_diversify`   | 0.0     | Diversification penalty (optional)    |
| `hidden_size`        | 64      | GRU hidden dimension                 |

---

## Known Limitations

1. **No true out-of-sample test** – Metrics are on the validation set; no held-out test period
2. **Near-constant weights** – Model outputs ~84% market / 16% IPO with minimal day-to-day change
3. **Survivorship bias** – IPO index excludes delisted stocks
4. **Turnover display** – Very small turnover (~1e-5) rounds to 0.0000 in the summary

See `report.md` and `self_critique.md` for details.

---

## License

Developed for STAT 4830 at the University of Pennsylvania.
