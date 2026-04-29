## Offline GRU Model (Static Policy)

```mermaid
flowchart TD
    A[Raw inputs: SDC IPO metadata + CRSP daily prices/returns + market proxy returns + VIX] --> B[Normalize dates and remove duplicate ticker-date rows]
    B --> C[Pivot IPO prices to daily matrix and forward/back-fill sparse gaps]
    C --> D[Estimate shares outstanding per ticker and build 180-day market-cap-weighted IPO sleeve return]
    D --> E[Align market and IPO sleeve on common dates; clip daily returns to robust bounds]
    E --> F[Engineer features: market_return, ipo_return, 21d rolling_vol, vix]
    F --> G[Create rolling windows: each sample uses past T=126 days as X; label is next-day pair [market, IPO]]
    G --> H[Chronological split: train/validation by val_frac (no random shuffle)]

    subgraph M["Policy Network (Static Offline Regime)"]
      M1[Input batch X: shape (B,126,F)] --> M2[GRU encoder over time]
      M2 --> M3[Take last hidden state]
      M3 --> M4[MLP head: Linear -> ReLU -> Dropout -> Linear]
      M4 --> M5[Softmax over 2 assets]
      M5 --> M6[Predicted weights w_t = [w_market, w_ipo], w>=0, sum=1]
    end

    H --> M1
    M6 --> L1[Compute portfolio return per sample: r_p = sum(w * realized_asset_returns)]
    L1 --> L2[Objective = weighted sum of mean-return/log-growth reward + CVaR + variance/vol-excess + turnover/path + diversification penalties]
    L2 --> L3[Train loop: backprop, gradient clipping/rescaling, Adam, LR schedule, early stopping on validation loss]
    L3 --> L4[Keep checkpoint with best validation loss and restore best parameters]
    L4 --> W1[Final static policy parameters theta_offline*]

    W1 --> P1[Run inference on validation timeline to generate daily portfolio weights]
    P1 --> O1[Export weights path and offline performance summary]
```

## Online GRU Model (Adaptive Policy)

```mermaid
flowchart TD
    A[Same cleaned feature pipeline as offline: aligned market/IPO returns + rolling_vol + vix] --> B[Build rolling windows X,R,dates over full timeline]
    B --> S[Create decision schedule: warmup period, update frequency (D/W/M), optional decision lag]

    subgraph M["Policy Network (Adaptive Online Regime)"]
      M1[Input batch X: shape (B,126,F)] --> M2[GRU encoder]
      M2 --> M3[Last hidden state]
      M3 --> M4[MLP head]
      M4 --> M5[Softmax]
      M5 --> M6[Weights [market, IPO]]
    end

    S --> T{Iterate through scheduled decision dates}
    T --> U{Is this an update date?}

    U -- No --> I1[Reuse previous parameters theta_(t-1)]
    U -- Yes --> H1[Select history up to train_end_idx using expanding history or fixed lookback]
    H1 --> H2[Chronological split inside history: last val_frac as validation]
    H2 --> H3[Warm-start fine-tuning for epochs_step epochs on recent history]
    H3 --> H4[Compute candidate update metrics: first vs last val_loss and relative improvement]
    H4 --> G{Gate decision}

    G -- Cadence gate --> GA[Always accept candidate theta_t]
    G -- Confidence gate --> GB{History size >= threshold AND improvement thresholds passed?}
    GB -- Yes --> GA
    GB -- No --> GR[Reject candidate and revert model/optimizer state]

    GA --> I2[Apply updated parameters theta_t]
    GR --> I1

    I1 --> P[Predict allocation for current decision window]
    I2 --> P
    P --> R1[Apply allocation to realized return at eval date, record net/gross return and turnover]
    R1 --> T

    T --> Z[End of timeline]
    Z --> O1[Export online path: weights, realized returns, turnover, cost, update flags]
    Z --> O2[Export online training history and schedule diagnostics]
    Z --> O3[Export update-benefit stats: update vs no-update forward return difference]
```

### Editable sources

- `docs/diagrams/offline_gru_static_policy.mmd`
- `docs/diagrams/online_gru_adaptive_policy.mmd`
