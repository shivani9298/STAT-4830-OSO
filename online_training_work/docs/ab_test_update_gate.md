## A/B Test: Cadence-Only vs Confidence-Gated Updates

Goal: compare current cadence-only updates against a confidence gate that only applies updates when validation improvement clears thresholds.

### Variants

- **A (control):** `gate_mode=cadence` (always apply updates on scheduled update dates)
- **B (treatment):** `gate_mode=confidence` (apply only if confidence thresholds pass)

### Command

Run:

```bash
python3 scripts/sweep_online_settings.py \
  --cadences W \
  --lookbacks 252 504 \
  --epochs-step 2 \
  --gate-modes cadence confidence \
  --gate-min-val-improvement 0.0 \
  --gate-min-relative-improvement 0.0 \
  --gate-min-history-windows 252 \
  --out-csv results/ab_update_gate_results.csv
```

### Primary readout (from `results/ab_update_gate_results.csv`)

- `net_total_return`
- `net_sharpe`
- `net_max_dd`
- `n_updates_applied`
- `update_accept_rate`
- `difference_update_minus_no_update`

### Decision rule

Choose confidence-gated updates only if they improve net risk-adjusted performance and do not worsen drawdown materially:

- Higher `net_sharpe`
- Higher `net_total_return`
- Similar or better `net_max_dd`
- Non-negative `difference_update_minus_no_update`

### Notes

- Keep all non-gate knobs fixed between A and B.
- Repeat with at least two lookbacks (`252`, `504`) to reduce selection noise.
