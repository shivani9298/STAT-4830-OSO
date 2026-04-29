# Self-Critique (OODA) — April 10, 2026


---

## OBSERVE

the **problem** (market vs IPO sleeves, sector heads) is clear; the **gap between training loss and economic metrics** is stated but easy to skim past. Re-running mental checks: **GRU/LSTM** behave better on 2010+ data; **transformer + 1970s history** was a negative surprise worth dropping; **context window** tweaks did little; **sectors** and **gradient clipping** helped signifcantly our loss curves are way better now! 

---

## ORIENT

### Strengths (max 3)

- **Clear problem–metric distinction:** we explicitly warn that **val_loss** (mean return + CVaR + vol penalties) is not a pure Sharpe objective—better than pretending loss equals trading PnL.
- **Credible iteration story:** tried **GRU, LSTM, transformer**; rejected **long-history 1970** panels; kept **2010+**; **sectors** and **grad clipping** are concrete, testable choices.
- **Path toward rigor:** **path metrics**, **retail composite** selection, **walk-forward** script, and **log-growth / segment** losses show intent to align optimization with **horizon and drawdown** concerns—not only one-step losses.

### Areas for improvement (max 3)

- **Verification of regularization wiring:** report claims **dropout/weight decay** risks; we must **confirm** `run_training` passes `dropout` into `build_model` and that increasing them changes runs measurably.
- **Empirical evidence:** add **one small table** (GRU vs LSTM, key path metrics on **the same split**) and **one walk-forward summary** (even 2–3 anchors) so “initial results” are not only narrative.

### Critical risks / assumptions (2–3 sentences)

We assume **WRDS-based** panels and sector labels are stable enough for **2010–2024** conclusions; **IPO regimes** may shift post-sample. We assume **validation** metrics approximate deployable behavior—**without** transaction costs, slippage, or does not beat the 50/50 the entire timeframe, but still improvement from before. 
---

## DECIDE — Concrete next actions (max 3, ~1 week)

1. **One-page math block:** write the **full loss** as implemented (batch terms + optional segment term + penalties) with λ list matching `DEFAULTS` / config keys.
2. **Regularization audit:** trace **`dropout` and `weight_decay`** from JSON → `run_training` → `build_model` / sector builder; run **one** GRU ablation: baseline vs **higher dropout** vs **higher weight decay**, fixed seed, same split.
3. **Evidence row:** produce **`walk_forward_summary_paths.csv`** for **two anchors** + paste **top-line** Sharpe / max DD vs **50/50** into the next report revision.

---

## ACT — Resource needs (2–3 sentences)

**Compute:** walk-forward multiplies training time; schedule overnight runs or reduce epochs for debugging.**Knowledge:** if we adopt **AdamW**, read PyTorch docs vs Adam+L2 for small models.**Help:** peer review of whether **val_retail_composite** weights (Sharpe, Sortino, drawdown penalty) match stakeholder preferences—tuning **`selection_drawdown_penalty`** is partly subjective.


