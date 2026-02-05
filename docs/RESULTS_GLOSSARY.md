# Optimization Results — Variable & Term Definitions

This document defines every variable and term in the random-search optimization output and explains what your numbers mean.

---

## How the score is actually generated (and why it’s low)

The score is produced by this pipeline:

1. **Episodes**  
   With the rich CSV (`ipo_stock_2010_2018_v2.csv`) you have **834 episodes**. Each episode has only **4 price bars**: day0 (first day), ~day5 (in week), ~day21 (in month), ~day252 (in year). So “day1” in code is really the **in-week** close (~5 days later), not the next calendar day.

2. **Per-episode decision**  
   For each episode, `decide_trade()` decides:
   - **Participate or not** using `participate_threshold`: participate only when |day1_close − day0_close| / day0_close ≥ threshold (i.e. a large move from first day to in-week).
   - If participate: **entry_day**, **exit_day** (entry + hold_k), **weight**.

3. **Per-episode backtest**  
   For each episode we get one **net return**:
   - If skip: `net_ret = 0`.
   - If trade: `entry_px = episode.df.iloc[entry_day]['close']`, `exit_px = episode.df.iloc[exit_day]['close']`, `gross_ret = (exit_px/entry_px) − 1`, `cost = (cost_bps/1e4)*weight`, `net_ret = weight * gross_ret − cost`.

4. **Aggregation**  
   - **E[R]** = mean of `net_ret` over **all 834 episodes** (including skips, which are 0).
   - **CVaR** = expected loss in the worst 10% of those 834 net returns.
   - **E[Cost]** = mean of cost over all episodes.
   - **MDD** = max drawdown of the equity curve (equity = cumulative product of (1 + net_ret)).
   - **Score** = `E[R] − λ·CVaR − κ·E[Cost] − μ·MDD` (default λ = κ = μ = 1).

**Why we use “per opportunity” (all IPOs) instead of “per executed trade”:**

- **Comparable baselines.** If we averaged only over executed trades, “always skip” would have *no* trades, so E[R] would be undefined (or we’d need a special case). Using “per opportunity” gives every strategy the same denominator: one observation per IPO. Skip = 0 return for that opportunity; trade = actual net_ret. So **always_skip** has a well-defined E[R] (near 0) and the score is comparable across policies.
- **Portfolio interpretation.** Each IPO is one “period” in which we either invest or don’t. Our return for that period is either the trade’s net_ret or 0. So **mean over all opportunities = portfolio return** (as if we had one unit of capital and either deployed it or left it idle). That’s a standard way to score a strategy that chooses *when* to act.
- **Avoids rewarding “trade once, win once.”** If we averaged only over executed trades, a policy that traded 1 IPO and made 50% would show E[R]=0.50. That ignores that we passed on 833 others. Per-opportunity averaging penalizes being too selective (0 on skips) and rewards a good *balance* of participation and return.

So E[R] and the score are **per IPO opportunity** by design. We also report **E[R] per executed trade** (and number of trades) so you can see both views.
- **Rich CSV has only 4 bars per episode.** “Day0” vs “day1” is first-day vs in-week (~5 days); entry_day=0, hold_k=2 means exit at **in-month** (~21 days). So you’re not trading daily; you’re trading between sparse milestones, which limits how often and how much you can capture.
- **Position size is fractional.** `net_ret = weight * gross_ret − cost` with weight ≤ 1, so per-trade return is scaled down.
- **Conservative policies win.** The optimizer prefers high `participate_threshold` (trade only huge movers), which keeps CVaR and MDD at 0 but also reduces how many times you participate, so E[R] stays modest.

So **0.009 is “low” in absolute terms** because of: (1) averaging over 834 episodes with many skips, (2) sparse bars per episode, (3) fractional weights. In **relative** terms, the best policy (0.009) is still about **4× the always_skip baseline (0.002)** and much better than always participating (−0.23).

---

## 1. Score (objective)

**Formula:**  
`score = E[R] − λ·CVaR − κ·E[Cost] − μ·MDD`

- **What it is:** A single number the optimizer tries to **maximize**. It combines expected return with penalties for tail risk (CVaR), costs, and drawdown.
- **Your result (0.009067):** Positive means, on average, return outweighs the penalties. The best policy found is about **4× better** than the “do nothing” baseline (`always_skip` at 0.002290).

---

## 2. E[R] — Expected return

- **Definition:** Mean of **net return** over **all episodes** (one `net_ret` per episode; skips contribute 0).  
  So E[R] = (sum of net_ret when we traded) / (total number of episodes).
- **Units:** Decimal (e.g. 0.009069 ≈ 0.91% per episode).
- **Your result (0.009069):** Across 834 episodes, average net_ret is ~0.9%. Most episodes are skipped (0); the average is pulled up by the few where the policy traded and made a positive net return. Slightly higher than the score because the score subtracts a small cost term (CVaR and MDD are 0).
- **E[R] per executed trade** (also reported): Mean of `net_ret` over **only** episodes where the policy traded (weight > 0). So if you traded 50 times and made 10% on average on those trades, E[R]_per_trade ≈ 0.10. This is the “return per trade” view; the **score** and main **E[R]** stay per-opportunity so baselines (e.g. always_skip) remain comparable.
- **n_trades:** Number of episodes in which the policy actually traded (weight > 0).

---

## 3. CVaR — Conditional Value at Risk

- **Definition:** **Conditional Value at Risk** at level α (default α = 0.9).  
  It is the **expected loss in the worst 10% of outcomes** (average of the worst 10% of net returns, expressed as a positive “loss”).
- **Units:** Same as return (e.g. 0.01 = 1% expected tail loss).
- **Your result (0.000000):** In the backtest, the worst tail of returns didn’t produce losses on average (or the tail was empty), so CVaR is zero. So the best policy has **no tail-risk penalty** in the score.

---

## 4. E[Cost] — Expected cost

- **Definition:** Average **trading cost** per episode (e.g. transaction cost in basis points applied to turnover).
- **Units:** Decimal (e.g. 0.000002 ≈ 0.2 bps).
- **Your result (0.000002):** Very small cost per trade; the policy doesn’t trade often or in size enough to incur large costs.

---

## 5. MDD — Maximum drawdown

- **Definition:** **Maximum drawdown** of the **equity curve** over the backtest.  
  If equity goes from 100 → 110 → 95, the drawdown from the peak (110) to the trough (95) is (110−95)/110 ≈ 13.6%.
- **Units:** Decimal (e.g. 0.10 = 10% drawdown).
- **Your result (0.000000):** Equity never went below a previous peak in the backtest, so MDD = 0. So the best policy has **no drawdown penalty** in the score.

---

## 6. Best parameters (policy knobs)

| Parameter | Meaning | Your best value |
|-----------|--------|------------------|
| **participate_threshold** | Minimum \|price change\| (day0→day1) to participate. Only trade when \|return\| ≥ this. | **0.957** → participate only when first-day move is very large (~95.7%). Very selective. |
| **entry_day** | Day to enter (0 = IPO day, 1 = next day). | **0** → enter on IPO day. |
| **hold_k** | Number of days to hold after entry. | **2** → hold 2 days. |
| **w_max** | Maximum position size (cap on weight). | **0.975** → can use almost full allowed weight. |
| **raw_weight** | Base position size (before caps). | **0.45** → moderate size. |
| **use_volume_cap** | Whether to cap position by volume. | **True** → volume cap is on. |
| **vol_cap_mult** | Multiplier for volume-based cap. | **0.122** → cap is 12.2% of relevant volume. |

So the best policy is: **enter on IPO day only when the first-day move is huge, hold 2 days, use a moderate size with a volume cap.**

---

## 7. Baselines (comparison)

| Baseline | Meaning | Your result |
|----------|--------|-------------|
| **always_skip** | Policy with `participate_threshold=1.0`: trade only when \|price_change\| ≥ 100% (first day → in-week). So we almost never trade; the few IPOs with 100%+ move give the small positive E[R] (0.00229). | **0.002290** — benchmark “almost never trade.” |
| **always_participate** | Trade every IPO with fixed weight (e.g. 0.1). | **−0.228** — loses after risk/cost penalties. |
| **fixed_hold_1** | Participate every time, hold 1 day. | **−0.228** — same as always_participate. |
| **fixed_hold_5** | Participate every time, hold 5 days. | **−0.973** — much worse (more exposure, more drawdown). |

So your **best random-search policy (0.009)** beats **always_skip (0.002)** and is much better than naive “always participate” or long-hold strategies.

---

## 8. What the numbers indicate (short summary)

- **Best score 0.009067:** The optimizer found a **selective** policy (high `participate_threshold`) that trades only in a few, strong-moving IPOs, giving a small positive expected return with **zero** CVaR and MDD in this backtest and tiny cost.
- **CVaR = 0, MDD = 0:** Under this policy, the equity path didn’t show tail losses or drawdowns in the historical run; that’s why the score is close to E[R] minus a tiny cost.
- **Top 5 trials:** All have CVaR = 0 and MDD = 0; they differ mainly in how much they trade (E[R]) and thus in score. The best one has the highest E[R] among these.

If you want, we can add this as a “Results explained” section in `WEEK3_DELIVERABLE.md` or keep it only in `docs/RESULTS_GLOSSARY.md`.
