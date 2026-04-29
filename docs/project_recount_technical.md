# IPO Optimization Project: Technical Recount (Readable Version)

## What This Document Is

This is the story of how our project actually evolved, across:
- `main`
- `origin/shivani`
- `origin/olivia`
- `origin/oceana`

Instead of just listing commits, this explains:
- what we were trying to optimize,
- what we changed and why,
- what helped,
- what did not help,
- what made things worse,
- and what we learned from all of it.

---

## The Core Problem (In Plain English)

Every day, our model chooses how much to put in:
1. a broad market sleeve, and
2. an IPO sleeve (later split into sector IPO sleeves).

Rules:
- no shorting,
- fully invested,
- weights must add up to 100%.

So the real question was never just "how do we maximize return?" It was:
"How do we get strong return **without** blowing up risk, concentration, and instability?"

That is why our loss/objective eventually had many parts:
- return term,
- tail-risk term (CVaR style),
- volatility term,
- turnover/path smoothness terms,
- diversification term,
- benchmark-relative terms,
- and later log-growth style terms.

Big lesson up front: **objective design mattered as much as (or more than) model architecture.**

---

## Timeline: What Changed Over Time

## Phase 1 - Build a working baseline

What we did:
- Built WRDS-based data pipeline and IPO index construction.
- Connected data to recurrent model training.
- Got the system running end-to-end.

Why this mattered:
- This gave us a working optimizer and repeatable experiments.

What we noticed:
- "It trains" did not mean "it is economically good."
- Early objective and evaluation were not perfectly aligned.

---

## Phase 2 - Concentration problem and objective shaping

What went wrong:
- In some settings, the model heavily concentrated into IPO exposure.
- This looked good for some in-sample return windows, but it was not robust.

What we changed:
- Added/retuned volatility and CVaR penalties.
- Added/retuned diversification controls.
- Added/retuned path and turnover controls.

Result:
- Better stability overall.
- But sometimes we over-corrected and got very static allocations (too little adaptation).

Interpretation:
- We learned that regularization is necessary, but too much regularization kills useful signal.

---

## Phase 3 - Model architecture experiments

What we tried:
- GRU baseline,
- LSTM variants (especially in `shivani`/`olivia` lines),
- Transformer variants (especially in `oceana`, then merged into `main`).

What we expected:
- More expressive models might adapt better to changing IPO regimes.

What we observed:
- GRU stayed a strong practical baseline.
- LSTM gave mixed outcomes (some improvements, not a universal win).
- Transformer added complexity and runtime without consistent enough gains in this regime.

Interpretation:
- "More complex model" was not a guaranteed upgrade.

---

## Phase 4 - Sector and multisector expansion

What we changed:
- Moved from one IPO sleeve to sector-aware sleeves / multisector variants.

What improved:
- Much better diagnostic visibility.
- We could see which sectors were helping vs hurting instead of hiding everything in one aggregate line.

What became harder:
- Sector label quality became critical.
- Missing/stale sector metadata could directly distort model behavior.

Interpretation:
- Better structure gave better insight, but also introduced a new data-quality dependency.

---

## Phase 5 - Evaluation got more serious (major turning point)

Early on:
- We leaned heavily on loss curves and single-split validation outcomes.

Later upgrades:
- Holdout-aware splitting,
- rolling tail/excess diagnostics,
- path-aware metrics (Sharpe, Sortino, drawdown, compound return),
- walk-forward tooling.

Why this is a big deal:
- It reduced the risk of celebrating models that only looked good under one narrow metric.
- It made our claims more defensible and more honest.

---

## Phase 6 - Most recent work (what to emphasize in presentation)

Recent improvements include:
- gradient clipping and training-stability refinements,
- direct GRU vs LSTM comparison artifacts,
- better artifact/report organization,
- clearer, more honest self-critique around failures and regressions.

Presentation message:
- Recent progress is strongest in **methodological quality** (how we evaluate and compare), not only in "raw score chasing."

---

## Requested Coverage Checklist (Explicitly Added)

This section directly answers whether each requested item is covered.

## 1) Update loss function

Yes, covered, but restated here more clearly.

Over time, the loss moved from a simpler return-plus-risk objective to a more complete one with:
- return terms,
- tail-risk control (CVaR-style),
- volatility controls,
- turnover/path penalties,
- diversification controls,
- benchmark-relative terms,
- and optional log-growth style terms.

Why this matters:
- This was the single biggest technical lever in the project.
- Many performance changes were caused by loss calibration, not by model architecture alone.

## 2) Varying hyperparameters / batch size / peripheral training changes

Yes, this was part of the iterative process and should be explicitly presented.

What we changed repeatedly:
- learning rate choices/schedules,
- batch sizes,
- penalty weights,
- context length,
- gradient clipping and training-stability settings,
- peripheral training controls (early stopping behavior, checkpoint criteria, etc.).

What we learned:
- these "small" settings often changed behavior as much as big architecture swaps.

## 3) Varying context window, walk-forward, training up to the day

Yes, included now explicitly.

We tried different context windows (short and long history windows) and observed mostly incremental effects unless coupled with objective changes.

Walk-forward and temporal split tooling was added to reduce false confidence from one static split.

"Training up to the day" / online-style evaluation should be framed as:
- train on history available up to a point,
- evaluate on future data only,
- optionally update and compare against no-update baselines.

This is exactly the direction of the newer online-evaluation artifacts.

## 4) Varying model architecture (including 2 that do not work)

Yes, and should be presented bluntly.

Architecture experiments included:
- GRU (working baseline),
- LSTM (mixed but useful comparator),
- Transformer (often underwhelming relative to complexity),
- additional policy/alternative variants in some branches.

At least two non-dominant or non-working directions should be called out transparently:
- transformer did not consistently justify cost in this regime,
- some alternative policy formulations underperformed materially.

## 5) Slide per model (Transformer and LSTM)

This should be explicitly in the presentation structure:
- one slide focused on LSTM behavior vs baseline,
- one slide focused on transformer behavior vs baseline,
- each with one clear takeaway and honest conclusion (not just showing curves).

## 6) Flow chart showing current working architecture and how pieces fit together

Included conceptually before, but now made explicit as a required deliverable:
- data ingestion (WRDS/SDC/CRSP) ->
- feature windows ->
- model encoder (current working model) ->
- allocation head(s) ->
- loss components ->
- evaluation and reporting.

Use this to answer "how does the system actually work?" in one visual.

## 7) Update results/graph

Yes, must be included as an explicit recount item:
- refresh all headline plots/tables to current branch state,
- avoid mixing outdated figures with newer code behavior,
- attach one-sentence takeaway under each updated graph.

## 8) Slide for commodities / cross-asset behavior

Partially covered before; now explicit.

You should include one slide that shows how the framework behaves across different asset-class settings (or proxy sleeves), and make clear where IPO-specific behavior causes problems.

Suggested message:
- framework may generalize structurally,
- but IPO sleeve dynamics (tail behavior/regime shifts/metadata noise) remain the hardest part.

## 9) Literature review

This was under-emphasized before and is now explicitly included.

Add a short literature-review section/slides with:
- online portfolio optimization foundations,
- IPO return and post-IPO anomaly literature,
- risk-aware portfolio optimization (CVaR/drawdown-aware objectives),
- sequence-model references for time-series allocation.

Why this matters:
- It shows your design choices were informed, not arbitrary.
- It improves technical depth and Q&A credibility.

---

## What Actually Worked

- Risk/diversification shaping that prevented concentration collapse.
- Gradient clipping for more stable optimization.
- Sector-aware decomposition for deeper diagnosis.
- Path-aware model-selection/evaluation tools.

In short:
- The system became more trustworthy and interpretable.

---

## What Was Mixed or Limited

- Context-window tuning alone gave mostly incremental gains.
- Transformer complexity often did not pay off enough for this dataset/setup.

In short:
- Some ideas were technically interesting but not practically dominant.

---

## What Worsened Outcomes

- Objective/selection mismatch in some stages:
  - loss improved,
  - but realized path quality did not improve proportionally.
- Some multisector or long-horizon setups under weak metadata/regime mismatch hurt risk-adjusted performance.

In short:
- Better optimization numbers can still produce worse portfolio behavior if the target metric is misaligned.

---

## How Evaluation Evolved (Key Story Arc)

1. Loss-first optimization/selection.
2. Benchmark-relative reporting (market-only, IPO-only, equal-weight).
3. Tail/excess diagnostics.
4. Path-aware and walk-forward framing.

This is the most important maturity arc to highlight in your presentation.

If someone asks, "What is your biggest technical improvement?" this is the answer.

---

## Remaining Risks (Be Explicit in Presentation)

- Possible overfitting to validation through repeated tuning.
- IPO regime dependence (historical relationships may shift).
- Sector-label quality and survivorship issues.
- Transaction costs/slippage not fully integrated into headline objective metrics.

---

## Final Takeaway

Our biggest contribution is not "we found one magic model."

Our biggest contribution is that we built a more rigorous optimization workflow:
- better objective shaping,
- better checkpoint criteria,
- better evaluation design,
- and more honest reporting of failures.

That is why the project is technically stronger now than it was early on.

