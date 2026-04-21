# Self-Critique (Week 6)
*Date: February 20, 2026*

*Max 1 page. OODA: Observe → Orient → Decide → Act.*

---

## OBSERVE

- Read the report as a first-time reader: the problem (policy over participate/entry/hold/size) and objective (E[R] − λ·CVaR − κ·Cost − μ·MDD) are clear, but the **report’s headline formula** includes β·Sharpe while the **code does not**—so there is a small documentation inconsistency to fix or explicitly justify.
- Re-running the code: `test_basic.py` passes. **Data**: The pipeline *is* wired to real data: `run_pytorch.py --data yfinance` fetches S&P 500 tickers and prices from Yahoo Finance and trains the policy. The **notebook** uses synthetic data by default for speed/reproducibility. So “only synthetic” applies to the notebook and to the default script run (`--data synth`), not to the project as a whole.
- First reactions: Strengths are the separation of backtest/objective and the REINFORCE setup, plus real-data support via yfinance. Weak spots: the report/notebook don’t yet show a concrete yfinance run or its output; validation is random train/val (no time-based split).

---

## ORIENT

**Strengths (max 3)**

- **Clear problem and objective**: Policy and score (E[R] − λ·CVaR − κ·Cost − μ·MDD) are stated precisely; reader can see what is being optimized and why.
- **Testable design**: Backtest and objective are separate from the policy; `test_basic.py` and the notebook validate edge cases (empty, never participate, CVaR/MDD).
- **Honest limitations**: Report and notebook state limitations (e.g. high validation variance, no walk-forward yet); next steps are listed. Real data is supported in `run_pytorch.py` (yfinance / path) but not demonstrated in the notebook.

**Areas for improvement (max 3)**

- **Align report formula with code**: Either add a Sharpe term in `objective.score()` or state once in the report that “we implement the reduced form without β·Sharpe” and remove/minimize the Sharpe term in the main formulation to avoid confusion.
- **Validation rigor**: No walk-forward or time-based split yet; only random train/val. Add one concrete step (e.g., split by IPO year or “last 20% by date”) in the notebook or a script and report OOS score.
- **Show one real-data run**: The pipeline already uses real data via `run_pytorch.py --data yfinance` (yfinance). Improve by documenting that command in the report/notebook and pasting example output (train/val score) so “works on real data” is clearly demonstrated.

**Critical risks / assumptions (2–3 sentences)**

- We assume the current synthetic episode format (e.g., `Episode.df` with `close`) matches what we will use for real data; if the rich CSV or index pipeline produces different columns or date alignment, the pipeline could break and needs a quick smoke test. We also assume λ, κ, μ = 1.0 are acceptable defaults; they are not tuned, so results are sensitive to these choices.

---

## DECIDE

**Concrete next actions (max 3; achievable within a week)**

1. **Report/code consistency**: In `report.md`, add one sentence after the main formula: “The current implementation uses the form without β·Sharpe (Score = E[R] − λ·CVaR − κ·E[Cost] − μ·MDD); Sharpe can be added later.” Optionally remove or de-emphasize β·Sharpe in the single “max” equation so the implemented objective is the default.
2. **One time-based validation**: In the notebook or a small script, split episodes by time (e.g., by `ipo_date`: train on earliest 80%, test on latest 20%), run backtest + score on the test set, and report the out-of-sample score in the notebook or report.
3. **Document a real-data run**: Add to the report or notebook the exact command for a yfinance run (e.g. `run_pytorch.py --data yfinance --max_tickers 50 --n_epochs 20`) and paste example output (train/val score) so readers see the pipeline works on live data from Yahoo Finance.

---

## ACT

**Resource needs (2–3 sentences)**

- Real data is already available via yfinance in `run_pytorch.py`; no extra dataset is required. The main need is to run once (e.g. `--data yfinance --max_tickers 50`), capture the output, and add it to the report or notebook. For walk-forward, no new libraries are needed—only a time-based split (e.g. by `ipo_date`) in the notebook or script. If you prefer to use your specific IPO list (e.g. from dailyhistorical_21-26.csv) instead of S&P 500, that would mean building a meta CSV from that list and using `--data path` with prices from yfinance or the index builder.
