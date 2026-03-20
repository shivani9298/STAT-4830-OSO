<!-- 9a588c5d-fb37-4ebd-8a07-c3375b9b1483 -->
---
todos:
  - id: "fix-env"
    content: "Remove spaces after = in .env file"
    status: pending
  - id: "load-dotenv"
    content: "Add dotenv loading to tune_hyperparameters_wrds.py after ROOT definition"
    status: pending
  - id: "run-grid"
    content: "Run the grid search script"
    status: pending
isProject: false
---
# Load .env and Run Grid Search

## Changes

### 1. Fix `.env` formatting
The current `.env` has spaces after `=` which can cause leading-space issues:
```
WRDS_USERNAME= oceanazhu
WRDS_PASSWORD= Wharton1234oz!
```
Remove the spaces so values parse cleanly.

### 2. Add dotenv loading to `notebooks/tune_hyperparameters_wrds.py`
Add two lines near the top (after the `ROOT` definition) to load the `.env`:
```python
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
```
This sets `WRDS_USERNAME` and `WRDS_PASSWORD` in `os.environ` before `get_connection()` is called.

### 3. Run the grid search
Execute `python3 notebooks/tune_hyperparameters_wrds.py` with a 10-minute timeout. The search covers 288 configs tuning `lambda_turnover` and `lambda_path` alongside the existing hyperparameters. Results save to `results/ipo_optimizer_best_config.json`.
