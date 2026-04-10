# Sector assignment workflow (CCM / GICS)

This project can assign **GICS-style sector labels** to IPO names for multi-head training. There are three sources, controlled by **`IPO_SECTOR_SOURCE`**:

| Value | Mechanism | Cache file |
|--------|------------|------------|
| `compustat` (default) | `comp.funda` → `comp.company` GICS via ticker (legacy) | `results/ticker_sector_cache_compustat.csv` |
| `yfinance` | Yahoo `info['sector']` | `results/ticker_sector_cache_yfinance.csv` |
| `ccm` / `wrds_chain` | Date-valid chain: **ticker + IPO date** → `crsp.stocknames` (then `crsp.dsenames`) → **`match_date`** → `crsp.ccmxpf_linktable` → `comp.company` GICS | `results/ticker_sector_cache_ccm.csv` |

## Recommended path: `ccm`

The CCM pipeline aligns identifiers the way empirical finance typically does:

1. **`match_date = max(ipo_date, first_crsp_date)`** — `first_crsp_date` is the first day CRSP has a price for that ticker in your panel, so name history is evaluated when CRSP actually lists the security.
2. **`crsp.stocknames`** first; if no date-valid row, **`crsp.dsenames`** (DSF name file, consistent with daily returns).
3. **CCM** `permno` → `gvkey` with `linkprim='P'` and `linktype ∈ {LC, LU, LS}` on **`match_date`**.
4. **`comp.company`** for `gsector` → mapped to display labels via `gics_sector_name_from_code`.

Code: `src/wrds_ipo_gics_enrichment.py`, `fetch_ticker_sectors_ccm_chain` in `src/sector_ipo.py`, wiring in `run_ipo_optimizer_wrds.prepare_data`.

## Generating the CCM cache only (no training)

From the repo root (WRDS credentials in `.env` or environment):

```bash
python scripts/generate_sector_cache_ccm.py
```

Defaults: **IPO offer dates and price panel** restricted to **2010-01-01 through 2024-12-31**. Only IPOs whose **`ipo_date`** falls in that window are labeled (same window as `START_DATE` / `END_DATE` in `run_ipo_optimizer_wrds.py`).

Override window:

```bash
python scripts/generate_sector_cache_ccm.py --start 2015-01-01 --end 2020-12-31
```

Force a full re-query (ignore stale rows in an old cache file):

```bash
python scripts/generate_sector_cache_ccm.py --refresh
```

## Using the cache in the optimizer

```bash
set IPO_SECTOR_SOURCE=ccm
python run_ipo_optimizer_wrds.py
```

(PowerShell: `$env:IPO_SECTOR_SOURCE = "ccm"`)

The optimizer merges **`first_crsp_date`** from the CRSP price panel when calling the CCM sector function (see `prepare_data`).

## Diagnostics

Full enrich + flags (optional):

```bash
python scripts/run_enrich_ipo_gics_diagnostics.py
```

Outputs: `results/ipo_enrich_diagnostics_sample.csv`.
