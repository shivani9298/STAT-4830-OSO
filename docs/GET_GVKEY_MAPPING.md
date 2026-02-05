# Getting a Ticker → GVKEY mapping for ritter_company_codes_after_2000.txt

**GVKEY** is Compustat’s (S&P Global) company identifier. It does **not** appear in the Ritter IPO CSV, so you need a separate mapping from ticker to GVKEY.

**Current file:** `ritter_company_codes_after_2000.txt` is built from Ritter using **CRSP.Perm** (CRSP permanent ID), not GVKEY, because Ritter has no GVKEY column. To get actual GVKEY codes, use one of the options below.

## Option 1: WRDS (if you have access)

Many universities subscribe to **WRDS** (Wharton Research Data Services). There you can get ticker–GVKEY from Compustat.

1. Log in at [wrds-www.wharton.upenn.edu](https://wrds-www.wharton.upenn.edu).
2. Go to **Compustat** → **North America** → **Company** (or **Security** / **CCM**).
3. Export a table that has **gvkey** and **tic** (ticker), e.g.:
   - **comp.company**: `gvkey`, `tic`, `conm`
   - Or the **CCM (CRSP–Compustat link)** table: `gvkey`, `lpermno`, and you can join to get ticker.
4. Save as CSV with columns **ticker** (or **tic**) and **gvkey**.
5. Save the file in the project root as **`ticker_gvkey_mapping.csv`** (or pass its path to the script).
6. Run:
   ```bash
   python3 scripts/tickers_to_gvkey.py [path/to/ticker_gvkey_mapping.csv]
   ```
   Output: **`ritter_company_codes_after_2000.txt`** with one GVKEY per line (same order as `ritter_tickers_after_2000.txt`).

## Option 2: Other data providers

If you have **S&P Capital IQ**, **Refinitiv**, or another vendor that provides GVKEY (or a Compustat feed), export a table with **ticker** and **gvkey** and use the same CSV format and script as above.

## Mapping CSV format

Header must include **ticker** (or **tic**) and **gvkey**:

```csv
ticker,gvkey
AAPL,001690
MSFT,001798
```

Or:

```csv
tic,gvkey
AAPL,001690
MSFT,001798
```

## Converting GVKEY back to ticker

If you have a list of GVKEY codes and a mapping CSV (gvkey, ticker), you can convert GVKEYs to tickers:

```bash
python3 scripts/gvkey_to_ticker.py <gvkey_list.txt> <gvkey_ticker_mapping.csv> [output.txt]
```

- **gvkey_list.txt**: one GVKEY per line.
- **gvkey_ticker_mapping.csv**: CSV with columns **gvkey** and **ticker** (or **tic**).
- **output.txt**: optional; default is `<gvkey_list_stem>_tickers.txt`.

Example:
```bash
python3 scripts/gvkey_to_ticker.py ritter_company_codes_after_2000.txt gvkey_ticker_mapping.csv ritter_tickers_from_gvkey.txt
```

---

## Script usage (ticker → GVKEY)

From the project root:

```bash
# Use default path: ticker_gvkey_mapping.csv
python3 scripts/tickers_to_gvkey.py

# Or specify the mapping file
python3 scripts/tickers_to_gvkey.py /path/to/your_ticker_gvkey.csv
```

Output: **`ritter_company_codes_after_2000.txt`** — one GVKEY per line, in the same order as **`ritter_tickers_after_2000.txt`**. Lines with no mapping are left empty.
