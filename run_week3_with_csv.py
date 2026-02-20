"""Patch and run week3_implementation notebook using 2025iposdata.csv"""
import json
import subprocess
import sys

NOTEBOOK_PATH = "notebooks/week3_implementation.ipynb"
CSV_PATH = "2025iposdata.csv"

# New IPO cell source (load from CSV)
IPO_CELL_SOURCE = '''# ============================================================
# IPO DATA - Load from 2025iposdata.csv
# ============================================================

# Load IPO price data from CSV
ipo_csv = pd.read_csv('../2025iposdata.csv')
ipo_csv['datadate'] = pd.to_datetime(ipo_csv['datadate'])

# Extract IPO dates: first trading date per ticker = ipo_date
ipo_df = ipo_csv.groupby('tic').agg({'datadate': 'min'}).reset_index()
ipo_df.columns = ['ticker', 'ipo_date']
ipo_df = ipo_df.sort_values('ipo_date').reset_index(drop=True)

print(f"Total IPO tickers: {len(ipo_df)}")
print(f"\\nIPOs by year:")
print(ipo_df.groupby(ipo_df['ipo_date'].dt.year).size())
'''

# New price cell: load IPO prices from CSV, fetch SPY + shares from Yahoo
PRICE_CELL_SOURCE = '''# ============================================================
# LOAD PRICE DATA: IPO from 2025iposdata.csv, SPY from Yahoo
# ============================================================

ipo_csv = pd.read_csv('../2025iposdata.csv')
ipo_csv['datadate'] = pd.to_datetime(ipo_csv['datadate'])

# Pivot: datadate x tic, values=prccd (close price)
prices_ipo = ipo_csv.pivot_table(index='datadate', columns='tic', values='prccd')
prices_ipo.index = pd.to_datetime(prices_ipo.index).normalize()

# Fetch SPY
start_d = prices_ipo.index.min().strftime('%Y-%m-%d')
end_d = (prices_ipo.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
spy = yf.download('SPY', start=start_d, end=end_d, progress=False, auto_adjust=True)['Close']
if spy.index.tz:
    spy.index = spy.index.tz_localize(None)
spy = spy.reindex(prices_ipo.index).ffill().bfill()
prices = prices_ipo.copy()
prices['SPY'] = spy
prices = prices.dropna(subset=['SPY']).ffill().bfill()

# Fetch shares for IPO tickers
ipo_tickers = ipo_df['ticker'].tolist()
shares_outstanding = {}
for t in ipo_tickers:
    try:
        info = yf.Ticker(t).info
        s = info.get('sharesOutstanding', info.get('impliedSharesOutstanding'))
        if s:
            shares_outstanding[t] = s
        else:
            mc = info.get('marketCap')
            if mc and t in prices.columns:
                p = prices[t].dropna()
                if len(p) > 0 and p.iloc[-1] > 0:
                    shares_outstanding[t] = mc / p.iloc[-1]
    except Exception:
        pass

print(f"\\n=== Data Summary ===")
print(f"Price matrix shape: {prices.shape}")
print(f"Tickers with shares: {len(shares_outstanding)}")
print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
'''

def main():
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    # Find and replace cell 3 (IPO data)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if 'IPO_LIST = [' in src and "ipo_df = pd.DataFrame(IPO_LIST" in src:
            nb['cells'][i]['source'] = IPO_CELL_SOURCE.split('\n')
            nb['cells'][i]['source'] = [line + '\n' for line in IPO_CELL_SOURCE.split('\n')]
            if nb['cells'][i]['source'][-1].endswith('\n\n'):
                nb['cells'][i]['source'][-1] = nb['cells'][i]['source'][-1].rstrip('\n')
            break

    # Find and replace cell 4 (price fetch)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if 'fetch_price_and_shares_data' in src and 'FETCH PRICE' in src:
            lines = PRICE_CELL_SOURCE.split('\n')
            nb['cells'][i]['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
            break

    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(nb, f, indent=1)

    print("Patched notebook. Running...")
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        ep = ExecutePreprocessor(timeout=600)
        with open(NOTEBOOK_PATH) as f:
            nb_exec = nbformat.read(f, as_version=4)
        ep.preprocess(nb_exec, {'metadata': {'path': 'notebooks/'}})
        with open(NOTEBOOK_PATH, 'w') as f:
            nbformat.write(nb_exec, f)
        print("Notebook executed successfully.")
    except ImportError:
        result = subprocess.run(
            [sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
             '--inplace', NOTEBOOK_PATH],
            cwd='.',
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("Jupyter/nbconvert not found. Notebook was patched.")
            print("Run it manually: open notebooks/week3_implementation.ipynb and Run All")
            print("STDERR:", result.stderr)
            sys.exit(1)
        print("Notebook executed successfully.")

if __name__ == '__main__':
    main()
