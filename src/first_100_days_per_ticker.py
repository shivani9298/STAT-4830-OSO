"""
Keep only the first 100 trading days per ticker in dailyhistorical_21-26.csv.
For each ticker, rows are sorted by datadate; only the first 100 days are kept.
Overwrites the file in place (writes to temp then renames).
"""
import csv
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "dailyhistorical_21-26.csv"
DAYS_PER_TICKER = 100


def main():
    if not CSV_PATH.exists():
        print(f"Not found: {CSV_PATH}")
        return

    # Group rows by ticker
    by_tic = defaultdict(list)
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            by_tic[row["tic"]].append(row)

    # Sort by datadate and take first 100 per ticker
    kept = []
    for tic, rows in by_tic.items():
        rows_sorted = sorted(rows, key=lambda r: r["datadate"])
        kept.extend(rows_sorted[:DAYS_PER_TICKER])

    # Sort all kept rows by ticker then date for a clean output
    kept.sort(key=lambda r: (r["tic"], r["datadate"]))

    tmp_path = CSV_PATH.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    tmp_path.replace(CSV_PATH)
    print(f"Kept first {DAYS_PER_TICKER} trading days per ticker: {len(kept)} rows from {len(by_tic)} tickers.")
    print(f"Original rows: {sum(len(rows) for rows in by_tic.values())}")


if __name__ == "__main__":
    main()
