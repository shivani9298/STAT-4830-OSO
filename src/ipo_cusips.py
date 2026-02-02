"""
Filter Ritter IPOs to 2000 onward and output CUSIPs (one per line).
Ritter file already has CUSIP; GVKEY would require Compustat/WRDS.
"""
import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RITTER_CSV = DATA_DIR / "ritter_ipos_1975_2025.csv"
OUTPUT_TXT = DATA_DIR / "ipo_cusips_2000_onward.txt"


def main():
    with open(RITTER_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Column names may have quotes stripped by csv
    col_date = "offer.date"
    col_cusip = "CUSIP"
    if rows and col_date not in rows[0]:
        # try without dot
        col_date = [c for c in rows[0] if "offer" in c and "date" in c.lower()]
        col_date = col_date[0] if col_date else "offer.date"
    if rows and col_cusip not in rows[0]:
        col_cusip = "CUSIP"

    seen = set()
    for row in rows:
        offer_date = row.get(col_date, "")
        cusip = (row.get(col_cusip) or "").strip()
        try:
            date_int = int(offer_date)
        except (ValueError, TypeError):
            continue
        if date_int < 20000101:
            continue
        if not cusip or cusip in (".", "NA", "nan", "None"):
            continue
        seen.add(cusip)

    unique_cusips = sorted(seen)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TXT.write_text("\n".join(unique_cusips) + "\n", encoding="utf-8")
    print(f"Wrote {len(unique_cusips)} CUSIPs to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
