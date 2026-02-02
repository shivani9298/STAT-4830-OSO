"""
Merge multiple WRDS/Wharton CSV files into one CSV with a unified schema.
Uses union of all columns; missing values left empty.
"""
import csv
from pathlib import Path

# Input files (user's Downloads)
DOWNLOADS = Path.home() / "Downloads"
INPUTS = [
    DOWNLOADS / "Data from Wharton (1).csv",
    DOWNLOADS / "Data from Wharton (2).csv",
    DOWNLOADS / "Data from WRDS (1).csv",
    DOWNLOADS / "Data from WRDS (2).csv",
    DOWNLOADS / "LLVJUMK5ZQZQUR5U (1).csv",
]

# Output in project data folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_CSV = DATA_DIR / "dailyhistorical_21-26.csv"


def main():
    all_rows = []
    all_columns = []
    seen_columns = set()

    for path in INPUTS:
        if not path.exists():
            print(f"Skip (not found): {path}")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            for c in cols:
                if c not in seen_columns:
                    seen_columns.add(c)
                    all_columns.append(c)
            for row in reader:
                all_rows.append(row)

    if not all_columns:
        print("No columns found.")
        return

    # Write merged CSV with union of columns (stable order: keep first-seen order)
    # Re-collect column order from first file that has most columns, then union
    all_columns = []
    for path in INPUTS:
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for c in (reader.fieldnames or []):
                if c not in all_columns:
                    all_columns.append(c)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            # Ensure row has all keys; missing -> empty
            out = {k: row.get(k, "") for k in all_columns}
            writer.writerow(out)

    print(f"Merged {len(all_rows)} rows into {OUTPUT_CSV}")
    print(f"Columns: {all_columns}")


if __name__ == "__main__":
    main()
