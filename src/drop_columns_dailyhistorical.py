"""
Remove company name, EPS, epsmo, and dividend columns from dailyhistorical_21-26.csv.
Overwrites the file in place (writes to temp then renames).
"""
import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "dailyhistorical_21-26.csv"
DROP_COLS = {"conm", "divd", "eps", "epsmo", "div", "trfd"}


def main():
    if not CSV_PATH.exists():
        print(f"Not found: {CSV_PATH}")
        return

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    keep_cols = [c for c in fieldnames if c not in DROP_COLS]
    if len(keep_cols) == len(fieldnames):
        print("No columns to drop (already removed?).")
        return

    tmp_path = CSV_PATH.with_suffix(".csv.tmp")
    with open(CSV_PATH, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        with open(tmp_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=keep_cols, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                writer.writerow({k: row.get(k, "") for k in keep_cols})

    tmp_path.replace(CSV_PATH)
    print(f"Removed {list(DROP_COLS)} from {CSV_PATH.name}. Columns now: {keep_cols}")


if __name__ == "__main__":
    main()
