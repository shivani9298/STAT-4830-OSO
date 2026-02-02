"""
Convert CUSIP list to GVKEYs using a mapping file.

GVKEY mapping must come from WRDS/Compustat. Steps:
1. In WRDS: use Linking Suite → "COMPUSTAT GVKEY by CUSIP", or run a query
   on Compustat Security (e.g. comp.security) to get cusip + gvkey.
2. Export to CSV with columns named 'cusip' and 'gvkey' (or 'CUSIP' and 'GVKEY').
3. Save as data/cusip_gvkey_mapping.csv, then run: python3 src/cusip_to_gvkey.py
"""
import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CUSIP_LIST = DATA_DIR / "ipo_cusips_2000_onward.txt"
MAPPING_CSV = DATA_DIR / "cusip_gvkey_mapping.csv"
OUTPUT_TXT = DATA_DIR / "ipo_gvkeys_2000_onward.txt"


def load_mapping(path: Path) -> dict:
    """Build cusip -> gvkey from CSV. Supports 8- and 9-digit CUSIP."""
    cusip_to_gvkey = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return cusip_to_gvkey
        cols = [c.lower() for c in (reader.fieldnames or [])]
        cusip_col = next((c for c in reader.fieldnames if "cusip" in c.lower()), None)
        gvkey_col = next((c for c in reader.fieldnames if "gvkey" in c.lower()), None)
        if not cusip_col or not gvkey_col:
            raise ValueError(
                f"Mapping CSV must have 'cusip' and 'gvkey' columns. Found: {reader.fieldnames}"
            )
        for row in reader:
            cusip = (row.get(cusip_col) or "").strip()
            gvkey = (row.get(gvkey_col) or "").strip()
            if not cusip or not gvkey:
                continue
            cusip_to_gvkey[cusip] = gvkey
            if len(cusip) == 9:
                cusip_to_gvkey[cusip[:8]] = gvkey
            elif len(cusip) == 8:
                cusip_to_gvkey[cusip + " "] = gvkey
    return cusip_to_gvkey


def main():
    cusips = [
        line.strip()
        for line in CUSIP_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cusips:
        print(f"No CUSIPs in {CUSIP_LIST}")
        return

    if not MAPPING_CSV.exists():
        MAPPING_CSV.parent.mkdir(parents=True, exist_ok=True)
        MAPPING_CSV.write_text("cusip,gvkey\n", encoding="utf-8")
        print(
            f"Created {MAPPING_CSV} with header. Add CUSIP→GVKEY rows from WRDS Compustat, then run again."
        )
        print(
            "WRDS: Linking Suite → COMPUSTAT GVKEY by CUSIP, or query comp.security for cusip, gvkey."
        )
        return

    mapping = load_mapping(MAPPING_CSV)
    gvkeys = []
    for c in cusips:
        gvkey = mapping.get(c) or (mapping.get(c[:8]) if len(c) >= 8 else None)
        if gvkey:
            gvkeys.append(gvkey)

    gvkeys_dedup = sorted(dict.fromkeys(gvkeys))
    OUTPUT_TXT.write_text("\n".join(gvkeys_dedup) + "\n", encoding="utf-8")
    print(f"Wrote {len(gvkeys_dedup)} GVKEYs to {OUTPUT_TXT}")
    matched = len(gvkeys)
    if matched < len(cusips):
        print(f"Note: {len(cusips) - matched} CUSIPs had no GVKEY in mapping.")


if __name__ == "__main__":
    main()
