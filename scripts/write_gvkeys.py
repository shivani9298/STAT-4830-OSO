"""Write space-separated codes from stdin to a txt file, one per line."""
import sys
from pathlib import Path

data = sys.stdin.read()
codes = data.split()
out = Path(__file__).resolve().parent.parent / "data" / "gvkeys.txt"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(codes) + "\n", encoding="utf-8")
print(f"Wrote {len(codes)} codes to {out}")
