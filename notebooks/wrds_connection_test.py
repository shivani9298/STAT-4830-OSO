# ==========================================
# WRDS Connection Script for Cursor
# ==========================================

# 1) Install package first (run in terminal if needed):
# pip install wrds
#
# 2) For non-interactive runs, set env vars in same terminal:
#    $env:WRDS_USERNAME = "your_username"
#    $env:WRDS_PASSWORD = "your_password"

import os
import wrds

def main():
    # Use env vars if set (for non-interactive runs); otherwise prompts
    uname = os.environ.get("WRDS_USERNAME")
    pwd = os.environ.get("WRDS_PASSWORD")
    if uname and pwd:
        db = wrds.Connection(wrds_username=uname, wrds_password=pwd)
    elif uname:
        # Username only: .pgpass provides password (set up via wrds.Connection() first run)
        db = wrds.Connection(wrds_username=uname)
    else:
        # Prompts for username/password, may ask to create .pgpass
        db = wrds.Connection()

    print("Connected to WRDS")

    # Simple test query (CRSP daily returns sample)
    query = """
        SELECT date, permno, ret
        FROM crsp.dsf
        WHERE date >= '2025-01-01'
        LIMIT 5
    """

    df = db.raw_sql(query)

    print("\nSample data:")
    print(df.head())

    db.close()
    print("\nConnection closed")

if __name__ == "__main__":
    main()
