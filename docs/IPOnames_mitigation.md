# Mitigating the IPOnames vs “true” IPO count issue

## Options (in order of practicality)

### 1. **Use “first public offering” per company (best you can do inside IPOnames)**

IPOnames does **not** have a column that says “IPO” vs “follow-on.” You can still get a **clean, one-row-per-company** list by keeping only each company’s **first** Public Offering (earliest `announced` date).

- **Meaning:** For each `companyid`, keep the row with the smallest `announced` date. That row is a **proxy** for “first time this issuer did a public offering” (often an IPO, but not guaranteed).
- **Result:** One row per company, so no duplicate company IDs. You still have ~102K companies (global/multi-country), but the table is deduplicated and easier to work with.
- **How:** Run the script below to create `IPOnames_first_offering.csv` and `company_ids_first_offering.txt`.

This does **not** get you to ~6K; it only cleans the file to “one company, one row (first offering).”

---

### 2. **Use archive-3 for “true” US IPOs (recommended for ~6K-style lists)**

For **US IPO** counts and company lists that align with “~6,000 IPOs on US exchanges,” **don’t rely on IPOnames**. Use your IPO-focused files instead:

- **`archive-3/ipo_clean_2010_2018.csv`** — ~1,600 rows (IPO-oriented, 2010–2018).
- **`archive-3/ipo_stock_2010_2018_v2.csv`** — ~836 rows (IPO + price summary).

These are built for IPO analysis (e.g. Symbol, Date Priced, Market) and are a much better match to “US IPOs” than IPOnames.

- **When to use IPOnames:** When you need **companyid**, deal-level history (many offerings per company), or a broader “all public offerings” universe.
- **When to use archive-3:** When you need **US IPO** counts, tickers, and dates (e.g. for your stat 4830 project).

---

### 3. **Restrict by geography/exchange (only if your data has it)**

If you ever get a version of IPOnames (or a linked table) with **exchange** (e.g. NASDAQ, NYSE) or **country**, filter to **US exchanges only**. That would cut the 102K down toward a US-only set. Current IPOnames has no such column, so this is for future data.

---

### 4. **Cross-reference with a known IPO list (if you have a link)**

If you have a table that maps **companyid** (or issuer name) to **ticker** or to an “IPO yes/no” flag, you can:

- Keep only companyids that appear in that IPO list, or  
- Flag “first offering” rows that match an IPO date.

That would give you an “IPO-only” subset. Right now there’s no direct join key between IPOnames (`companyid`, `issuingcompany`) and archive-3 (Symbol, Company Name), so this would require an extra mapping file or fuzzy matching by name.

---

## Summary

| Goal | Use |
|------|-----|
| One row per company, first offering only (still ~102K companies) | `IPOnames_first_offering.csv` + `company_ids_first_offering.txt` (from script below) |
| US IPO count / list (~6K-style) | `archive-3/ipo_clean_2010_2018.csv` or `ipo_stock_2010_2018_v2.csv` |
| Deal-level history, companyid, all public offerings | `IPOnames.csv` (with “Public Offering” filter as you did) |

Running the script creates the first-offering subset and the corresponding company IDs file.
