# Why 102,619 “company” IDs vs ~6,000 US IPOs?

## Short answer

**“Public Offering” in this dataset is not the same as “IPO.”**  
The file counts **all public offerings** (IPOs + follow-ons + secondaries + shelf takedowns, etc.), and the data likely covers **many issuers and possibly multiple countries**, not only “first-time US listings.”

---

## 1. Public Offering ≠ IPO

- **IPO** = **Initial** Public Offering = a company’s **first** listing on an exchange.  
  There are only about **~6,000** of those on US exchanges in the last 25 years.

- **Public Offering** in deal databases (e.g. S&P Capital IQ, Refinitiv, PitchBook) usually means **any** registered public sale of securities, including:
  - **IPOs** (first-time listings)
  - **Follow-on offerings** (same company selling more shares after it’s already public)
  - **Secondary offerings** (existing shareholders selling)
  - **Shelf takedowns**, at-the-market (ATM) offerings, etc.

So one company can have **one IPO** and then **many** “Public Offering” rows over time. In your data, one `companyid` has up to **16,080** “Public Offering” rows — that’s one issuer with many offerings, not 16,080 different companies.

---

## 2. What your numbers show

| What | Count |
|------|--------|
| Rows with `transactiontype == "Public Offering"` | 609,481 (these are **deals**, not companies) |
| Unique `companyid` among those rows | 102,619 |
| Other transaction type in the file | “Shelf Registration” (193,045 rows) |
| One company’s max number of Public Offering rows | 16,080 |
| Company_ids with only 1 Public Offering row | 53,453 |
| Company_ids with 2+ Public Offering rows | 49,166 |

So: **609K = number of public-offering *events*;** **102K = number of distinct *issuers* (companyid)** that had at least one such event.

---

## 3. Why 102K issuers is still much larger than ~6K US IPOs

Even if “Public Offering” were only IPOs, 102K would be too high for “US IPOs only.” So in practice:

1. **Scope is likely broader than “US IPO only”:**
   - Data may include **non-US exchanges** (e.g. LSE, HKEX, TSX, etc.), so 102K = issuers globally (or in a multi-country database) that did at least one “Public Offering.”
   - That’s consistent with vendor datasets that cover “all public offerings” across many countries.

2. **“Company” ID may be issuer/deal-entity, not “one legal company”:**
   - Different subsidiaries, SPVs, or deal entities might get different `companyid`s, so the count of “companies” can be larger than the count of unique “companies” in the everyday sense.

3. **Definition of “Public Offering” in the source:**
   - If the source includes follow-ons, secondaries, and other non-IPO public offerings, then 102K is the number of **distinct issuers that ever did any such offering**, not the number of companies that did an **IPO**.

---

## 4. Bottom line

- **~6,000** = number of **US IPOs** (first-time listings) in the last ~25 years.
- **102,619** = number of **unique `companyid`** in your file among rows labeled **“Public Offering”** in a dataset that:
  - Counts **all public offering events** (likely IPOs + follow-ons + secondaries + …), and
  - Likely includes **more than US-only IPOs** (e.g. global or multi-country), and/or
  - Uses an issuer/entity definition that can give multiple IDs per “company” in the colloquial sense.

So the gap is explained by: **(1) Public Offering ≠ IPO, (2) multiple offerings per company, (3) broader geographic/entity scope than “US IPO only.”**
