"""
WRDS IPO → GICS enrichment via CRSP permno and Compustat gvkey (date-valid chain).

================================================================================
1. CONCEPTUAL EXPLANATION (why this workflow)
================================================================================

**Why ticker alone is not a stable identifier**
    Tickers are reused, recycled, and can refer to different securities over time.
    The same string may map to different CRSP ``permno`` values across eras, and
    Compustat ``tic`` is not guaranteed to align with CRSP naming on a given calendar
    day. Matching IPOs on raw ticker without a **date context** invites wrong-firm links.

**Why permno and gvkey are the correct WRDS identifiers**
    ``permno`` is CRSP’s permanent **security** identifier (within CRSP stock data).
    ``gvkey`` is Compustat’s permanent **company** identifier. The CRSP–Compustat
    merged (CCM) **link table** is the supported bridge between them, with explicit
    validity intervals and link-quality flags used in academic research.

**Why date-valid joins matter for IPO matching**
    An IPO observation is defined by ``(ticker, ipo_date)``. The economically correct
    security identity is the one whose **CRSP name history** covers ``ipo_date``, and
    the correct Compustat link is the CCM row whose **[linkdt, linkenddt]** covers
    ``ipo_date``. Using a name or link that is off by months or years assigns the
    wrong firm.

**Why some IPOs still legitimately lack GICS in Compustat**
    Coverage lags, foreign issuers, SPACs/shells, ADRs, non-operating entities, or
    firms not yet in ``comp.company`` with populated GICS fields can yield missing
    ``gsector``/``ggroup``/``gind``/``gsubind`` even when ``gvkey`` exists.

================================================================================
2. ASSUMPTIONS ABOUT WRDS TABLES (verify on your subscription)
================================================================================

PostgreSQL hosts on WRDS typically expose (lowercase identifiers):

* ``crsp.stocknames`` — historical ``permno``–``ticker`` intervals ``namedt``–``nameenddt``.
* ``crsp.ccmxpf_linktable`` — CCM link: ``lpermno``, ``gvkey``, ``linkdt``,
  ``linkenddt``, ``linktype``, ``linkprim`` (names may vary slightly; see
  ``list_wrds_tables`` / information_schema).
* ``comp.company`` — firm header including GICS codes (``gsector``, ``ggroup``,
  ``gind``, ``gsubind``) keyed by ``gvkey``.

If a query fails with *undefined column* or *relation does not exist*, inspect::

    conn.list_libraries()
    conn.raw_sql("SELECT table_schema, table_name FROM information_schema.tables "
                 "WHERE table_schema IN ('crsp','comp') AND table_name LIKE '%link%' LIMIT 50")

and adjust identifiers in **WRDS_TABLES** below.

================================================================================
5. ENVIRONMENT ASSUMPTIONS TO ADJUST
================================================================================

* **SQLAlchemy 2 + pandas**: WRDS + ``pandas.read_sql`` expects a recent SQLAlchemy;
  this repo patches ``wrds`` in ``wrds_data.get_connection`` for SA2 compatibility.
* **Subscriptions**: CRSP + Compustat + CCM link must be licensed; otherwise tables
  are empty or inaccessible.
* **Column names**: Occasionally WRDS migrates column names; verify with one-row
  ``SELECT * FROM ... LIMIT 1``.

================================================================================
4. COMMON FAILURE MODES
================================================================================

* **No permno**: ticker not in ``stocknames`` on ``ipo_date`` (foreign listing,
  OTC-only history, typo, or pre-CRSP coverage).
* **Multiple permnos**: ticker reuse or overlapping name rows — resolved with
  explicit rules and flagged.
* **No CCM link**: no primary research link for that permno on ``ipo_date``.
* **Missing GICS**: ``gvkey`` present but GICS null — coverage or non-standard firm.
* **Very recent IPO**: Compustat may not yet have GICS populated.

For production, persist raw match tables and audit flags before research use.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Repo root on path when running scripts
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.wrds_data import close_wrds_connection, ensure_sqlalchemy_compat_for_pandas, get_connection

# --- Adjustable table/column names (PostgreSQL defaults on WRDS) ---------------
WRDS_TABLES = {
    "stocknames": "crsp.stocknames",
    "dsenames": "crsp.dsenames",
    "ccm_link": "crsp.ccmxpf_linktable",
    "comp_company": "comp.company",
}

# Primary equity share classes in CRSP (common stock); used to break ties on names.
_CRSP_COMMON_SHRCD = frozenset({10, 11})

# CCM link filters (see CRSP-Compustat Merged documentation on WRDS):
# - linkprim = 'P'  → primary security link (standard in firm-level Compustat merges).
# - linktype IN ('LC','LU','LS') → research-grade link types (LC is most common;
#   LU/LS included as fallback when LC missing — tighten to ('LC',) if you require
#   stricter matching).
_CCM_LINKPRIM_OK = ("P",)
_CCM_LINKTYPE_OK = ("LC", "LU", "LS")


def _norm_ticker(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return str(s).upper().strip().replace(".", "-")


def _to_datetime64_date(s: Any) -> pd.Series:
    """Coerce to timezone-naive dates (midnight)."""
    out = pd.to_datetime(s, errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.dt.normalize()


def clean_ipo_input(ipo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize ``ticker`` and ``ipo_date``; drop rows with missing essentials.

    Optional column ``first_crsp_date``: first date CRSP has prices for the ticker
    (e.g. from ``prices_ipo``). When present, ``match_date = max(ipo_date, first_crsp_date)``
    so ``stocknames`` / ``dsenames`` intervals align with when CRSP actually lists the name.

    Adds ``_ipo_row`` stable index for merge tracking.
    """
    df = ipo_df.copy()
    if "ticker" not in df.columns or "ipo_date" not in df.columns:
        raise ValueError("ipo_df must contain columns 'ticker' and 'ipo_date'")
    df["_ipo_row"] = np.arange(len(df), dtype=np.int64)
    df["ticker_norm"] = df["ticker"].map(_norm_ticker)
    df["ipo_date"] = _to_datetime64_date(df["ipo_date"])
    df["match_date"] = df["ipo_date"]
    if "first_crsp_date" in df.columns:
        fd = _to_datetime64_date(df["first_crsp_date"])
        ok = fd.notna() & df["ipo_date"].notna()
        df.loc[ok, "match_date"] = np.maximum(
            df.loc[ok, "ipo_date"].values,
            fd.loc[ok].values,
        )
    bad = df["ticker_norm"].eq("") | df["ipo_date"].isna()
    df["_input_invalid"] = bad
    return df


def _sql_in_list(values: list[str]) -> str:
    return ",".join("'" + str(v).replace("'", "''") + "'" for v in values)


def fetch_stocknames_bulk(
    conn,
    tickers: list[str],
    *,
    chunk_size: int = 400,
) -> pd.DataFrame:
    """
    Pull CRSP ``stocknames`` for all candidate tickers (``crsp.stocknames``).

    Columns used: ``permno``, ``ticker``, ``namedt``, ``nameenddt``, ``shrcd``,
    ``exchcd``, ``comnam`` — for date-valid matching and tie-breaking.
    """
    tickers = sorted({_norm_ticker(t) for t in tickers if _norm_ticker(t)})
    if not tickers:
        return pd.DataFrame(
            columns=[
                "permno",
                "ticker",
                "namedt",
                "nameenddt",
                "shrcd",
                "exchcd",
                "comnam",
            ]
        )

    parts: list[pd.DataFrame] = []
    sn = WRDS_TABLES["stocknames"]
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        in_list = _sql_in_list(chunk)
        sql = f"""
            SELECT permno, ticker, namedt, nameenddt, shrcd, exchcd, comnam
            FROM {sn}
            WHERE UPPER(TRIM(ticker)) IN ({in_list})
        """
        parts.append(conn.raw_sql(sql, date_cols=["namedt", "nameenddt"]))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(out) == 0:
        return out
    out["ticker_norm"] = out["ticker"].map(_norm_ticker)
    return out


def fetch_dsenames_bulk(
    conn,
    tickers: list[str],
    *,
    chunk_size: int = 400,
) -> pd.DataFrame:
    """
    Pull ``crsp.dsenames`` for DSF-linked names (same ticker universe as daily returns).

    Used when ``stocknames`` has no date-valid row but ``dsenames`` does (common for
    exchange-listed names in the CRSP daily feed).
    """
    tickers = sorted({_norm_ticker(t) for t in tickers if _norm_ticker(t)})
    if not tickers:
        return pd.DataFrame(columns=["permno", "ticker", "namedt", "nameenddt", "comnam"])

    tbl = WRDS_TABLES["dsenames"]
    parts: list[pd.DataFrame] = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        in_list = _sql_in_list(chunk)
        chunk_df: pd.DataFrame | None = None
        for sql in (
            f"""
            SELECT permno, ticker, namedt, nameenddt AS nameenddt, comnam
            FROM {tbl}
            WHERE UPPER(TRIM(ticker)) IN ({in_list})
            """,
            f"""
            SELECT permno, ticker, namedt, nameendt AS nameenddt, comnam
            FROM {tbl}
            WHERE UPPER(TRIM(ticker)) IN ({in_list})
            """,
            f"""
            SELECT permno, ticker, namedt, nameenddt AS nameenddt
            FROM {tbl}
            WHERE UPPER(TRIM(ticker)) IN ({in_list})
            """,
        ):
            try:
                chunk_df = conn.raw_sql(sql, date_cols=["namedt", "nameenddt"])
                break
            except Exception:
                chunk_df = None
                continue
        if chunk_df is not None and len(chunk_df) > 0:
            parts.append(chunk_df)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(out) == 0:
        return pd.DataFrame(columns=["permno", "ticker", "namedt", "nameenddt", "comnam"])
    out["ticker_norm"] = out["ticker"].map(_norm_ticker)
    if "comnam" not in out.columns:
        out["comnam"] = pd.NA
    out["shrcd"] = np.nan
    out["exchcd"] = np.nan
    return out


def _pick_permno_from_hits(hit: pd.DataFrame, *, has_shrcd: bool) -> tuple[int, Any, Any]:
    """Return permno, shrcd, comnam from candidate rows."""
    if len(hit) == 1:
        row = hit.iloc[0]
        return int(row["permno"]), row.get("shrcd"), row.get("comnam")
    if has_shrcd:
        hit = hit.copy()
        hit["shrcd"] = pd.to_numeric(hit["shrcd"], errors="coerce")
        common = hit[hit["shrcd"].isin(_CRSP_COMMON_SHRCD)]
        pick = common if not common.empty else hit
    else:
        pick = hit
    pick = pick.sort_values(["namedt", "permno"], ascending=[False, True])
    row = pick.iloc[0]
    return int(row["permno"]), row.get("shrcd"), row.get("comnam")


def _date_in_name_range(ipo: pd.Timestamp, namedt: Any, nameenddt: Any) -> bool:
    if pd.isna(ipo) or pd.isna(namedt):
        return False
    nd = pd.Timestamp(namedt).normalize()
    ne = pd.Timestamp(nameenddt).normalize() if not pd.isna(nameenddt) else pd.Timestamp("2099-12-31")
    # CRSP sometimes uses far-future sentinel for "current" name end
    if ne.year >= 2030:
        ne = pd.Timestamp("2099-12-31")
    return nd <= ipo <= ne


def match_permno_date_valid(
    ipo_clean: pd.DataFrame,
    stocknames: pd.DataFrame,
    dsenames: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    For each IPO row, find ``permno`` whose CRSP name interval contains **match_date**
    (``max(ipo_date, first_crsp_date)`` when ``first_crsp_date`` was supplied).

    Tries ``crsp.stocknames`` first, then ``crsp.dsenames`` if the first has no hit
    (aligns with the DSF price feed used elsewhere in this repo).

    Tie-breaking: prefer ``shrcd`` in {10, 11} when available; else latest ``namedt``,
    then smallest ``permno``.
    """
    rows: list[dict[str, Any]] = []
    sn = stocknames.copy()
    ds = dsenames.copy() if dsenames is not None and not dsenames.empty else pd.DataFrame()
    if not sn.empty:
        sn["shrcd"] = pd.to_numeric(sn["shrcd"], errors="coerce")
        if "ticker_norm" not in sn.columns:
            sn["ticker_norm"] = sn["ticker"].map(_norm_ticker)
    if not ds.empty and "ticker_norm" not in ds.columns:
        ds["ticker_norm"] = ds["ticker"].map(_norm_ticker)

    for _, r in ipo_clean.iterrows():
        rec: dict[str, Any] = {
            "_ipo_row": int(r["_ipo_row"]),
            "permno": pd.NA,
            "permno_source": pd.NA,
            "no_crsp_match": True,
            "multiple_crsp_matches": False,
            "multiple_crsp_matches_resolved": False,
            "crsp_shrcd": pd.NA,
            "crsp_comnam": pd.NA,
            "match_stage": "invalid_input",
        }
        if bool(r.get("_input_invalid", False)):
            rows.append(rec)
            continue
        if sn.empty and ds.empty:
            rec["match_stage"] = "no_crsp_name_tables"
            rows.append(rec)
            continue

        tic = r["ticker_norm"]
        d = r["match_date"] if pd.notna(r.get("match_date")) else r["ipo_date"]

        def _hits(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty:
                return pd.DataFrame()
            cand = frame[frame["ticker_norm"] == tic]
            mask = cand.apply(
                lambda x: _date_in_name_range(d, x["namedt"], x["nameenddt"]),
                axis=1,
            )
            return cand[mask]

        hit = _hits(sn) if not sn.empty else pd.DataFrame()
        src = "stocknames"
        if hit.empty and not ds.empty:
            hit = _hits(ds)
            src = "dsenames"

        rec["no_crsp_match"] = False
        rec["match_stage"] = "ok"

        if hit.empty:
            rec["no_crsp_match"] = True
            rec["match_stage"] = "no_crsp_match"
        elif len(hit) == 1:
            rec["permno"] = int(hit.iloc[0]["permno"])
            rec["crsp_shrcd"] = hit.iloc[0].get("shrcd")
            rec["crsp_comnam"] = hit.iloc[0].get("comnam")
            rec["permno_source"] = src
        else:
            rec["multiple_crsp_matches"] = True
            has_sh = "shrcd" in hit.columns and hit["shrcd"].notna().any()
            pno, sh, cn = _pick_permno_from_hits(hit, has_shrcd=has_sh)
            rec["permno"] = pno
            rec["crsp_shrcd"] = sh
            rec["crsp_comnam"] = cn
            rec["permno_source"] = src
            rec["multiple_crsp_matches_resolved"] = len(hit) > 1
            rec["match_stage"] = "multiple_crsp_resolved"
        rows.append(rec)

    return pd.DataFrame(rows)


def fetch_ccm_links_bulk(
    conn,
    permnos: list[int],
    *,
    chunk_size: int = 450,
) -> pd.DataFrame:
    """
    ``crsp.ccmxpf_linktable``: map ``lpermno`` → ``gvkey`` with link metadata.

    We pull broadly here; date filtering and linktype/linkprim screens happen in pandas
    so we can apply transparent duplicate-resolution rules.
    """
    permnos = sorted({int(p) for p in permnos if pd.notna(p)})
    if not permnos:
        return pd.DataFrame(
            columns=["lpermno", "gvkey", "linkdt", "linkenddt", "linktype", "linkprim"]
        )
    parts: list[pd.DataFrame] = []
    lt = WRDS_TABLES["ccm_link"]
    for i in range(0, len(permnos), chunk_size):
        chunk = permnos[i : i + chunk_size]
        plist = ",".join(str(p) for p in chunk)
        sql = f"""
            SELECT lpermno, gvkey, linkdt, linkenddt, linktype, linkprim
            FROM {lt}
            WHERE lpermno IN ({plist})
        """
        parts.append(conn.raw_sql(sql, date_cols=["linkdt", "linkenddt"]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _ccm_covers_date(
    ipo: pd.Timestamp,
    linkdt: Any,
    linkenddt: Any,
) -> bool:
    if pd.isna(ipo) or pd.isna(linkdt):
        return False
    ld = pd.Timestamp(linkdt).normalize()
    le = pd.Timestamp(linkenddt).normalize() if not pd.isna(linkenddt) else pd.Timestamp("2099-12-31")
    if le.year >= 2030:
        le = pd.Timestamp("2099-12-31")
    return ld <= ipo <= le


def _linktype_rank(lt: str) -> int:
    """Lower is better (research preference)."""
    order = {"LC": 0, "LU": 1, "LS": 2}
    return order.get(str(lt).strip().upper(), 9)


def match_gvkey_from_ccm(
    ipo_with_permno: pd.DataFrame,
    ccm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Date-valid CCM merge: keep rows with ``linkprim`` primary and allowed ``linktype``,
    interval containing ``match_date`` (fallback ``ipo_date``).

    Residual duplicates: prefer best ``linktype`` (LC first), then latest ``linkdt``
    still on or before IPO (stronger economic identification of the active link at IPO).
    """
    df = ipo_with_permno.copy()
    if ccm.empty:
        df["gvkey"] = pd.NA
        df["no_ccm_link"] = True
        df["multiple_ccm_links"] = False
        df["ccm_linktype"] = pd.NA
        df["ccm_linkprim"] = pd.NA
        return df

    c = ccm.copy()
    c["gvkey"] = c["gvkey"].astype(str).str.strip().str.zfill(6)
    c["linktype_u"] = c["linktype"].astype(str).str.strip().str.upper()
    c["linkprim_u"] = c["linkprim"].astype(str).str.strip().str.upper()

    gvkeys: list[Any] = []
    no_link: list[bool] = []
    multi: list[bool] = []
    lts: list[Any] = []
    lps: list[Any] = []

    for _, r in df.iterrows():
        p = r.get("permno")
        d = r.get("match_date")
        if pd.isna(d):
            d = r.get("ipo_date")
        if pd.isna(p) or pd.isna(d):
            gvkeys.append(pd.NA)
            no_link.append(True)
            multi.append(False)
            lts.append(pd.NA)
            lps.append(pd.NA)
            continue
        sub = c[c["lpermno"] == int(p)]
        sub = sub[
            sub["linkprim_u"].isin(_CCM_LINKPRIM_OK)
            & sub["linktype_u"].isin(_CCM_LINKTYPE_OK)
        ]
        sub = sub[sub.apply(lambda x: _ccm_covers_date(d, x["linkdt"], x["linkenddt"]), axis=1)]
        if sub.empty:
            gvkeys.append(pd.NA)
            no_link.append(True)
            multi.append(False)
            lts.append(pd.NA)
            lps.append(pd.NA)
        elif len(sub) == 1:
            gvkeys.append(sub.iloc[0]["gvkey"])
            no_link.append(False)
            multi.append(False)
            lts.append(sub.iloc[0]["linktype_u"])
            lps.append(sub.iloc[0]["linkprim_u"])
        else:
            multi.append(True)
            no_link.append(False)
            sub = sub.copy()
            sub["_ltr"] = sub["linktype_u"].map(_linktype_rank)
            sub = sub.sort_values(["_ltr", "linkdt"], ascending=[True, False])
            chosen = sub.iloc[0]
            gvkeys.append(chosen["gvkey"])
            lts.append(chosen["linktype_u"])
            lps.append(chosen["linkprim_u"])

    df["gvkey"] = gvkeys
    df["no_ccm_link"] = no_link
    df["multiple_ccm_links"] = multi
    df["ccm_linktype"] = lts
    df["ccm_linkprim"] = lps
    return df


def fetch_compustat_gics_bulk(
    conn,
    gvkeys: list[str],
    *,
    chunk_size: int = 400,
) -> pd.DataFrame:
    """``comp.company`` GICS fields (keyed by ``gvkey``)."""
    gvkeys = sorted({str(g).strip().zfill(6) for g in gvkeys if pd.notna(g) and str(g).strip()})
    if not gvkeys:
        return pd.DataFrame(columns=["gvkey", "gsector", "ggroup", "gind", "gsubind", "conm"])
    parts: list[pd.DataFrame] = []
    cc = WRDS_TABLES["comp_company"]
    for i in range(0, len(gvkeys), chunk_size):
        chunk = gvkeys[i : i + chunk_size]
        glist = _sql_in_list(chunk)
        sql = f"""
            SELECT gvkey, gsector, ggroup, gind, gsubind, conm
            FROM {cc}
            WHERE gvkey IN ({glist})
        """
        parts.append(conn.raw_sql(sql))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(out) == 0:
        return out
    out["gvkey"] = out["gvkey"].astype(str).str.strip().str.zfill(6)
    return out.drop_duplicates(subset=["gvkey"], keep="first")


def enrich_ipo_with_gics(
    ipo_df: pd.DataFrame,
    conn,
    *,
    include_compustat_name: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: ``(ticker, ipo_date)`` → ``permno`` → ``gvkey`` → Compustat GICS.

    Optional column ``first_crsp_date``: first CRSP price date per ticker; when set,
    matching uses ``match_date = max(ipo_date, first_crsp_date)`` and tries
    ``dsenames`` if ``stocknames`` misses.

    Returns
    -------
    enriched : pd.DataFrame
        One row per input IPO (same order as ``_ipo_row``); includes diagnostics.
    diagnostics : pd.DataFrame
        Single-row summary counts plus optional unmatched samples.
    """
    ensure_sqlalchemy_compat_for_pandas()
    clean = clean_ipo_input(ipo_df)
    tickers = clean["ticker_norm"].dropna().unique().tolist()
    sn = fetch_stocknames_bulk(conn, tickers)
    ds = fetch_dsenames_bulk(conn, tickers)
    mperm = match_permno_date_valid(clean, sn, ds)
    step1 = clean.merge(mperm, on="_ipo_row", how="left", validate="one_to_one")
    permnos = (
        pd.to_numeric(step1["permno"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    ccm = fetch_ccm_links_bulk(conn, permnos)
    step2 = match_gvkey_from_ccm(step1, ccm)
    gvkeys = (
        step2["gvkey"].dropna().astype(str).str.strip().str.zfill(6).unique().tolist()
    )
    comp = fetch_compustat_gics_bulk(conn, gvkeys)
    if not comp.empty:
        step2["gvkey"] = step2["gvkey"].map(
            lambda x: str(x).strip().zfill(6)
            if pd.notna(x) and str(x).strip() != ""
            else pd.NA
        )
        step2 = step2.merge(comp, on="gvkey", how="left", suffixes=("", "_compdup"))
        # If duplicate gvkey columns from merge, prefer left
        for c in step2.columns:
            if c.endswith("_compdup"):
                step2 = step2.drop(columns=[c])
    else:
        for col in ("gsector", "ggroup", "gind", "gsubind", "conm"):
            step2[col] = pd.NA

    if not include_compustat_name:
        step2 = step2.drop(columns=["conm"], errors="ignore")

    step2["no_compustat_gics"] = (
        step2["gvkey"].notna()
        & step2["gsector"].isna()
        & step2["ggroup"].isna()
        & step2["gind"].isna()
        & step2["gsubind"].isna()
    )
    shrcd_num = pd.to_numeric(step2["crsp_shrcd"], errors="coerce")
    step2["likely_nonoperating_entity"] = (
        shrcd_num.notna() & ~shrcd_num.isin(_CRSP_COMMON_SHRCD)
    ) | (step2["permno"].notna() & step2["gvkey"].notna() & step2["gsector"].isna())

    # Final column order (minimum requested + diagnostics)
    diag_cols = [
        "no_crsp_match",
        "multiple_crsp_matches",
        "multiple_crsp_matches_resolved",
        "no_ccm_link",
        "multiple_ccm_links",
        "no_compustat_gics",
        "likely_nonoperating_entity",
        "match_stage",
    ]
    base = ["ticker", "ipo_date", "permno", "gvkey", "gsector", "ggroup", "gind", "gsubind"]
    extra = [
        c
        for c in (
            "match_date",
            "first_crsp_date",
            "permno_source",
            "conm",
            "crsp_comnam",
            "crsp_shrcd",
            "ccm_linktype",
            "ccm_linkprim",
        )
        if c in step2.columns
    ]
    out = step2[base + extra + [c for c in diag_cols if c in step2.columns] + ["_ipo_row"]].sort_values("_ipo_row")

    # Diagnostics table
    n = len(out)
    summary = {
        "n_ipo": n,
        "n_matched_permno": int(out["permno"].notna().sum()),
        "n_matched_gvkey": int(out["gvkey"].notna().sum()),
        "n_with_gsector": int(out["gsector"].notna().sum()),
        "n_no_crsp": int(out["no_crsp_match"].fillna(False).sum()),
        "n_multi_crsp": int(out["multiple_crsp_matches"].fillna(False).sum()),
        "n_no_ccm": int(out["no_ccm_link"].fillna(False).sum()),
        "n_no_gics": int(out["no_compustat_gics"].fillna(False).sum()),
    }
    diag = pd.DataFrame([summary])
    # Sample unmatched
    samples = []
    for label, cond in (
        ("unmatched_crsp", out["no_crsp_match"].fillna(False)),
        ("unmatched_ccm", out["no_ccm_link"].fillna(False) & ~out["no_crsp_match"].fillna(False)),
        ("missing_gics", out["no_compustat_gics"].fillna(False)),
    ):
        sub = out.loc[cond, ["ticker", "ipo_date", "permno", "gvkey"]].head(5)
        sub.insert(0, "sample_type", label)
        samples.append(sub)
    diag_samples = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    return out, pd.concat([diag, diag_samples], ignore_index=True)


def print_wrds_table_checklist(conn) -> None:
    """Optional: print one row from each core table to verify access/columns."""
    for key, table in WRDS_TABLES.items():
        try:
            df = conn.raw_sql(f"SELECT * FROM {table} LIMIT 1")
            print(f"[OK] {key} -> {table}  cols={list(df.columns)[:12]}...")
        except Exception as e:
            print(f"[FAIL] {key} -> {table}: {e}")


def _demo() -> None:
    """Minimal runnable example (requires WRDS credentials)."""
    ensure_sqlalchemy_compat_for_pandas()
    conn = get_connection()
    ipo_df = pd.DataFrame(
        {
            "ticker": ["AAPL", "BADTICKERZZ"],
            "ipo_date": ["1980-12-12", "2020-01-15"],
        }
    )
    enriched, diag = enrich_ipo_with_gics(ipo_df, conn)
    print(enriched.to_string())
    print("\nDiagnostics:")
    print(diag.head(20).to_string())
    close_wrds_connection(conn)


if __name__ == "__main__":
    _demo()
