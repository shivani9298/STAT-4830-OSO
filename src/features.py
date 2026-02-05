"""
Feature extraction for policy network (course: Linear Regression, Best Linear Approximation).
Episode -> fixed-size feature vector; uses episode.df and episode.meta (full IPO row).
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import torch

from src.data import Episode


# Feature dimension: 8 base + 12 meta-based
N_FEATURES_BASE = 8
N_FEATURES_META = 12
N_FEATURES = N_FEATURES_BASE + N_FEATURES_META  # 20

# Base feature indices (from episode.df)
IDX_DAY0_CLOSE_NORM = 0
IDX_DAY1_CLOSE_NORM = 1
IDX_RETURN_01 = 2
IDX_VOL_PROXY = 3
IDX_VOLUME_NORM = 4
IDX_N_DAYS = 5
IDX_DAY0_VOLUME = 6
IDX_CONST = 7

# Meta feature indices (from episode.meta)
IDX_OFFER_PRICE_NORM = 8   # Price (offer price) / 100
IDX_SHARES_NORM = 9        # log1p(Shares) / 20
IDX_OFFER_AMOUNT_NORM = 10 # log1p(Offer Amount) / 25
IDX_EMPLOYEES_NORM = 11    # log1p(employees) / 15
IDX_SECTOR_ID = 12        # hash(sector) % 1000 / 1000
IDX_INDUSTRY_ID = 13      # hash(industry) % 1000 / 1000
IDX_EMPLOYEES2019_NORM = 14  # log1p(employees2019) / 15
IDX_CEO_PAY_NORM = 15     # log1p(CEO_pay) / 20
IDX_CEO_BORN_NORM = 16    # (CEO_born - 1950) / 50
IDX_FIRSTDAY_OPEN_NORM = 17   # firstday_open / 100
IDX_FIRSTDAY_ADJCLOSE_NORM = 18  # firstday_adjclose / 100
IDX_FIRST_DAY_RETURN = 19  # (firstday_adjclose - firstday_open) / firstday_open if available


def _safe_float(x, default: float = 0.0) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _encode_cat(s: Optional[str]) -> float:
    if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == "":
        return 0.0
    return (hash(str(s).strip()) % 1000) / 1000.0


def episode_to_features(episode: Episode) -> np.ndarray:
    """
    Extract fixed-size feature vector from episode (df + meta).
    Uses episode.df for price/volume and episode.meta for company, offer, sector, CEO, etc.
    """
    out = np.zeros(N_FEATURES, dtype=np.float32)
    df = episode.df
    meta = episode.meta or {}
    
    # Base features from df
    ret_01 = 0.0
    if len(df) >= 2:
        day0_close = float(df.iloc[0]["close"])
        day1_close = float(df.iloc[1]["close"])
        ret_01 = (day1_close - day0_close) / day0_close if day0_close > 0 else 0.0
        out[IDX_DAY0_CLOSE_NORM] = day0_close / 100.0
        out[IDX_DAY1_CLOSE_NORM] = day1_close / 100.0
        out[IDX_RETURN_01] = ret_01
        out[IDX_VOL_PROXY] = np.abs(ret_01)
    out[IDX_N_DAYS] = min(episode.N, 20) / 20.0
    if "volume" in df.columns and len(df) > 0:
        v0 = df.iloc[0].get("volume", 0)
        v0 = np.nan_to_num(v0, nan=0.0, posinf=0.0, neginf=0.0)
        out[IDX_VOLUME_NORM] = np.log1p(max(0, float(v0))) / 10.0
        out[IDX_DAY0_VOLUME] = min(1.0, float(v0) / 1e6)
    out[IDX_CONST] = 1.0
    
    # Meta-based features (all columns from dataset)
    out[IDX_OFFER_PRICE_NORM] = _safe_float(meta.get("Price"), 0.0) / 100.0
    out[IDX_SHARES_NORM] = np.log1p(max(0, _safe_float(meta.get("Shares"), 0.0))) / 20.0
    out[IDX_OFFER_AMOUNT_NORM] = np.log1p(max(0, _safe_float(meta.get("Offer Amount"), 0.0))) / 25.0
    out[IDX_EMPLOYEES_NORM] = np.log1p(max(0, _safe_float(meta.get("employees"), 0.0))) / 15.0
    out[IDX_SECTOR_ID] = _encode_cat(meta.get("sector"))
    out[IDX_INDUSTRY_ID] = _encode_cat(meta.get("industry"))
    out[IDX_EMPLOYEES2019_NORM] = np.log1p(max(0, _safe_float(meta.get("employees2019"), 0.0))) / 15.0
    out[IDX_CEO_PAY_NORM] = np.log1p(max(0, _safe_float(meta.get("CEO_pay"), 0.0))) / 20.0
    cb = _safe_float(meta.get("CEO_born"), 1950.0)
    out[IDX_CEO_BORN_NORM] = (cb - 1950.0) / 50.0
    fo = _safe_float(meta.get("firstday_open"), 0.0)
    fc = _safe_float(meta.get("firstday_adjclose"), 0.0)
    out[IDX_FIRSTDAY_OPEN_NORM] = fo / 100.0 if fo > 0 else 0.0
    out[IDX_FIRSTDAY_ADJCLOSE_NORM] = fc / 100.0 if fc > 0 else 0.0
    if fo > 0 and fc > 0:
        out[IDX_FIRST_DAY_RETURN] = (fc - fo) / fo
    else:
        out[IDX_FIRST_DAY_RETURN] = ret_01  # fallback to df return
    
    return out


def episodes_to_tensor(episodes: List[Episode], device: torch.device) -> torch.Tensor:
    """Batch of episodes -> (B, N_FEATURES) tensor."""
    rows = [episode_to_features(ep) for ep in episodes]
    return torch.tensor(np.stack(rows), dtype=torch.float32, device=device)
