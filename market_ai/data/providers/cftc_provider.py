from __future__ import annotations

import urllib.request
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from market_ai.data.storage import read_table


def _first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {str(col).lower(): str(col) for col in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def normalize_cftc_cot_frame(frame: pd.DataFrame, *, filter_crude: bool = True) -> pd.DataFrame:
    raw = frame.copy()
    market_col = _first_existing(raw, ["Market_and_Exchange_Names", "market", "market_name", "contract_market_name"])
    if filter_crude and market_col:
        mask = raw[market_col].astype(str).str.contains("CRUDE|WTI|LIGHT SWEET", case=False, na=False)
        if mask.any():
            raw = raw[mask].copy()
    date_col = _first_existing(raw, ["report_date", "Report_Date_as_YYYY-MM-DD", "As_of_Date_In_Form_YYMMDD", "date"])
    if date_col is None:
        raise ValueError("CFTC COT data requires report_date/date column")
    out = pd.DataFrame({"report_date": pd.to_datetime(raw[date_col], errors="coerce", utc=True)})
    aliases = {
        "open_interest": ["open_interest", "Open_Interest_All", "Open Interest"],
        "managed_money_long": ["managed_money_long", "M_Money_Positions_Long_All", "Money_Manager_Long_All"],
        "managed_money_short": ["managed_money_short", "M_Money_Positions_Short_All", "Money_Manager_Short_All"],
        "commercial_long": ["commercial_long", "Prod_Merc_Positions_Long_All", "Commercial_Long_All"],
        "commercial_short": ["commercial_short", "Prod_Merc_Positions_Short_All", "Commercial_Short_All"],
    }
    for target, candidates in aliases.items():
        col = _first_existing(raw, candidates)
        out[target] = pd.to_numeric(raw[col], errors="coerce") if col else pd.NA
    out = out.dropna(subset=["report_date"]).sort_values("report_date")
    out["managed_money_net"] = out["managed_money_long"].fillna(0.0) - out["managed_money_short"].fillna(0.0)
    out["commercial_net"] = out["commercial_long"].fillna(0.0) - out["commercial_short"].fillna(0.0)
    rolling = out["managed_money_net"].rolling(52, min_periods=8)
    out["managed_money_net_zscore"] = (out["managed_money_net"] - rolling.mean()) / rolling.std().replace(0.0, pd.NA)
    out["open_interest_change"] = out["open_interest"].diff()
    out["release_time"] = out["report_date"].map(
        lambda ts: pd.Timestamp(datetime.combine((ts + pd.Timedelta(days=3)).date(), time(20, 30), tzinfo=timezone.utc))
        if not pd.isna(ts)
        else pd.NaT
    )
    out["as_of_time"] = out["release_time"]
    out["source"] = "cftc_cot"
    return out.replace([float("inf"), float("-inf")], pd.NA).reset_index(drop=True)


def load_cftc_manual_csv(path: str | Path) -> pd.DataFrame:
    return normalize_cftc_cot_frame(read_table(path))


def fetch_cftc_csv(url: str) -> pd.DataFrame:
    if not url:
        raise RuntimeError("CFTC URL is required for live download; pass --manual-csv for offline ingest.")
    with urllib.request.urlopen(url, timeout=30) as response:
        frame = pd.read_csv(response)
    return normalize_cftc_cot_frame(frame)


def cot_weekly_to_daily_point_in_time(frame: pd.DataFrame, *, end: str | None = None) -> pd.DataFrame:
    weekly = normalize_cftc_cot_frame(frame, filter_crude=False)
    if weekly.empty:
        return weekly
    start = weekly["as_of_time"].min().floor("D")
    stop = pd.to_datetime(end, errors="coerce", utc=True) if end else pd.Timestamp(datetime.now(timezone.utc))
    daily = pd.DataFrame({"timestamp": pd.date_range(start, stop.floor("D"), freq="D", tz="UTC")})
    merged = pd.merge_asof(
        daily.sort_values("timestamp"),
        weekly.sort_values("as_of_time"),
        left_on="timestamp",
        right_on="as_of_time",
        direction="backward",
    )
    merged["feature_available_at"] = merged["as_of_time"]
    return merged.reset_index(drop=True)
