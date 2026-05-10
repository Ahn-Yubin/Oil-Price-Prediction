from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from urllib.parse import urlencode

import pandas as pd
import requests


DEFAULT_FRED_SERIES = {
    "DCOILWTICO": "WTI crude oil spot price",
    "DCOILBRENTEU": "Brent crude oil spot price",
    "DHHNGSP": "Henry Hub natural gas spot price",
    "DTWEXBGS": "Trade weighted U.S. dollar index",
    "DEXUSEU": "U.S. dollar to euro exchange rate",
    "DEXKOUS": "South Korean won to U.S. dollar exchange rate",
    "DEXJPUS": "Japanese yen to U.S. dollar exchange rate",
    "DGS10": "10-year Treasury constant maturity rate",
    "T10YIE": "10-year breakeven inflation rate",
    "VIXCLS": "CBOE volatility index",
}


def normalize_fred_frame(frame: pd.DataFrame, *, series_id: str, label: str | None = None) -> pd.DataFrame:
    if "observation_date" not in frame.columns:
        raise ValueError("FRED CSV requires observation_date column")
    if series_id not in frame.columns:
        raise ValueError(f"FRED CSV missing series column: {series_id}")
    out = frame[["observation_date", series_id]].copy()
    out = out.rename(columns={"observation_date": "date", series_id: "value"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True)
    out["value"] = pd.to_numeric(out["value"].replace(".", pd.NA), errors="coerce")
    out["series_id"] = series_id
    out["series_name"] = label or series_id
    out["provider"] = "fred"
    out["fetched_at"] = datetime.now(timezone.utc).isoformat()
    out = out.dropna(subset=["date"]).sort_values("date")
    return out[["date", "series_id", "series_name", "value", "provider", "fetched_at"]].reset_index(drop=True)


def fred_csv_url(series_id: str) -> str:
    return "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urlencode({"id": series_id})


def fetch_fred_series(series_id: str, *, label: str | None = None) -> pd.DataFrame:
    response = requests.get(
        fred_csv_url(series_id),
        headers={"User-Agent": "market-ai-data-collector/1.0"},
        timeout=15,
        verify=_requests_verify(),
    )
    response.raise_for_status()
    payload = response.text
    frame = pd.read_csv(StringIO(payload))
    return normalize_fred_frame(frame, series_id=series_id, label=label)


def build_fred_wide_panel(long_frame: pd.DataFrame) -> pd.DataFrame:
    if long_frame.empty:
        return pd.DataFrame(columns=["date"])
    wide = long_frame.pivot_table(index="date", columns="series_id", values="value", aggfunc="last").reset_index()
    wide.columns.name = None
    return wide.sort_values("date").reset_index(drop=True)


def _requests_verify() -> str | bool:
    try:
        import certifi

        return certifi.where()
    except Exception:
        return True
