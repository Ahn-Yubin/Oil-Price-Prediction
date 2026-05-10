from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
import zipfile

import pandas as pd
import requests

from market_ai.data.storage import read_table


EIA_SERIES = {
    "crude_stocks": "PET.WCESTUS1.W",
    "cushing_stocks": "PET.WCESTP11.W",
    "gasoline_stocks": "PET.WGTSTUS1.W",
    "distillate_stocks": "PET.WDISTUS1.W",
    "refinery_utilization": "PET.WPULEUS3.W",
    "crude_production": "PET.WCRFPUS2.W",
    "crude_imports": "PET.WCRIMUS2.W",
    "crude_exports": "PET.WCREXUS2.W",
    "spr_stocks": "PET.WCSSTUS1.W",
}


@dataclass(frozen=True)
class EIAFetchConfig:
    api_key: str | None = None
    base_url: str = "https://api.eia.gov/v2/seriesid"
    bulk_url: str = "https://api.eia.gov/bulk/PET.zip"


def _weekly_release_time(date_value: Any) -> pd.Timestamp:
    report_date = pd.to_datetime(date_value, errors="coerce", utc=True)
    if pd.isna(report_date):
        return pd.NaT
    # EIA weekly petroleum status reports are usually released Wednesday 10:30 ET.
    # Store a conservative UTC timestamp after the scheduled release.
    release_date = report_date + pd.Timedelta(days=5)
    return pd.Timestamp(datetime.combine(release_date.date(), time(16, 0), tzinfo=timezone.utc))


def normalize_eia_manual_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out.rename(
        columns={
            "period": "report_date",
            "date": "report_date",
            "week": "report_date",
            "value": "value",
            "series": "series_id",
            "series_id": "series_id",
        }
    )
    if {"series_id", "value"}.issubset(out.columns):
        out["series_id"] = out["series_id"].astype(str)
        out["metric"] = out.get("metric", out["series_id"].map({v: k for k, v in EIA_SERIES.items()})).astype(str)
        pivot = out.pivot_table(index="report_date", columns="metric", values="value", aggfunc="last").reset_index()
        out = pivot
    if "report_date" not in out.columns:
        raise ValueError("EIA data requires report_date/date/period column")
    out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce", utc=True)
    out = out.dropna(subset=["report_date"]).sort_values("report_date")
    numeric_cols = [col for col in out.columns if col not in {"report_date", "release_time", "as_of_time", "source"}]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "release_time" not in out.columns:
        out["release_time"] = out["report_date"].map(_weekly_release_time)
    out["release_time"] = pd.to_datetime(out["release_time"], errors="coerce", utc=True)
    out["as_of_time"] = out["release_time"]
    out["source"] = out.get("source", "eia_manual")
    for col in ["crude_stocks", "cushing_stocks", "gasoline_stocks", "distillate_stocks", "crude_production", "crude_imports", "crude_exports"]:
        if col in out.columns:
            out[f"{col}_change"] = out[col].diff()
    if {"crude_imports", "crude_exports"}.issubset(out.columns):
        out["imports_exports_spread"] = out["crude_imports"] - out["crude_exports"]
    return out.reset_index(drop=True)


def load_eia_manual_csv(path: str | Path) -> pd.DataFrame:
    return normalize_eia_manual_frame(read_table(path))


def fetch_eia_series(config: EIAFetchConfig | None = None) -> pd.DataFrame:
    cfg = config or EIAFetchConfig(api_key=os.environ.get("EIA_API_KEY"))
    if not cfg.api_key:
        raise RuntimeError("EIA_API_KEY is not set; pass --manual-csv for offline/manual ingest.")
    rows: list[dict[str, Any]] = []
    for metric, series_id in EIA_SERIES.items():
        response = requests.get(
            f"{cfg.base_url}/{series_id}",
            params={"api_key": cfg.api_key, "facets[series][]": series_id},
            headers={"User-Agent": "market-ai-data-collector/1.0"},
            timeout=20,
            verify=_requests_verify(),
        )
        response.raise_for_status()
        body = response.json()
        data_rows = body.get("response", {}).get("data", [])
        for row in data_rows:
            rows.append({"report_date": row.get("period"), "metric": metric, "value": row.get("value"), "series_id": series_id, "source": "eia_api"})
    if not rows:
        raise RuntimeError("EIA API returned no rows.")
    return normalize_eia_manual_frame(pd.DataFrame(rows))


def fetch_eia_bulk_series(config: EIAFetchConfig | None = None) -> pd.DataFrame:
    cfg = config or EIAFetchConfig(api_key=os.environ.get("EIA_API_KEY"))
    response = requests.get(
        cfg.bulk_url,
        headers={"User-Agent": "market-ai-data-collector/1.0"},
        timeout=90,
        verify=_requests_verify(),
    )
    response.raise_for_status()
    reverse_map = {series_id: metric for metric, series_id in EIA_SERIES.items()}
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if not members:
            raise RuntimeError("EIA bulk ZIP did not contain a text payload.")
        with archive.open(members[0]) as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                obj = json.loads(raw_line.decode("utf-8"))
                series_id = str(obj.get("series_id") or "")
                metric = reverse_map.get(series_id)
                if not metric:
                    continue
                for period, value in obj.get("data") or []:
                    rows.append(
                        {
                            "report_date": period,
                            "metric": metric,
                            "value": value,
                            "series_id": series_id,
                            "source": "eia_bulk",
                        }
                    )
    if not rows:
        raise RuntimeError("EIA bulk file did not contain the configured petroleum series.")
    return normalize_eia_manual_frame(pd.DataFrame(rows))


def weekly_to_daily_point_in_time(frame: pd.DataFrame, *, end: str | None = None) -> pd.DataFrame:
    weekly = normalize_eia_manual_frame(frame)
    if weekly.empty:
        return weekly
    start = weekly["as_of_time"].min().floor("D")
    stop = pd.to_datetime(end, errors="coerce", utc=True) if end else pd.Timestamp(datetime.now(timezone.utc) + timedelta(days=1))
    daily = pd.DataFrame({"timestamp": pd.date_range(start, stop.floor("D"), freq="D", tz="UTC")})
    weekly = weekly.sort_values("as_of_time")
    merged = pd.merge_asof(
        daily.sort_values("timestamp"),
        weekly.sort_values("as_of_time"),
        left_on="timestamp",
        right_on="as_of_time",
        direction="backward",
    )
    merged["feature_available_at"] = merged["as_of_time"]
    return merged.reset_index(drop=True)


def _requests_verify() -> str | bool:
    try:
        import certifi

        return certifi.where()
    except Exception:
        return True
