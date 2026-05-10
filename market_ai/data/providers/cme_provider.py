from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

from market_ai.data.storage import read_table


MONTH_CODES = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}


def _first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {str(col).lower(): str(col) for col in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _contract_sort_key(value: object) -> tuple[int, int, str]:
    text = str(value or "").upper().strip()
    year = 9999
    month = 99
    if len(text) >= 3:
        code = text[-3]
        maybe_year = text[-2:]
        if code in MONTH_CODES and maybe_year.isdigit():
            month = MONTH_CODES[code]
            year = 2000 + int(maybe_year)
    return year, month, text


def normalize_cme_settlements_frame(frame: pd.DataFrame) -> pd.DataFrame:
    raw = frame.copy()
    date_col = _first_existing(raw, ["trade_date", "date", "timestamp", "business_date"])
    settle_col = _first_existing(raw, ["settle", "settlement", "settle_price", "Settle"])
    contract_col = _first_existing(raw, ["contract", "contract_month", "month", "symbol"])
    if date_col is None or settle_col is None:
        raise ValueError("CME settlements require trade_date/date and settle/settlement columns")
    if contract_col is None:
        raw["contract"] = raw.groupby(date_col).cumcount().add(1).map(lambda idx: f"M{idx}")
        contract_col = "contract"
    raw["trade_date"] = pd.to_datetime(raw[date_col], errors="coerce", utc=True)
    raw["settle"] = pd.to_numeric(raw[settle_col], errors="coerce")
    raw["contract"] = raw[contract_col].astype(str)
    volume_col = _first_existing(raw, ["volume", "Volume"])
    oi_col = _first_existing(raw, ["open_interest", "Open Interest", "openInterest"])
    raw["volume"] = pd.to_numeric(raw[volume_col], errors="coerce") if volume_col else pd.NA
    raw["open_interest"] = pd.to_numeric(raw[oi_col], errors="coerce") if oi_col else pd.NA
    raw = raw.dropna(subset=["trade_date", "settle"]).sort_values(["trade_date", "contract"])
    rows: list[dict] = []
    for date_value, group in raw.groupby("trade_date"):
        sorted_group = group.assign(_sort=group["contract"].map(_contract_sort_key)).sort_values("_sort")
        settlements = sorted_group["settle"].to_numpy(dtype=float)
        row = {"timestamp": date_value, "trade_date": date_value, "feature_available_at": date_value, "source": "cme_manual"}
        for idx in [1, 2, 3, 6]:
            row[f"m{idx}_settle"] = float(settlements[idx - 1]) if len(settlements) >= idx else pd.NA
        row["m1_m2_spread"] = row["m1_settle"] - row["m2_settle"] if pd.notna(row.get("m1_settle")) and pd.notna(row.get("m2_settle")) else pd.NA
        row["m1_m3_spread"] = row["m1_settle"] - row["m3_settle"] if pd.notna(row.get("m1_settle")) and pd.notna(row.get("m3_settle")) else pd.NA
        row["curve_slope_m1_m6"] = row["m1_settle"] - row["m6_settle"] if pd.notna(row.get("m1_settle")) and pd.notna(row.get("m6_settle")) else pd.NA
        row["volume"] = float(sorted_group["volume"].sum(skipna=True)) if "volume" in sorted_group else pd.NA
        row["open_interest"] = float(sorted_group["open_interest"].sum(skipna=True)) if "open_interest" in sorted_group else pd.NA
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if "open_interest" in out.columns:
        out["open_interest_change"] = out["open_interest"].diff()
    return out


def load_cme_manual_csv(path: str | Path) -> pd.DataFrame:
    return normalize_cme_settlements_frame(read_table(path))


def fetch_cme_csv(url: str) -> pd.DataFrame:
    if not url:
        raise RuntimeError("CME provider URL is required; pass --manual-csv for licensed/manual data.")
    response = requests.get(
        url,
        headers={"User-Agent": "market-ai-data-collector/1.0"},
        timeout=60,
        verify=_requests_verify(),
    )
    response.raise_for_status()
    frame = pd.read_csv(BytesIO(response.content))
    return normalize_cme_settlements_frame(frame)


def _requests_verify() -> str | bool:
    try:
        import certifi

        return certifi.where()
    except Exception:
        return True
