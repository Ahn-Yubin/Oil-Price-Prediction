from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import StringIO
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import pandas as pd
import requests

from market_ai.data.storage import read_table, safe_symbol, write_table


SUPPORTED_INTERVALS = {"1d", "1h", "30m", "15m"}
INTERVAL_FREQ = {"1d": "D", "1h": "h", "30m": "30min", "15m": "15min"}
STOOQ_DAILY_SYMBOLS = {
    "CL=F": "cl.f",
    "BZ=F": "brn.f",
    "NG=F": "ng.f",
    "RB=F": "rb.f",
    "HO=F": "ho.f",
    "GC=F": "gc.f",
    "SI=F": "si.f",
    "HG=F": "hg.f",
    "DX-Y.NYB": "dx.f",
    "EURUSD=X": "eurusd",
    "USDKRW=X": "usdkrw",
    "JPY=X": "usdjpy",
    "SPY": "spy.us",
    "QQQ": "qqq.us",
    "^GSPC": "^spx",
    "^VIX": "^vix",
    "XLE": "xle.us",
    "USO": "uso.us",
}


@dataclass(frozen=True)
class FetchResult:
    symbol: str
    path: Path | None
    rows: int
    status: str
    error: str | None = None


class MarketPriceProvider:
    provider_name = "base"

    def fetch(self, symbol: str, *, interval: str, period: str) -> pd.DataFrame:
        raise NotImplementedError


class YFinanceMarketPriceProvider(MarketPriceProvider):
    provider_name = "yfinance"

    def fetch(self, symbol: str, *, interval: str, period: str) -> pd.DataFrame:
        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}")
        import yfinance as yf

        data = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if data.empty:
            raise RuntimeError(f"yfinance returned no rows for {symbol} {interval} {period}")
        frame = data.reset_index()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in frame.columns]
        timestamp_col = "Date" if "Date" in frame.columns else "Datetime" if "Datetime" in frame.columns else None
        if timestamp_col is None:
            raise RuntimeError(f"yfinance output did not contain Date/Datetime for {symbol}")
        frame = frame.rename(
            columns={
                timestamp_col: "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        return normalize_market_price_frame(frame, symbol=symbol, provider=self.provider_name)


class CsvMarketPriceProvider(MarketPriceProvider):
    provider_name = "csv"

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)

    def fetch(self, symbol: str, *, interval: str, period: str = "") -> pd.DataFrame:
        del period
        candidates = [
            self.base_path / interval / f"{safe_symbol(symbol)}.csv",
            self.base_path / f"{safe_symbol(symbol)}.csv",
            self.base_path,
        ]
        path = next((candidate for candidate in candidates if candidate.exists() and candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"No CSV cache found for {symbol} under {self.base_path}")
        frame = read_table(path)
        return normalize_market_price_frame(frame, symbol=symbol, provider=self.provider_name)


class StooqMarketPriceProvider(MarketPriceProvider):
    provider_name = "stooq"

    def fetch(self, symbol: str, *, interval: str, period: str) -> pd.DataFrame:
        if interval != "1d":
            raise ValueError("Stooq public CSV fetch currently supports daily bars only.")
        provider_symbol = STOOQ_DAILY_SYMBOLS.get(symbol, symbol.lower())
        api_key = os.environ.get("STOOQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Stooq CSV download requires STOOQ_API_KEY in this environment.")
        url = _stooq_daily_url(provider_symbol, period=period, api_key=api_key)
        response = requests.get(
            url,
            headers={"User-Agent": "market-ai-data-collector/1.0"},
            timeout=15,
            verify=_requests_verify(),
        )
        response.raise_for_status()
        payload = response.text
        if payload.lstrip().startswith("Get your apikey"):
            raise RuntimeError("Stooq rejected CSV download; STOOQ_API_KEY is missing or invalid.")
        frame = pd.read_csv(StringIO(payload))
        if frame.empty or "Close" not in frame.columns:
            raise RuntimeError(f"Stooq returned no daily rows for {symbol} ({provider_symbol})")
        return normalize_market_price_frame(frame, symbol=symbol, provider=self.provider_name)


def normalize_market_price_frame(frame: pd.DataFrame, *, symbol: str, provider: str) -> pd.DataFrame:
    out = frame.copy()
    rename_map = {
        "date": "timestamp",
        "datetime": "timestamp",
        "time": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    out = out.rename(columns={col: rename_map[col] for col in out.columns if col in rename_map})
    if "timestamp" not in out.columns:
        raise ValueError("market price frame requires timestamp/date/time column")
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"market price frame missing columns: {missing}")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    for col in [*required, "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out["symbol"] = symbol
    out["provider"] = provider
    out["fetched_at"] = datetime.now(timezone.utc).isoformat()
    out = out.dropna(subset=["timestamp", *required]).sort_values("timestamp")
    out = out[out["close"] > 0.0]
    out = out.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    return out[["timestamp", "symbol", "open", "high", "low", "close", "volume", "provider", "fetched_at"]].reset_index(drop=True)


def _period_start(period: str, *, now: datetime | None = None) -> datetime | None:
    value = period.strip().lower()
    if value in {"", "max"}:
        return None
    now = now or datetime.now(timezone.utc)
    if value == "ytd":
        return datetime(now.year, 1, 1, tzinfo=timezone.utc)
    if value.endswith("mo"):
        try:
            return now - timedelta(days=31 * int(value[:-2]))
        except ValueError:
            return None
    unit = value[-1]
    try:
        count = int(value[:-1])
    except ValueError:
        return None
    if unit == "y":
        return now - timedelta(days=365 * count)
    if unit == "d":
        return now - timedelta(days=count)
    return None


def _stooq_daily_url(provider_symbol: str, *, period: str, api_key: str | None = None) -> str:
    params = {"s": provider_symbol, "i": "d"}
    start = _period_start(period)
    if start is not None:
        params["d1"] = start.strftime("%Y%m%d")
        params["d2"] = datetime.now(timezone.utc).strftime("%Y%m%d")
    if api_key:
        params["apikey"] = api_key
    return "https://stooq.com/q/d/l/?" + urlencode(params)


def _requests_verify() -> str | bool:
    try:
        import certifi

        return certifi.where()
    except Exception:
        return True


def write_market_cache(frame: pd.DataFrame, *, root: Path, provider: str, interval: str, symbol: str) -> Path:
    path = root / "raw" / "market" / provider / interval / f"{safe_symbol(symbol)}.csv"
    result = write_table(frame, path)
    return result.path


def load_market_cache(path: str | Path) -> pd.DataFrame:
    frame = read_table(path)
    symbol = str(frame["symbol"].iloc[0]) if "symbol" in frame.columns and len(frame) else "UNKNOWN"
    provider = str(frame["provider"].iloc[0]) if "provider" in frame.columns and len(frame) else "csv"
    return normalize_market_price_frame(frame, symbol=symbol, provider=provider)


def missing_bars_report(frame: pd.DataFrame, *, interval: str, symbol: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "interval", "missing_timestamp"])
    freq = INTERVAL_FREQ.get(interval)
    if freq is None:
        return pd.DataFrame(columns=["symbol", "interval", "missing_timestamp"])
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dropna().drop_duplicates().sort_values()
    if len(timestamps) < 2:
        return pd.DataFrame(columns=["symbol", "interval", "missing_timestamp"])
    expected = pd.date_range(timestamps.iloc[0], timestamps.iloc[-1], freq=freq, tz="UTC")
    missing = expected.difference(pd.DatetimeIndex(timestamps))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "interval": interval,
            "missing_timestamp": [ts.isoformat() for ts in missing],
        }
    )


def combine_market_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    rows = [frame for frame in frames if frame is not None and not frame.empty]
    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume", "provider", "fetched_at"])
    out = pd.concat(rows, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["timestamp", "symbol"]).sort_values(["symbol", "timestamp"])
    out = out.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    return out.reset_index(drop=True)
