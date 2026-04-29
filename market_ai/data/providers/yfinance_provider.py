from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from market_ai.config import Settings, get_settings
from market_ai.constants import INTERVAL_TO_PERIOD_CANDIDATES
from market_ai.schemas.market import Candle, DataStatus, DataStatusKind, MarketDataWindow, MarketSymbol, Timeframe
from market_ai.data.symbols import normalize_symbol, symbol_candidates
from market_ai.data.timeframes import normalize_timeframe


class MarketDataUnavailable(RuntimeError):
    def __init__(self, message: str, data_status: DataStatus | None = None):
        super().__init__(message)
        self.data_status = data_status


def to_unix_seconds(dt_value) -> int:
    parsed = pd.to_datetime(dt_value, errors="coerce", utc=True)
    if isinstance(parsed, pd.Series):
        parsed = parsed.dropna()
        if parsed.empty:
            raise ValueError("Invalid datetime series")
        parsed = parsed.iloc[0]
    if isinstance(parsed, pd.DatetimeIndex):
        if len(parsed) == 0:
            raise ValueError("Empty datetime index")
        parsed = parsed[0]
    if pd.isna(parsed):
        raise ValueError(f"Invalid datetime value: {dt_value}")
    return int(pd.Timestamp(parsed).timestamp())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candle_frame_to_models(frame: pd.DataFrame) -> list[Candle]:
    candles: list[Candle] = []
    for _, row in frame.iterrows():
        volume = row.get("volume")
        candles.append(
            Candle(
                time=to_unix_seconds(row["date"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(volume) if volume is not None and pd.notna(volume) else None,
            )
        )
    return candles


def _status_for_frame(
    *,
    status: DataStatusKind,
    source: str,
    symbol: MarketSymbol,
    timeframe: Timeframe,
    frame: pd.DataFrame,
    settings: Settings,
    warnings: list[str] | None = None,
) -> DataStatus:
    warnings = list(warnings or [])
    last_bar = pd.to_datetime(frame["date"].iloc[-1], utc=True) if not frame.empty else None
    is_stale = False
    resolved_status = status
    if last_bar is not None and status == DataStatusKind.real:
        age = datetime.now(timezone.utc) - last_bar.to_pydatetime()
        is_stale = age.total_seconds() > settings.data_stale_threshold_seconds
        if is_stale:
            resolved_status = DataStatusKind.stale
            warnings.append(f"Last bar is older than {settings.data_stale_threshold_seconds} seconds.")
    return DataStatus(
        status=resolved_status,
        source=source,
        symbol_requested=symbol.requested,
        symbol_resolved=symbol.provider_symbol,
        interval_requested=timeframe.requested,
        interval_resolved=timeframe.normalized,
        last_bar_time=last_bar.isoformat() if last_bar is not None else None,
        updated_at=_utc_now_iso(),
        is_stale=is_stale,
        warnings=warnings,
    )


def _normalize_yfinance_frame(data: pd.DataFrame) -> pd.DataFrame | None:
    if data.empty:
        return None
    frame = data.reset_index()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in frame.columns]

    date_col = "Date" if "Date" in frame.columns else "Datetime" if "Datetime" in frame.columns else None
    if not date_col:
        return None

    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col])
    if frame.empty:
        return None

    renamed = frame.rename(
        columns={
            date_col: "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    required = ["date", "open", "high", "low", "close"]
    if not set(required).issubset(renamed.columns):
        return None
    cols = required + (["volume"] if "volume" in renamed.columns else [])
    out = renamed[cols].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required)
    out = out[out["close"] > 0.0]
    if out.empty:
        return None
    return out.sort_values("date").reset_index(drop=True)


def _download_ohlc(provider_symbol: str, timeframe: Timeframe) -> pd.DataFrame:
    periods = INTERVAL_TO_PERIOD_CANDIDATES.get(timeframe.normalized, [timeframe.provider_period])
    last_error: Exception | None = None
    for period in periods:
        try:
            data = yf.download(
                provider_symbol,
                period=period,
                interval=timeframe.provider_interval,
                auto_adjust=False,
                progress=False,
            )
        except Exception as exc:
            last_error = exc
            continue
        frame = _normalize_yfinance_frame(data)
        if frame is not None:
            return frame
    if last_error is not None:
        raise ValueError(str(last_error)) from last_error
    raise ValueError(f"No market data for symbol: {provider_symbol}")


def _mock_ohlc_frame(timeframe: Timeframe, rows: int = 180) -> pd.DataFrame:
    step = timedelta(seconds=timeframe.seconds)
    end = datetime.now(timezone.utc)
    dates = [end - step * (rows - i - 1) for i in range(rows)]
    base = np.arange(rows, dtype=float)
    close = 72.0 + 0.07 * base + 2.5 * np.sin(base / 9.0)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    span = np.maximum(np.abs(close - open_) * 0.3, 0.6)
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.zeros(rows, dtype=float),
        }
    )


def _fallback_ohlc_frame(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    required = {"date", "open", "high", "low", "close"}
    if required.issubset(frame.columns):
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"])
        if "volume" not in frame.columns:
            frame["volume"] = 0.0
        return frame.sort_values("date").reset_index(drop=True)
    return None


def _fallback_predictions_to_ohlc(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    required = {"date", "actual", "predicted"}
    if not required.issubset(frame.columns):
        return None
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
    frame = frame.dropna(subset=["date", "actual"]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        return None
    close = frame["actual"].astype(float).reset_index(drop=True)
    open_ = close.shift(1).fillna(close.iloc[0])
    span = np.maximum(np.abs(close - open_) * 0.3, 0.6)
    return pd.DataFrame(
        {
            "date": frame["date"],
            "open": open_,
            "high": np.maximum(open_, close) + span,
            "low": np.minimum(open_, close) - span,
            "close": close,
            "volume": 0.0,
        }
    )


def _load_development_fallback(
    *,
    symbol: MarketSymbol,
    timeframe: Timeframe,
    settings: Settings,
    reason: str,
) -> MarketDataWindow:
    warnings = [f"Live data fetch failed: {reason}"]
    frame = _fallback_ohlc_frame(settings.baseline_ohlc_path)
    source = "baseline-ohlc-fallback"
    status = DataStatusKind.fallback
    if frame is None:
        frame = _fallback_predictions_to_ohlc(settings.baseline_predictions_path)
        source = "baseline-predictions-fallback"
    if frame is None:
        frame = _mock_ohlc_frame(timeframe)
        source = "mock"
        status = DataStatusKind.mock
        warnings.append("Using deterministic development mock data.")

    data_status = _status_for_frame(
        status=status,
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        frame=frame,
        settings=settings,
        warnings=warnings,
    )
    return MarketDataWindow(
        symbol=symbol,
        timeframe=timeframe,
        candles=_candle_frame_to_models(frame),
        data_status=data_status,
    )


def load_market_data_window(
    raw_symbol: str,
    raw_interval: str,
    *,
    settings: Settings | None = None,
) -> MarketDataWindow:
    settings = settings or get_settings()
    timeframe = normalize_timeframe(raw_interval, settings)
    requested_symbol = normalize_symbol(raw_symbol, default_symbol=settings.default_symbol)
    warnings = [timeframe.warning] if timeframe.warning else []

    errors: list[str] = []
    for candidate in symbol_candidates(requested_symbol.requested, default_symbol=settings.default_symbol):
        candidate_symbol = normalize_symbol(candidate, default_symbol=settings.default_symbol)
        try:
            frame = _download_ohlc(candidate_symbol.provider_symbol, timeframe)
            data_status = _status_for_frame(
                status=DataStatusKind.real,
                source="yfinance",
                symbol=candidate_symbol.model_copy(update={"requested": requested_symbol.requested}),
                timeframe=timeframe,
                frame=frame,
                settings=settings,
                warnings=warnings,
            )
            return MarketDataWindow(
                symbol=candidate_symbol.model_copy(update={"requested": requested_symbol.requested}),
                timeframe=timeframe,
                candles=_candle_frame_to_models(frame),
                data_status=data_status,
            )
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    reason = " | ".join(errors[:4]) if errors else "No candidate symbol worked"
    error_status = DataStatus(
        status=DataStatusKind.error,
        source="yfinance",
        symbol_requested=requested_symbol.requested,
        symbol_resolved=requested_symbol.provider_symbol,
        interval_requested=timeframe.requested,
        interval_resolved=timeframe.normalized,
        last_bar_time=None,
        updated_at=_utc_now_iso(),
        is_stale=False,
        warnings=warnings + [reason],
    )
    if not settings.mock_data_enabled:
        raise MarketDataUnavailable(
            f"Market data unavailable for '{requested_symbol.requested}' and mock fallback is disabled.",
            data_status=error_status,
        )

    return _load_development_fallback(
        symbol=requested_symbol,
        timeframe=timeframe,
        settings=settings,
        reason=reason,
    )
