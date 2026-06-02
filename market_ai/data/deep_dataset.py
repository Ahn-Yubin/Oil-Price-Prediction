from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from market_ai.data.event_providers import FileEventProvider, NullEventProvider
from market_ai.features.deep_features import (
    CROSS_ASSET_FEATURE_COLUMNS,
    DEEP_FEATURE_VERSION,
    PRICE_FEATURE_COLUMNS,
    build_deep_price_features,
    build_static_features,
    empty_cross_asset_window,
)
from market_ai.features.context_features import EVENT_CONTEXT_DIM
from market_ai.schemas.deep_learning import EventContextVector
from market_ai.schemas.deep_learning import DeepDatasetConfig, DeepLearningSample


@dataclass(frozen=True)
class DeepDataset:
    samples: list[DeepLearningSample]
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    price_feature_names: tuple[str, ...] = PRICE_FEATURE_COLUMNS
    cross_asset_feature_names: tuple[str, ...] = CROSS_ASSET_FEATURE_COLUMNS
    feature_version: str = DEEP_FEATURE_VERSION

    def tensors(self, indices: Iterable[int] | None = None) -> dict[str, np.ndarray]:
        idx = list(indices) if indices is not None else list(range(len(self.samples)))
        selected = [self.samples[i] for i in idx]
        return {
            "x_price": np.asarray([sample.x_price for sample in selected], dtype=np.float32),
            "x_cross_asset": np.asarray([sample.x_cross_asset for sample in selected], dtype=np.float32),
            "x_event_context": np.asarray([sample.x_event_context for sample in selected], dtype=np.float32),
            "x_static": np.asarray([sample.x_static for sample in selected], dtype=np.float32),
            "y_vol_scaled_cum_return": np.asarray([sample.y_vol_scaled_cum_return for sample in selected], dtype=np.float32),
            "y_direction": np.asarray([sample.y_direction for sample in selected], dtype=np.float32),
            "y_future_volatility": np.asarray([sample.y_future_volatility for sample in selected], dtype=np.float32),
        }


def sort_samples_chronologically(samples: list[DeepLearningSample]) -> list[DeepLearningSample]:
    return sorted(samples, key=lambda sample: (pd.Timestamp(sample.as_of_time), sample.symbol))


def _normalize_candles(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.copy()
    if "date" not in frame.columns:
        if "time" in frame.columns:
            frame["date"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        else:
            frame["date"] = pd.date_range("2000-01-01", periods=len(frame), freq="D", tz="UTC")
    frame["date"] = _utc_ns(pd.to_datetime(frame["date"], errors="coerce", utc=True))
    required = ["open", "high", "low", "close"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    for col in [*required, "volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame = frame.dropna(subset=["date", *required]).sort_values("date").reset_index(drop=True)
    frame = frame[frame["close"] > 0.0].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid positive-close candles")
    return frame


def _available_time_column(frame: pd.DataFrame) -> str:
    for col in ("feature_available_at", "as_of_time", "release_time", "timestamp", "date", "report_date", "trade_date"):
        if col in frame.columns:
            return col
    raise ValueError("feature frame requires feature_available_at/as_of_time/release_time/timestamp/date column")


def _normalize_feature_time_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    out = frame.copy()
    col = _available_time_column(out)
    out["feature_available_at"] = _utc_ns(pd.to_datetime(out[col], errors="coerce", utc=True))
    out = out.dropna(subset=["feature_available_at"]).sort_values("feature_available_at")
    return out.reset_index(drop=True)


def _utc_ns(values) -> pd.Series:
    return pd.Series(values).dt.tz_convert("UTC").astype("datetime64[ns, UTC]")


def _standardize(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    rolling = numeric.rolling(60, min_periods=8)
    return ((numeric - rolling.mean()) / rolling.std().replace(0.0, np.nan)).clip(-6.0, 6.0)


def combine_auxiliary_feature_frames(
    *,
    oil_fundamentals: pd.DataFrame | None = None,
    cot: pd.DataFrame | None = None,
    cme_curve: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    frames = []
    for source, frame in [("oil_fundamentals", oil_fundamentals), ("cot", cot), ("cme_curve", cme_curve), ("macro", macro)]:
        normalized = _normalize_feature_time_frame(frame)
        if normalized is None:
            continue
        normalized = normalized.copy()
        normalized["_source"] = source
        frames.append(normalized)
    if not frames:
        return None
    base = pd.DataFrame(
        {
            "feature_available_at": sorted(
                set(pd.concat([frame[["feature_available_at"]] for frame in frames], ignore_index=True)["feature_available_at"])
            )
        }
    )
    out = base.sort_values("feature_available_at")
    for frame in frames:
        suffix = str(frame["_source"].iloc[0])
        use = frame.drop(columns=["_source"]).sort_values("feature_available_at")
        out = pd.merge_asof(out, use, on="feature_available_at", direction="backward", suffixes=("", f"_{suffix}"))
    numeric_cols = [col for col in out.columns if col != "feature_available_at"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _first_numeric(row: pd.Series, candidates: tuple[str, ...], default: float = 0.0) -> float:
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            try:
                val = float(row[col])
            except (TypeError, ValueError):
                continue
            if np.isfinite(val):
                return val
    return default


def _build_auxiliary_cross_asset_matrix(frame: pd.DataFrame, symbol: str, auxiliary_frame: pd.DataFrame | None) -> np.ndarray:
    if auxiliary_frame is None or auxiliary_frame.empty:
        return empty_cross_asset_window(len(frame))
    left = pd.DataFrame({"date": _utc_ns(pd.to_datetime(frame["date"], errors="coerce", utc=True))})
    right = _normalize_feature_time_frame(auxiliary_frame)
    if right is None or right.empty:
        return empty_cross_asset_window(len(frame))
    merged = pd.merge_asof(left.sort_values("date"), right.sort_values("feature_available_at"), left_on="date", right_on="feature_available_at", direction="backward")
    arr = np.zeros((len(merged), len(CROSS_ASSET_FEATURE_COLUMNS)), dtype=np.float32)
    feature_cols = set(right.columns)
    has_feature = merged["feature_available_at"].notna().to_numpy()
    for idx, row in merged.iterrows():
        arr[idx, CROSS_ASSET_FEATURE_COLUMNS.index("spread")] = _first_numeric(
            row,
            ("m1_m2_spread", "imports_exports_spread", "crude_stocks_change", "cushing_stocks_change", "DCOILWTICO", "DCOILBRENTEU"),
        )
        arr[idx, CROSS_ASSET_FEATURE_COLUMNS.index("relative_strength")] = _first_numeric(
            row,
            ("managed_money_net_zscore", "managed_money_net", "refinery_utilization", "DGS10", "T10YIE"),
        )
        arr[idx, CROSS_ASSET_FEATURE_COLUMNS.index("risk_on_off_proxy")] = _first_numeric(
            row,
            ("curve_slope_m1_m6", "open_interest_change", "crude_production_change", "DTWEXBGS", "DEXKOUS", "DEXUSEU", "VIXCLS"),
        )
        arr[idx, CROSS_ASSET_FEATURE_COLUMNS.index("missing_indicator")] = 0.0 if has_feature[idx] else 1.0
    del symbol, feature_cols
    return np.nan_to_num(arr, nan=0.0, posinf=6.0, neginf=-6.0).astype(np.float32)


def _build_market_panel_cross_asset_matrix(frame: pd.DataFrame, symbol: str, market_panel: pd.DataFrame | None) -> np.ndarray | None:
    if market_panel is None or market_panel.empty:
        return None
    panel = market_panel.copy()
    if "timestamp" not in panel.columns or "symbol" not in panel.columns or "close" not in panel.columns:
        return None
    panel["timestamp"] = _utc_ns(pd.to_datetime(panel["timestamp"], errors="coerce", utc=True))
    panel["close"] = pd.to_numeric(panel["close"], errors="coerce")
    pivot = panel.dropna(subset=["timestamp", "symbol", "close"]).pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last").sort_index()
    if symbol not in pivot.columns or len(pivot.columns) < 2:
        return None
    returns = np.log(pivot / pivot.shift(1)).replace([np.inf, -np.inf], np.nan)
    others = [col for col in returns.columns if col != symbol]
    related_return = returns[others].mean(axis=1)
    corr_parts = [returns[symbol].rolling(20, min_periods=6).corr(returns[col]) for col in others]
    rolling_corr = pd.concat(corr_parts, axis=1).mean(axis=1) if corr_parts else pd.Series(0.0, index=returns.index)
    target = pd.DataFrame(
        {
            "feature_available_at": related_return.index,
            "related_returns": related_return.to_numpy(),
            "related_rolling_corr": rolling_corr.reindex(related_return.index).to_numpy(),
        }
    )
    left = pd.DataFrame({"date": _utc_ns(pd.to_datetime(frame["date"], errors="coerce", utc=True))})
    merged = pd.merge_asof(left.sort_values("date"), target.sort_values("feature_available_at"), left_on="date", right_on="feature_available_at", direction="backward")
    arr = np.zeros((len(merged), len(CROSS_ASSET_FEATURE_COLUMNS)), dtype=np.float32)
    arr[:, CROSS_ASSET_FEATURE_COLUMNS.index("related_returns")] = pd.to_numeric(merged["related_returns"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    arr[:, CROSS_ASSET_FEATURE_COLUMNS.index("related_rolling_corr")] = pd.to_numeric(merged["related_rolling_corr"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    arr[:, CROSS_ASSET_FEATURE_COLUMNS.index("missing_indicator")] = merged["feature_available_at"].isna().astype(float).to_numpy(dtype=np.float32)
    return arr


def _merge_cross_asset_matrices(aux: np.ndarray, market: np.ndarray | None) -> np.ndarray:
    if market is None:
        return aux
    out = aux.copy()
    for name in ("related_returns", "related_rolling_corr"):
        idx = CROSS_ASSET_FEATURE_COLUMNS.index(name)
        out[:, idx] = market[:, idx]
    missing_idx = CROSS_ASSET_FEATURE_COLUMNS.index("missing_indicator")
    out[:, missing_idx] = np.minimum(out[:, missing_idx], market[:, missing_idx])
    return out


def _event_context_frame_vector(
    event_context_frame: pd.DataFrame | None,
    *,
    symbol: str,
    as_of_time: datetime,
    normalized: bool = False,
) -> list[float] | None:
    frame = event_context_frame if normalized else _normalize_feature_time_frame(event_context_frame)
    if frame is None or frame.empty:
        return None
    if "symbol" in frame.columns:
        symbol_upper = symbol.upper()
        frame = frame[frame["symbol"].astype(str).str.upper().isin([symbol_upper, "ALL", "*"])]
    if frame.empty:
        return None
    ts = pd.Timestamp(as_of_time).tz_convert("UTC") if pd.Timestamp(as_of_time).tzinfo else pd.Timestamp(as_of_time, tz="UTC")
    left = pd.DataFrame({"date": _utc_ns(pd.to_datetime([ts], utc=True))})
    merged = pd.merge_asof(left, frame.sort_values("feature_available_at"), left_on="date", right_on="feature_available_at", direction="backward")
    if merged.empty or pd.isna(merged.loc[0, "feature_available_at"]):
        return None
    names = list(EventContextVector.model_fields)
    values = [float(pd.to_numeric(pd.Series([merged.loc[0, name] if name in merged.columns else 0.0]), errors="coerce").fillna(0.0).iloc[0]) for name in names]
    if len(values) < EVENT_CONTEXT_DIM:
        values.extend([0.0] * (EVENT_CONTEXT_DIM - len(values)))
    return values[:EVENT_CONTEXT_DIM]


def _event_context_frame_window_vector(
    event_context_frame: pd.DataFrame | None,
    *,
    symbol: str,
    start_time: datetime,
    as_of_time: datetime,
    normalized: bool = False,
) -> list[float] | None:
    frame = event_context_frame if normalized else _normalize_feature_time_frame(event_context_frame)
    if frame is None or frame.empty:
        return None
    if "symbol" in frame.columns:
        symbol_upper = symbol.upper()
        frame = frame[frame["symbol"].astype(str).str.upper().isin([symbol_upper, "ALL", "*"])]
    if frame.empty:
        return None
    start_ts = pd.Timestamp(start_time).tz_convert("UTC") if pd.Timestamp(start_time).tzinfo else pd.Timestamp(start_time, tz="UTC")
    end_ts = pd.Timestamp(as_of_time).tz_convert("UTC") if pd.Timestamp(as_of_time).tzinfo else pd.Timestamp(as_of_time, tz="UTC")
    window = frame[(frame["feature_available_at"] >= start_ts) & (frame["feature_available_at"] <= end_ts)].copy()
    if window.empty:
        return _event_context_frame_vector(
            frame,
            symbol=symbol,
            as_of_time=as_of_time,
            normalized=True,
        )
    names = list(EventContextVector.model_fields)
    weight_col = "impact_strength" if "impact_strength" in window.columns else "impact_score" if "impact_score" in window.columns else ""
    if weight_col:
        weights = pd.to_numeric(window[weight_col], errors="coerce").fillna(1.0).clip(lower=0.05)
    else:
        weights = pd.Series(1.0, index=window.index, dtype=float)
    values: list[float] = []
    for name in names:
        series = pd.to_numeric(window[name], errors="coerce").fillna(0.0) if name in window.columns else pd.Series(0.0, index=window.index)
        if name.startswith("event_count"):
            values.append(float(np.clip(series.sum(), 0.0, 12.0)))
        elif name == "uncertainty":
            values.append(float(np.clip(series.tail(20).mean(), 0.0, 1.0)))
        elif name == "time_decay":
            values.append(float(np.clip(series.max(), 0.0, 1.0)))
        else:
            weighted = float((series * weights).sum() / max(float(weights.sum()), 1e-8))
            values.append(weighted)
    if len(values) < EVENT_CONTEXT_DIM:
        values.extend([0.0] * (EVENT_CONTEXT_DIM - len(values)))
    return values[:EVENT_CONTEXT_DIM]


def _future_volatility(returns: np.ndarray, start: int, horizon: int) -> np.ndarray:
    vals: list[float] = []
    for h in range(1, horizon + 1):
        segment = returns[start : start + h]
        vals.append(max(float(np.std(segment)) if len(segment) > 1 else 0.0, 0.0))
    return np.asarray(vals, dtype=np.float32)


def _time_split_indices(n: int, validation_ratio: float, test_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty, empty
    test_n = int(np.floor(n * test_ratio))
    val_n = int(np.floor(n * validation_ratio))
    if n >= 3:
        test_n = max(1, test_n)
        val_n = max(1, val_n)
    train_n = max(n - val_n - test_n, 0)
    if train_n == 0 and n > 0:
        train_n = max(1, n - val_n - test_n)
    train_end = min(train_n, n)
    val_end = min(train_end + val_n, n)
    return (
        np.arange(0, train_end, dtype=np.int64),
        np.arange(train_end, val_end, dtype=np.int64),
        np.arange(val_end, n, dtype=np.int64),
    )


def build_deep_dataset_from_frame(
    *,
    symbol: str,
    interval: str,
    candles: pd.DataFrame,
    config: DeepDatasetConfig,
    event_provider: FileEventProvider | None = None,
    auxiliary_frame: pd.DataFrame | None = None,
    market_panel: pd.DataFrame | None = None,
    event_context_frame: pd.DataFrame | None = None,
) -> DeepDataset:
    frame = _normalize_candles(candles)
    provider = event_provider or (FileEventProvider.from_env() if config.event_context_enabled else NullEventProvider())
    features = build_deep_price_features(frame)
    auxiliary_cross_asset = _build_auxiliary_cross_asset_matrix(frame, symbol, auxiliary_frame)
    market_cross_asset = _build_market_panel_cross_asset_matrix(frame, symbol, market_panel)
    cross_asset_matrix = _merge_cross_asset_matrices(auxiliary_cross_asset, market_cross_asset)
    close = frame["close"].to_numpy(dtype=np.float64)
    log_close = np.log(close)
    returns_by_bar = np.zeros(len(close), dtype=np.float64)
    returns_by_bar[1:] = np.diff(log_close)
    lookback = int(config.lookback)
    horizon = int(config.horizon)
    min_history = max(int(config.min_history), lookback)
    samples: list[DeepLearningSample] = []

    last_origin = len(frame) - horizon - 1
    first_origin = min_history - 1
    if config.max_samples is not None:
        first_origin = max(first_origin, last_origin - int(config.max_samples) + 1)
    normalized_event_context = _normalize_feature_time_frame(event_context_frame)
    for origin in range(first_origin, last_origin + 1):
        feature_window = features.iloc[origin - lookback + 1 : origin + 1]
        if len(feature_window) != lookback:
            continue
        current_price = float(close[origin])
        recent = returns_by_bar[max(1, origin - 60 + 1) : origin + 1]
        recent_vol = max(float(np.std(recent)) if len(recent) > 1 else 0.0, 1e-8)
        future_log_path = log_close[origin + 1 : origin + horizon + 1] - log_close[origin]
        scaled_target = np.clip(future_log_path / recent_vol, -12.0, 12.0).astype(np.float32)
        future_returns = returns_by_bar[origin + 1 : origin + horizon + 1]
        as_of_time = pd.Timestamp(frame.loc[origin, "date"]).to_pydatetime()
        window_start_time = pd.Timestamp(frame.loc[origin - lookback + 1, "date"]).to_pydatetime()
        event_vector = _event_context_frame_window_vector(
            normalized_event_context,
            symbol=symbol,
            start_time=window_start_time,
            as_of_time=as_of_time,
            normalized=True,
        )
        if event_vector is None:
            event_vector = provider.context_vector(symbol=symbol, as_of_time=as_of_time).as_list()
        samples.append(
            DeepLearningSample(
                symbol=symbol,
                interval=interval,
                as_of_time=as_of_time,
                lookback=lookback,
                horizon=horizon,
                x_price=feature_window[list(PRICE_FEATURE_COLUMNS)].to_numpy(dtype=np.float32).tolist(),
                x_cross_asset=cross_asset_matrix[origin - lookback + 1 : origin + 1].astype(np.float32).tolist(),
                x_event_context=[float(x) for x in event_vector],
                x_static=build_static_features(
                    current_price=current_price,
                    recent_realized_volatility=recent_vol,
                    lookback=lookback,
                    horizon=horizon,
                ).tolist(),
                y_vol_scaled_cum_return=scaled_target.tolist(),
                y_direction=(future_log_path > 0.0).astype(np.int64).tolist(),
                y_future_volatility=_future_volatility(future_returns, 0, horizon).tolist(),
                current_price=current_price,
                recent_realized_volatility=recent_vol,
                feature_version=DEEP_FEATURE_VERSION,
                data_status={"status": "real", "source": "dataframe"},
            )
        )

    if config.max_samples is not None and len(samples) > config.max_samples:
        samples = samples[-int(config.max_samples) :]

    train_idx, val_idx, test_idx = _time_split_indices(len(samples), config.validation_ratio, config.test_ratio)
    return DeepDataset(samples=samples, train_indices=train_idx, validation_indices=val_idx, test_indices=test_idx)


def synthetic_ohlcv(rows: int = 420, *, seed: int = 42, start: str = "2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.arange(rows, dtype=np.float64)
    noise = rng.normal(0.0, 0.004, size=rows)
    returns = 0.0003 + 0.006 * np.sin(x / 18.0) + noise
    close = 75.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(np.abs(close - open_) * 0.6, close * 0.006)
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=rows, freq="D", tz="UTC"),
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": 10_000 + 500 * np.sin(x / 7.0) + rng.normal(0.0, 50.0, size=rows),
        }
    )


def build_synthetic_deep_dataset(config: DeepDatasetConfig) -> DeepDataset:
    frames = [synthetic_ohlcv(max(config.min_history + config.horizon + 80, 260), seed=config.seed + idx) for idx, _ in enumerate(config.symbols)]
    all_samples: list[DeepLearningSample] = []
    for symbol, frame in zip(config.symbols, frames):
        ds = build_deep_dataset_from_frame(symbol=symbol, interval=config.interval, candles=frame, config=config, event_provider=NullEventProvider())
        all_samples.extend(ds.samples)
    all_samples = sort_samples_chronologically(all_samples)
    if config.max_samples is not None and len(all_samples) > config.max_samples:
        all_samples = all_samples[-int(config.max_samples) :]
    train_idx, val_idx, test_idx = _time_split_indices(len(all_samples), config.validation_ratio, config.test_ratio)
    return DeepDataset(samples=all_samples, train_indices=train_idx, validation_indices=val_idx, test_indices=test_idx)
