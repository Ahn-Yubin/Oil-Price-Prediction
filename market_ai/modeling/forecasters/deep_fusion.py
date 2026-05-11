from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_ai.config import Settings, get_settings
from market_ai.constants import INTERVAL_TO_HORIZON
from market_ai.data.deep_dataset import combine_auxiliary_feature_frames
from market_ai.data.event_providers import FileEventProvider
from market_ai.data.storage import read_table
from market_ai.modeling.deep.availability import deep_artifact_availability
from market_ai.modeling.deep.inference import predict_deep_quantiles


DEEP_COLORS = {
    "deep_lstm_tcn_fusion": "#2dd4bf",
    "llm_context_seq_moe": "#f2cc60",
}

DEEP_LABELS = {
    "deep_lstm_tcn_fusion": "Deep LSTM+TCN Fusion",
    "llm_context_seq_moe": "LLM Context Seq MoE",
}


class DeepModelUnavailable(RuntimeError):
    pass


def _frame_from_close(close: np.ndarray) -> pd.DataFrame:
    close = np.asarray(close, dtype=np.float64)
    date = pd.date_range("2000-01-01", periods=len(close), freq="D", tz="UTC")
    open_ = np.r_[close[0], close[:-1]]
    span = np.maximum(np.abs(close - open_) * 0.4, close * 0.002)
    return pd.DataFrame(
        {
            "date": date,
            "open": open_,
            "high": np.maximum(open_, close) + span,
            "low": np.minimum(open_, close) - span,
            "close": close,
            "volume": 0.0,
        }
    )


@lru_cache(maxsize=16)
def _resolve_artifact_path(model_name: str, interval: str, horizon: int, model_dir: str, metadata_dir: str) -> Path:
    settings = get_settings().model_copy(update={"model_dir": Path(model_dir), "metadata_dir": Path(metadata_dir)})
    availability = deep_artifact_availability(settings=settings, model_name=model_name, interval=interval, horizon=horizon)
    if not availability.is_available:
        action = f"Run: {availability.training_command}"
        detail = availability.reason or f"Missing artifact file: {availability.expected_artifact_file}"
        raise DeepModelUnavailable(f"{detail} {action}")
    return availability.artifact_path


@lru_cache(maxsize=4)
def _load_processed_event_context(data_dir: str) -> pd.DataFrame | None:
    path = Path(data_dir) / "processed" / "event_context" / "event_context_daily.csv"
    if not path.exists():
        return None
    try:
        return read_table(path)
    except Exception:
        return None


@lru_cache(maxsize=4)
def _load_processed_market_panel(data_dir: str, interval: str) -> pd.DataFrame | None:
    root = Path(data_dir) / "processed" / "market_panel" / interval
    for name in ("panel.parquet", "panel.csv"):
        path = root / name
        if path.exists():
            try:
                return read_table(path)
            except Exception:
                return None
    return None


@lru_cache(maxsize=4)
def _load_processed_auxiliary(data_dir: str) -> pd.DataFrame | None:
    root = Path(data_dir) / "processed" / "oil_fundamentals"
    try:
        return combine_auxiliary_feature_frames(
            oil_fundamentals=read_table(root / "eia_weekly.csv") if (root / "eia_weekly.csv").exists() else None,
            cot=read_table(root / "cftc_cot_weekly.csv") if (root / "cftc_cot_weekly.csv").exists() else None,
            cme_curve=read_table(root / "cme_curve_daily.csv") if (root / "cme_curve_daily.csv").exists() else None,
        )
    except Exception:
        return None


def forecast_with_deep_model(
    *,
    model_name: str,
    close: np.ndarray,
    interval: str,
    horizon: int | None = None,
    settings: Settings | None = None,
    symbol: str = "UNKNOWN",
    candles: pd.DataFrame | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    model_horizon = int(horizon or INTERVAL_TO_HORIZON.get(interval, 45))
    try:
        artifact_path = _resolve_artifact_path(model_name, interval, model_horizon, str(settings.model_dir), str(settings.metadata_dir))
    except DeepModelUnavailable:
        raise
    except Exception as exc:
        raise DeepModelUnavailable(str(exc)) from exc
    frame = candles if candles is not None else _frame_from_close(close)
    event_context_frame = _load_processed_event_context(str(settings.data_dir)) if settings.enable_llm_context else None
    market_panel = _load_processed_market_panel(str(settings.data_dir), interval)
    auxiliary_frame = _load_processed_auxiliary(str(settings.data_dir))
    try:
        prediction = predict_deep_quantiles(
            artifact_path=artifact_path,
            candles=frame,
            symbol=symbol,
            interval=interval,
            horizon=model_horizon,
            event_provider=FileEventProvider.from_env() if settings.enable_llm_context else None,
            event_context_frame=event_context_frame,
            auxiliary_frame=auxiliary_frame,
            market_panel=market_panel,
            device="cpu",
        )
    except Exception as exc:
        raise DeepModelUnavailable(str(exc)) from exc
    p50 = np.asarray(prediction["quantile_prices"]["p50"], dtype=np.float64)
    return {
        "id": model_name,
        "label": DEEP_LABELS.get(model_name, model_name),
        "description": "Artifact-based volatility-scaled cumulative log-return deep model",
        "color": DEEP_COLORS.get(model_name, "#8b949e"),
        "values": p50,
        "quantile_prices": {key: np.asarray(value, dtype=np.float64) for key, value in prediction["quantile_prices"].items()},
        "prob_up": np.asarray(prediction["prob_up"], dtype=np.float64),
        "expected_volatility": np.asarray(prediction["expected_volatility"], dtype=np.float64),
        "confidence": np.asarray(prediction["confidence"], dtype=np.float64),
        "metadata": prediction["metadata"],
        "artifact_file": artifact_path.name,
    }
