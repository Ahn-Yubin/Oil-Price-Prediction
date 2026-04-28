from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from app.global_dl_model import (
    PretrainedModelNotFoundError,
    forecast_with_global_model,
)


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
BASELINE_OUTPUT_DIR = PROJECT_DIR.parent / "oil-price-baseline" / "outputs"
SYMBOL_MAP = {
    "NYMEX:CL1!": "CL=F",
    "TVC:USOIL": "CL=F",
    "ICEEUR:BRN1!": "BZ=F",
    "TVC:UKOIL": "BZ=F",
    "FX_IDC:USDKRW": "USDKRW=X",
    "TVC:DXY": "DX-Y.NYB",
    "OANDA:XAUUSD": "GC=F",
}
ALLOWED_INTERVALS = {"1d", "1h", "30m", "15m"}
INTERVAL_TO_PERIOD = {
    "1d": "2y",
    "1h": "180d",
    "30m": "60d",
    "15m": "30d",
}
INTERVAL_TO_PERIOD_CANDIDATES = {
    "1d": ["2y", "1y", "6mo"],
    "1h": ["180d", "120d", "90d"],
    "30m": ["60d", "30d", "14d"],
    "15m": ["60d", "30d", "14d"],
}
INTERVAL_TO_HORIZON = {
    "1d": 45,
    "1h": 72,
    "30m": 120,
    "15m": 192,
}
INTERVAL_TO_RETURN_CLIP = {
    "1d": 0.08,
    "1h": 0.03,
    "30m": 0.02,
    "15m": 0.015,
}
INTERVAL_TO_MAX_LOG_BAND = {
    "1d": 0.38,
    "1h": 0.22,
    "30m": 0.16,
    "15m": 0.12,
}
CONFIDENCE_Z = 1.96

app = FastAPI(title="Oil Forecast Dashboard", version="0.1.0")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def _to_unix_seconds(dt_value: str) -> int:
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


def _mock_frame() -> pd.DataFrame:
    dates = pd.date_range(datetime.now() - timedelta(days=180), periods=180, freq="D")
    base = np.arange(180, dtype=float)
    actual = 72 + 0.07 * base + 2.5 * np.sin(base / 9.0)
    predicted = actual + np.random.default_rng(42).normal(0, 0.9, size=len(actual))
    return pd.DataFrame({"date": dates, "actual": actual, "predicted": predicted})


def _load_predictions() -> pd.DataFrame:
    pred_path = BASELINE_OUTPUT_DIR / "predictions.csv"
    if not pred_path.exists():
        return _mock_frame()

    frame = pd.read_csv(pred_path)
    required = {"date", "actual", "predicted"}
    if not required.issubset(set(frame.columns)):
        return _mock_frame()
    return frame


def _load_metrics() -> dict:
    metrics_path = BASELINE_OUTPUT_DIR / "metrics.json"
    if not metrics_path.exists():
        return {"mae": None, "rmse": None, "mape": None, "note": "Using mock data"}

    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_ohlc_or_build(frame: pd.DataFrame) -> pd.DataFrame:
    ohlc_path = BASELINE_OUTPUT_DIR / "ohlc.csv"
    if ohlc_path.exists():
        ohlc = pd.read_csv(ohlc_path)
        required = {"date", "open", "high", "low", "close"}
        if required.issubset(set(ohlc.columns)):
            ohlc["date"] = pd.to_datetime(ohlc["date"])
            return ohlc.sort_values("date").reset_index(drop=True)

    close = frame["actual"].astype(float).reset_index(drop=True)
    open_ = close.shift(1).fillna(close.iloc[0])
    span = np.maximum(np.abs(close - open_) * 0.3, 0.6)
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    return pd.DataFrame(
        {
            "date": frame["date"].reset_index(drop=True),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def _build_symbol_candidates(raw_symbol: str) -> list[str]:
    raw = (raw_symbol or "NYMEX:CL1!").strip()
    if not raw:
        raw = "NYMEX:CL1!"
    upper = raw.upper()

    candidates: list[str] = []
    if upper in SYMBOL_MAP:
        candidates.append(SYMBOL_MAP[upper])

    # TV style -> right token
    right = upper.split(":", 1)[1] if ":" in upper else upper
    candidates.append(right)

    # common futures aliases
    futures_alias = {
        "CL1!": "CL=F",
        "USOIL": "CL=F",
        "UKOIL": "BZ=F",
        "BRN1!": "BZ=F",
        "XAUUSD": "GC=F",
    }
    if right in futures_alias:
        candidates.append(futures_alias[right])

    # forex pair -> pair=X
    if re.fullmatch(r"[A-Z]{6}", right):
        candidates.append(f"{right}=X")

    # keep original (user may already input yfinance ticker)
    candidates.append(raw)
    candidates.append(upper)

    unique = []
    seen = set()
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _normalize_interval(raw_interval: str) -> str:
    interval = (raw_interval or "1d").strip().lower()
    return interval if interval in ALLOWED_INTERVALS else "1d"


def _download_ohlc(symbol: str, interval: str) -> pd.DataFrame:
    candidates = INTERVAL_TO_PERIOD_CANDIDATES.get(interval, [INTERVAL_TO_PERIOD.get(interval, "2y")])
    for period in candidates:
        data = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            continue
        frame = data.reset_index()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in frame.columns]

        date_col = "Date" if "Date" in frame.columns else "Datetime" if "Datetime" in frame.columns else None
        if not date_col:
            continue

        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame = frame.dropna(subset=[date_col])
        if frame.empty:
            continue
        return frame.rename(
            columns={
                date_col: "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )[["date", "open", "high", "low", "close", "volume"]]
    raise ValueError(f"No market data for symbol: {symbol}")


def _download_ohlc_with_retry(raw_symbol: str, interval: str) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for candidate in _build_symbol_candidates(raw_symbol):
        try:
            frame = _download_ohlc(candidate, interval)
            return frame, candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            continue
    raise ValueError(" | ".join(errors[:4]) if errors else "No candidate symbol worked")


def _return_feature_vector(return_history: np.ndarray, lags: int) -> np.ndarray:
    if len(return_history) < lags:
        padded = np.zeros(lags, dtype=float)
        if len(return_history) > 0:
            padded[: len(return_history)] = return_history[::-1]
        lag_vec = padded
    else:
        lag_vec = return_history[-lags:][::-1]

    mom_3 = float(np.sum(return_history[-3:])) if len(return_history) >= 3 else float(np.sum(return_history))
    mom_8 = float(np.sum(return_history[-8:])) if len(return_history) >= 8 else float(np.sum(return_history))
    mom_21 = float(np.sum(return_history[-21:])) if len(return_history) >= 21 else float(np.sum(return_history))
    vol_5 = float(np.std(return_history[-5:])) if len(return_history) >= 5 else float(np.std(return_history))
    vol_13 = float(np.std(return_history[-13:])) if len(return_history) >= 13 else float(np.std(return_history))
    vol_34 = float(np.std(return_history[-34:])) if len(return_history) >= 34 else float(np.std(return_history))
    last = float(lag_vec[0]) if len(lag_vec) > 0 else 0.0
    second = float(lag_vec[1]) if len(lag_vec) > 1 else 0.0
    accel = last - second
    abs_mean_10 = (
        float(np.mean(np.abs(return_history[-10:])))
        if len(return_history) >= 10
        else float(np.mean(np.abs(return_history))) if len(return_history) else 0.0
    )

    x = np.concatenate(
        [
            np.array([1.0]),
            lag_vec,
            np.array(
                [
                    mom_3,
                    mom_8,
                    mom_21,
                    vol_5,
                    vol_13,
                    vol_34,
                    last**2,
                    last * second,
                    accel,
                    abs_mean_10,
                ]
            ),
        ]
    )
    return x


def _fit_ridge(X: np.ndarray, y: np.ndarray, ridge: float = 5e-2) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    xtx = X.T @ X
    xty = X.T @ y
    reg = ridge * np.eye(xtx.shape[0], dtype=float)
    reg[0, 0] = 0.0  # keep intercept unbiased
    try:
        beta = np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(xtx + reg, xty, rcond=None)
    return beta


def _fit_direct_models(
    returns: np.ndarray, horizon: int, lags: int
) -> list[tuple[np.ndarray, float] | None]:
    models: list[tuple[np.ndarray, float] | None] = []
    n = len(returns)
    for h in range(1, horizon + 1):
        rows = []
        targets = []
        for t in range(lags, n - h + 1):
            hist = returns[:t]
            rows.append(_return_feature_vector(hist, lags))
            targets.append(float(np.sum(returns[t : t + h])))

        if len(rows) < max(30, lags + 8):
            models.append(None)
            continue

        X = np.asarray(rows, dtype=float)
        y = np.asarray(targets, dtype=float)
        beta = _fit_ridge(X, y, ridge=7e-2)
        resid = y - (X @ beta)
        tail = resid[-min(len(resid), 180) :]
        resid_std = float(np.std(tail)) if len(tail) else 0.0
        models.append((beta, max(resid_std, 1e-4)))
    return models


def _forecast_with_interval(
    close: np.ndarray,
    interval: str,
    horizon: int,
    lags: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(close) < lags + 25:
        flat = np.repeat(close[-1], horizon)
        return flat, flat, flat

    returns = np.diff(np.log(close.astype(float)))
    models = _fit_direct_models(returns, horizon=horizon, lags=lags)

    clip = INTERVAL_TO_RETURN_CLIP.get(interval, 0.05)
    max_log_band = INTERVAL_TO_MAX_LOG_BAND.get(interval, 0.25)
    hist_returns = returns.astype(float)
    base_price = float(close[-1])
    base_feat = _return_feature_vector(hist_returns, lags)
    recent_drift = (
        float(np.mean(hist_returns[-12:])) if len(hist_returns) >= 12 else float(np.mean(hist_returns))
    )
    recent_vol = float(np.std(hist_returns[-34:])) if len(hist_returns) >= 34 else float(np.std(hist_returns))

    mean_out = []
    lower_out = []
    upper_out = []
    for step_idx in range(1, horizon + 1):
        model = models[step_idx - 1]
        if model is None:
            cum_log_ret = recent_drift * step_idx
            resid_std = max(recent_vol, 1e-4)
        else:
            beta_h, resid_std = model
            cum_log_ret = float(np.dot(base_feat, beta_h))
            # Blend with recent drift to prevent abrupt regime flips.
            cum_log_ret = 0.85 * cum_log_ret + 0.15 * (recent_drift * step_idx)

        clip_h = clip * np.sqrt(step_idx)
        cum_log_ret = float(np.clip(cum_log_ret, -clip_h, clip_h))
        log_band = float(min(CONFIDENCE_Z * resid_std, max_log_band))

        mean_price = float(base_price * np.exp(cum_log_ret))
        lower_price = float(base_price * np.exp(cum_log_ret - log_band))
        upper_price = float(base_price * np.exp(cum_log_ret + log_band))
        mean_out.append(mean_price)
        lower_out.append(lower_price)
        upper_out.append(upper_price)

    return (
        np.asarray(mean_out, dtype=float),
        np.asarray(lower_out, dtype=float),
        np.asarray(upper_out, dtype=float),
    )


def _backtest_metrics(close: np.ndarray, interval: str, lags: int = 12, test_size: int = 40) -> dict:
    returns = np.diff(np.log(close.astype(float)))
    if len(returns) < lags + test_size + 10:
        return {"mae": None, "rmse": None, "mape": None}

    rows = []
    targets = []
    return_idx = []
    for t in range(lags, len(returns)):
        rows.append(_return_feature_vector(returns[:t], lags))
        targets.append(returns[t])
        return_idx.append(t)

    X = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    split = max(1, len(X) - test_size)
    X_train, y_train = X[:split], y[:split]
    X_test, idx_test = X[split:], return_idx[split:]
    if len(X_train) < 20 or len(X_test) == 0:
        return {"mae": None, "rmse": None, "mape": None}

    beta = _fit_ridge(X_train, y_train, ridge=5e-2)
    pred_ret = X_test @ beta
    clip = INTERVAL_TO_RETURN_CLIP.get(interval, 0.05)
    pred_ret = np.clip(pred_ret, -clip, clip)

    # return index t predicts close[t+1] from close[t]
    prev_close = np.asarray([close[t] for t in idx_test], dtype=float)
    y_true_arr = np.asarray([close[t + 1] for t in idx_test], dtype=float)
    y_pred_arr = prev_close * np.exp(pred_ret)
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
    mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / (np.abs(y_true_arr) + 1e-8))) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/chart")
def chart_data(symbol: str = "NYMEX:CL1!", interval: str = "1d"):
    input_symbol = (symbol or "NYMEX:CL1!").strip()
    resolved_interval = _normalize_interval(interval)
    try:
        market, resolved_symbol = _download_ohlc_with_retry(input_symbol, resolved_interval)
        market = market.sort_values("date").reset_index(drop=True)
        close = market["close"].to_numpy(dtype=float)

        candles = [
            {
                "time": _to_unix_seconds(row["date"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
            for _, row in market.iterrows()
        ]

        horizon = INTERVAL_TO_HORIZON.get(resolved_interval, 14)
        try:
            pred_mean, pred_low, pred_high, model_info = forecast_with_global_model(
                close=close,
                interval=resolved_interval,
                horizon=horizon,
                z_value=CONFIDENCE_Z,
                return_clip=INTERVAL_TO_RETURN_CLIP.get(resolved_interval, 0.05),
                max_log_band=INTERVAL_TO_MAX_LOG_BAND.get(resolved_interval, 0.25),
            )
        except PretrainedModelNotFoundError as model_exc:
            raise HTTPException(status_code=503, detail=str(model_exc)) from model_exc
        start_date = pd.to_datetime(market["date"].iloc[-1])
        last_close = float(market["close"].iloc[-1])
        delta_map = {
            "1d": timedelta(days=1),
            "1h": timedelta(hours=1),
            "30m": timedelta(minutes=30),
            "15m": timedelta(minutes=15),
        }
        step = delta_map.get(resolved_interval, timedelta(days=1))
        future_dates = [start_date + step * (i + 1) for i in range(horizon)]
        predicted = [{"time": _to_unix_seconds(start_date), "value": last_close}] + [
            {"time": _to_unix_seconds(dt), "value": float(val)}
            for dt, val in zip(future_dates, pred_mean)
        ]
        predicted_lower = [{"time": _to_unix_seconds(start_date), "value": last_close}] + [
            {"time": _to_unix_seconds(dt), "value": float(val)}
            for dt, val in zip(future_dates, pred_low)
        ]
        predicted_upper = [{"time": _to_unix_seconds(start_date), "value": last_close}] + [
            {"time": _to_unix_seconds(dt), "value": float(val)}
            for dt, val in zip(future_dates, pred_high)
        ]

        metrics = {
            "mae": model_info.get("val_mae_ret"),
            "rmse": model_info.get("val_rmse_ret"),
            "mape": None,
            "symbol": resolved_symbol,
            "model": model_info.get("model_name", "Global DL model"),
        }

        return {
            "candles": candles,
            "predicted": predicted,
            "predicted_lower": predicted_lower,
            "predicted_upper": predicted_upper,
            "metrics": metrics,
            "symbol_input": input_symbol,
            "symbol_resolved": resolved_symbol,
            "interval_resolved": resolved_interval,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "data_source": "yfinance",
            "warning": None,
            "forecast_horizon": horizon,
            "confidence_level": 0.95,
            "model_trained_at": model_info.get("trained_at"),
            "model_train_symbols": model_info.get("train_symbols"),
            "model_sample_info": {
                "n_train": model_info.get("n_train"),
                "n_val": model_info.get("n_val"),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        frame = _load_predictions().copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date")
        ohlc = _load_ohlc_or_build(frame)
        candles = [
            {
                "time": _to_unix_seconds(row["date"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
            for _, row in ohlc.iterrows()
        ]
        close = ohlc["close"].to_numpy(dtype=float)
        horizon = INTERVAL_TO_HORIZON.get(resolved_interval, 45)
        try:
            pred_mean, pred_low, pred_high, model_info = forecast_with_global_model(
                close=close,
                interval=resolved_interval,
                horizon=horizon,
                z_value=CONFIDENCE_Z,
                return_clip=INTERVAL_TO_RETURN_CLIP.get(resolved_interval, 0.05),
                max_log_band=INTERVAL_TO_MAX_LOG_BAND.get(resolved_interval, 0.25),
            )
        except PretrainedModelNotFoundError as model_exc:
            raise HTTPException(status_code=503, detail=str(model_exc)) from model_exc
        start_date = pd.to_datetime(ohlc["date"].iloc[-1])
        last_close = float(ohlc["close"].iloc[-1])
        delta_map = {
            "1d": timedelta(days=1),
            "1h": timedelta(hours=1),
            "30m": timedelta(minutes=30),
            "15m": timedelta(minutes=15),
        }
        step = delta_map.get(resolved_interval, timedelta(days=1))
        future_dates = [start_date + step * (i + 1) for i in range(horizon)]
        predicted = [{"time": _to_unix_seconds(start_date), "value": last_close}] + [
            {"time": _to_unix_seconds(dt), "value": float(val)}
            for dt, val in zip(future_dates, pred_mean)
        ]
        predicted_lower = [{"time": _to_unix_seconds(start_date), "value": last_close}] + [
            {"time": _to_unix_seconds(dt), "value": float(val)}
            for dt, val in zip(future_dates, pred_low)
        ]
        predicted_upper = [{"time": _to_unix_seconds(start_date), "value": last_close}] + [
            {"time": _to_unix_seconds(dt), "value": float(val)}
            for dt, val in zip(future_dates, pred_high)
        ]

        metrics = {
            "mae": model_info.get("val_mae_ret"),
            "rmse": model_info.get("val_rmse_ret"),
            "mape": None,
            "model": model_info.get("model_name", "Global DL model"),
            "symbol": "fallback-baseline",
        }
        return {
            "candles": candles,
            "predicted": predicted,
            "predicted_lower": predicted_lower,
            "predicted_upper": predicted_upper,
            "metrics": metrics,
            "symbol_input": input_symbol,
            "symbol_resolved": "fallback-baseline",
            "interval_resolved": resolved_interval,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "data_source": "baseline-fallback",
            "warning": f"Live data fetch failed for '{input_symbol}'. Reason: {exc}",
            "forecast_horizon": horizon,
            "confidence_level": 0.95,
            "model_trained_at": model_info.get("trained_at"),
            "model_train_symbols": model_info.get("train_symbols"),
            "model_sample_info": {
                "n_train": model_info.get("n_train"),
                "n_val": model_info.get("n_val"),
            },
        }
