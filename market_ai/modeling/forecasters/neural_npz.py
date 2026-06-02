from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from market_ai.config import get_settings
from market_ai.modeling.registry import metadata_for_artifact, write_model_metadata_sidecar

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = get_settings().model_dir
BASELINE_OHLC_PATH = get_settings().baseline_ohlc_path

DEFAULT_SYMBOLS = [
    "CL=F",
    "BZ=F",
    "NG=F",
    "RB=F",
    "HO=F",
    "GC=F",
    "SI=F",
    "DX-Y.NYB",
    "USDKRW=X",
    "EURUSD=X",
    "JPY=X",
    "SPY",
    "QQQ",
    "XLE",
    "USO",
]

INTERVAL_CFG = {
    "1d": {"period": "10y", "window": 64, "max_samples": 24000, "epochs": 90},
    "1h": {"period": "730d", "window": 96, "max_samples": 20000, "epochs": 80},
    "30m": {"period": "60d", "window": 120, "max_samples": 16000, "epochs": 70},
    "15m": {"period": "60d", "window": 144, "max_samples": 12000, "epochs": 60},
}
INTERVAL_PERIOD_CANDIDATES = {
    "1d": ["10y", "5y", "2y"],
    "1h": ["730d", "365d", "180d"],
    "30m": ["60d", "30d", "14d"],
    "15m": ["60d", "30d", "14d"],
}

_MODEL_CACHE: dict[str, dict] = {}
FEATURE_VERSION = "pattern_features_v4_cum_path"
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


class PretrainedModelNotFoundError(RuntimeError):
    pass


def _model_path(interval: str, horizon: int) -> Path:
    model_dir = get_settings().model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"global_dl_{interval}_h{horizon}.npz"


def _download_close(symbol: str, interval: str, period: str) -> np.ndarray | None:
    candidates = INTERVAL_PERIOD_CANDIDATES.get(interval, [period])
    for p in candidates:
        data = yf.download(
            symbol,
            period=p,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            continue
        frame = data.reset_index()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in frame.columns]
        if "Close" not in frame.columns:
            continue
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna().to_numpy(dtype=float)
        close = close[close > 0]
        if len(close) < 120:
            continue
        return close
    return None


def _collect_series(interval: str) -> tuple[list[np.ndarray], list[str]]:
    cfg = INTERVAL_CFG.get(interval, INTERVAL_CFG["1d"])
    period = cfg["period"]
    series: list[np.ndarray] = []
    used_symbols: list[str] = []

    for sym in DEFAULT_SYMBOLS:
        try:
            close = _download_close(sym, interval=interval, period=period)
            if close is not None:
                series.append(close)
                used_symbols.append(sym)
        except Exception:
            continue

    # Final fallback if external data is unavailable.
    baseline_ohlc_path = get_settings().baseline_ohlc_path
    if not series and baseline_ohlc_path.exists():
        try:
            ohlc = pd.read_csv(baseline_ohlc_path)
            if "close" in ohlc.columns:
                close = pd.to_numeric(ohlc["close"], errors="coerce").dropna().to_numpy(dtype=float)
                close = close[close > 0]
                if len(close) > 200:
                    series.append(close)
                    used_symbols.append("fallback-baseline")
        except Exception:
            pass

    return series, used_symbols


def _collect_series_from_symbols(interval: str, symbols: list[str]) -> tuple[list[np.ndarray], list[str]]:
    cfg = INTERVAL_CFG.get(interval, INTERVAL_CFG["1d"])
    period = cfg["period"]
    series: list[np.ndarray] = []
    used: list[str] = []
    for sym in symbols:
        try:
            close = _download_close(sym, interval=interval, period=period)
            if close is not None:
                series.append(close)
                used.append(sym)
        except Exception:
            continue
    return series, used


def _build_dataset(
    series_list: list[np.ndarray],
    window: int,
    horizon: int,
    max_samples: int,
    interval: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    scale_rows: list[float] = []
    val_rows: list[bool] = []

    for close in series_list:
        returns = np.diff(np.log(close))
        if len(returns) < window + horizon + 16:
            continue

        idx = np.arange(window, len(returns) - horizon + 1)
        # Downsample long series so one symbol does not dominate.
        max_per_series = max(800, max_samples // max(1, len(series_list)))
        if len(idx) > max_per_series:
            step = int(np.ceil(len(idx) / max_per_series))
            idx = idx[::step]

        val_count = max(1, int(np.ceil(len(idx) * 0.12))) if len(idx) > 1 else 0
        val_start = len(idx) - val_count
        for sample_idx, t in enumerate(idx):
            history = returns[t - window : t]
            scale = max(_recent_step_volatility(history, window), 1e-5)
            x_rows.append(_pattern_feature_vector(history, window, interval))
            future_path = np.cumsum(returns[t : t + horizon])
            y_rows.append(np.clip(future_path / scale, -12.0, 12.0).astype(np.float32))
            scale_rows.append(float(scale))
            val_rows.append(sample_idx >= val_start)

    if not x_rows:
        raise ValueError("No training samples available")

    X = np.vstack(x_rows)
    Y = np.vstack(y_rows)
    y_scale = np.asarray(scale_rows, dtype=np.float32)
    val_mask = np.asarray(val_rows, dtype=bool)

    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(X), size=max_samples, replace=False)
        X = X[sel]
        Y = Y[sel]
        y_scale = y_scale[sel]
        val_mask = val_mask[sel]
    return X, Y, y_scale, val_mask


def _safe_std(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    val = float(np.std(values))
    return val if np.isfinite(val) else 0.0


def _tail(values: np.ndarray, length: int) -> np.ndarray:
    length = min(len(values), max(1, int(length)))
    return values[-length:]


def _autocorr(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag + 3:
        return 0.0
    a = values[:-lag]
    b = values[lag:]
    a_std = _safe_std(a)
    b_std = _safe_std(b)
    if a_std <= 1e-8 or b_std <= 1e-8:
        return 0.0
    return float(np.mean((a - np.mean(a)) * (b - np.mean(b))) / (a_std * b_std))


def _cycle_projection(values: np.ndarray, period: int) -> tuple[float, float]:
    if len(values) < max(6, period // 2):
        return 0.0, 0.0
    tail = _tail(values, min(len(values), period * 2))
    tail = tail - float(np.mean(tail))
    std = _safe_std(tail)
    if std <= 1e-8:
        return 0.0, 0.0
    t = np.arange(len(tail), dtype=np.float64)
    angle = 2.0 * np.pi * t / float(period)
    scaled = tail / std
    return float(np.mean(scaled * np.sin(angle))), float(np.mean(scaled * np.cos(angle)))


def _pattern_feature_vector(return_window: np.ndarray, window: int, interval: str) -> np.ndarray:
    returns = np.asarray(return_window, dtype=np.float64).reshape(-1)
    returns = returns[np.isfinite(returns)]
    if len(returns) >= window:
        returns = returns[-window:]
    else:
        padded = np.zeros(window, dtype=np.float64)
        if len(returns):
            padded[-len(returns) :] = returns
        returns = padded

    recent_vol = max(_safe_std(_tail(returns, max(8, window // 3))), 1e-6)
    normalized = np.clip(returns / recent_vol, -6.0, 6.0)
    cum_path = np.cumsum(returns)
    cum_path = cum_path - float(cum_path[-1])
    path_scale = max(_safe_std(cum_path), recent_vol * np.sqrt(window), 1e-6)
    norm_path = np.clip(cum_path / path_scale, -6.0, 6.0)

    momentum_windows = [3, 5, 8, 13, 21, 34, max(8, window // 2), window]
    vol_windows = [5, 13, 21, 34, max(8, window // 2), window]
    stats: list[float] = []
    for n in momentum_windows:
        tail = _tail(returns, n)
        stats.append(float(np.sum(tail)))
        stats.append(float(np.mean(tail)))
    for n in vol_windows:
        tail = _tail(returns, n)
        stats.append(_safe_std(tail))
        stats.append(float(np.mean(np.abs(tail))) if len(tail) else 0.0)

    signs = np.sign(returns)
    nonzero_signs = signs[signs != 0]
    if len(nonzero_signs) >= 2:
        turn_rate = float(np.mean(nonzero_signs[1:] != nonzero_signs[:-1]))
        last_sign = float(nonzero_signs[-1])
        run_len = 1
        for sign in nonzero_signs[-2::-1]:
            if sign != last_sign:
                break
            run_len += 1
    else:
        turn_rate = 0.0
        last_sign = 0.0
        run_len = 0
    stats.extend(
        [
            turn_rate,
            last_sign,
            float(run_len / max(1, window)),
            float(np.mean(returns > 0.0)),
            float(returns[-1]),
            float(returns[-1] - returns[-2]),
            float(np.max(cum_path) - np.min(cum_path)),
        ]
    )

    for lag in [1, 2, 3, 5, 8, 13, 21]:
        stats.append(_autocorr(returns, lag))

    interval_periods = {
        "1d": [5, 8, 13, 21, 34],
        "1h": [8, 13, 21, 34, 55],
        "30m": [8, 13, 21, 34, 55],
        "15m": [13, 21, 34, 55, 89],
    }
    for period in interval_periods.get(interval, [8, 13, 21, 34, 55]):
        sin_proj, cos_proj = _cycle_projection(returns, period)
        stats.extend([sin_proj, cos_proj])

    feature = np.concatenate(
        [
            returns.astype(np.float64),
            normalized.astype(np.float64),
            norm_path.astype(np.float64),
            np.asarray(stats, dtype=np.float64),
        ]
    )
    feature = np.nan_to_num(feature, nan=0.0, posinf=6.0, neginf=-6.0)
    return feature.astype(np.float32)


def _init_params(in_dim: int, out_dim: int, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    h1, h2 = 160, 128
    w1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, h1)).astype(np.float32)
    b1 = np.zeros(h1, dtype=np.float32)
    w2 = rng.normal(0.0, np.sqrt(2.0 / h1), size=(h1, h2)).astype(np.float32)
    b2 = np.zeros(h2, dtype=np.float32)
    w3 = rng.normal(0.0, np.sqrt(2.0 / h2), size=(h2, out_dim)).astype(np.float32)
    b3 = np.zeros(out_dim, dtype=np.float32)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}


def _forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z1 = x @ params["w1"] + params["b1"]
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ params["w2"] + params["b2"]
    a2 = np.maximum(z2, 0.0)
    y = a2 @ params["w3"] + params["b3"]
    cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
    return y, cache


def _backward(
    params: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    grad_y: np.ndarray,
) -> dict[str, np.ndarray]:
    x = cache["x"]
    z1, a1 = cache["z1"], cache["a1"]
    z2, a2 = cache["z2"], cache["a2"]

    dw3 = a2.T @ grad_y
    db3 = grad_y.sum(axis=0)
    da2 = grad_y @ params["w3"].T
    dz2 = da2 * (z2 > 0)
    dw2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ params["w2"].T
    dz1 = da1 * (z1 > 0)
    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2, "w3": dw3, "b3": db3}


def _adam_update(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    m: dict[str, np.ndarray],
    v: dict[str, np.ndarray],
    step: int,
    lr: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    for k in params:
        m[k] = b1 * m[k] + (1.0 - b1) * grads[k]
        v[k] = b2 * v[k] + (1.0 - b2) * (grads[k] ** 2)
        m_hat = m[k] / (1.0 - b1**step)
        v_hat = v[k] / (1.0 - b2**step)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def _train_mlp(
    X: np.ndarray,
    Y: np.ndarray,
    y_scale: np.ndarray,
    val_mask: np.ndarray,
    epochs: int = 80,
) -> dict:
    rng = np.random.default_rng(42)
    n = len(X)
    val_mask = np.asarray(val_mask, dtype=bool)
    if len(val_mask) != n:
        raise ValueError("Validation mask length does not match dataset")
    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    if len(train_idx) == 0 or len(val_idx) == 0:
        split = max(1, int(n * 0.8))
        split = min(split, n - 1)
        train_idx = np.arange(split)
        val_idx = np.arange(split, n)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    scale_val = y_scale[val_idx].reshape(-1, 1).astype(np.float32)

    x_mean = X_train.mean(axis=0).astype(np.float32)
    x_std = (X_train.std(axis=0) + 1e-6).astype(np.float32)
    y_mean = Y_train.mean(axis=0).astype(np.float32)
    y_std = (Y_train.std(axis=0) + 1e-6).astype(np.float32)

    Xn = ((X_train - x_mean) / x_std).astype(np.float32)
    Yn = ((Y_train - y_mean) / y_std).astype(np.float32)
    Xv = ((X_val - x_mean) / x_std).astype(np.float32)
    Yv = ((Y_val - y_mean) / y_std).astype(np.float32)
    horizon_weights = (1.0 / np.sqrt(np.arange(1, Y.shape[1] + 1, dtype=np.float32))).reshape(1, -1)
    horizon_weights = horizon_weights / float(np.mean(horizon_weights))

    params = _init_params(X.shape[1], Y.shape[1], seed=42)
    m = {k: np.zeros_like(v_) for k, v_ in params.items()}
    v = {k: np.zeros_like(v_) for k, v_ in params.items()}

    best = {k: val.copy() for k, val in params.items()}
    best_val = float("inf")
    patience = 12
    wait = 0
    step = 0
    batch = 256

    for _epoch in range(epochs):
        order = rng.permutation(len(Xn))
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            xb = Xn[idx]
            yb = Yn[idx]
            pred, cache = _forward(params, xb)
            diff = pred - yb
            grad_y = (2.0 / len(xb)) * diff * horizon_weights
            grads = _backward(params, cache, grad_y)
            step += 1
            _adam_update(params, grads, m, v, step, lr=1e-3)

        val_pred_n, _ = _forward(params, Xv)
        val_loss = float(np.mean(((val_pred_n - Yv) ** 2) * horizon_weights))
        if val_loss < best_val:
            best_val = val_loss
            best = {k: val.copy() for k, val in params.items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    params = best
    val_pred_n, _ = _forward(params, Xv)
    val_pred = val_pred_n * y_std + y_mean
    val_pred_raw = val_pred * scale_val
    y_val_raw = Y_val * scale_val
    cum_resid_std = np.std(y_val_raw - val_pred_raw, axis=0).astype(np.float32)
    cum_resid_std = np.maximum(cum_resid_std, 1e-4)

    # Model-level validation metrics for dashboard.
    one_step_pred = val_pred_raw[:, 0]
    one_step_true = y_val_raw[:, 0]
    val_mae = float(np.mean(np.abs(one_step_true - one_step_pred)))
    val_rmse = float(np.sqrt(np.mean((one_step_true - one_step_pred) ** 2)))
    val_mape = float(np.mean(np.abs(np.exp(one_step_pred - one_step_true) - 1.0)) * 100)

    return {
        "params": params,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "cum_resid_std": cum_resid_std,
        "val_mae_ret": val_mae,
        "val_rmse_ret": val_rmse,
        "val_mape_pct": val_mape,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "input_dim": int(X.shape[1]),
        "target_mode": "volatility_scaled_cumulative_returns",
    }


def _save_model(interval: str, horizon: int, bundle: dict, window: int, symbols: list[str]) -> dict:
    path = _model_path(interval, horizon)
    meta = {
        "model_name": "pattern_mlp",
        "model_type": "global_dl_mlp",
        "version": "global_mlp_v1",
        "artifact_file": path.name,
        "interval": interval,
        "window": int(window),
        "horizon": int(horizon),
        "symbols": symbols,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_cutoff": datetime.now(timezone.utc).isoformat(),
        "asset_universe": symbols,
        "supported_asset_classes": ["unknown"],
        "supported_intervals": [interval],
        "val_mae_ret": bundle["val_mae_ret"],
        "val_rmse_ret": bundle["val_rmse_ret"],
        "val_mape_pct": bundle["val_mape_pct"],
        "n_train": bundle["n_train"],
        "n_val": bundle["n_val"],
        "feature_version": FEATURE_VERSION,
        "feature_set": FEATURE_VERSION,
        "input_dim": bundle["input_dim"],
        "target_mode": bundle["target_mode"],
        "target": bundle["target_mode"],
        "scaler": "recent_realized_volatility",
        "metrics": {
            "val_mae_ret": bundle["val_mae_ret"],
            "val_rmse_ret": bundle["val_rmse_ret"],
            "val_mape_pct": bundle["val_mape_pct"],
            "n_train": bundle["n_train"],
            "n_val": bundle["n_val"],
        },
    }
    np.savez_compressed(
        path,
        w1=bundle["params"]["w1"],
        b1=bundle["params"]["b1"],
        w2=bundle["params"]["w2"],
        b2=bundle["params"]["b2"],
        w3=bundle["params"]["w3"],
        b3=bundle["params"]["b3"],
        x_mean=bundle["x_mean"],
        x_std=bundle["x_std"],
        y_mean=bundle["y_mean"],
        y_std=bundle["y_std"],
        cum_resid_std=bundle["cum_resid_std"],
        meta=np.array(json.dumps(meta)),
    )
    write_model_metadata_sidecar(path, meta, metadata_dir=get_settings().metadata_dir)
    model = _load_model(path)
    if model is None:
        raise RuntimeError("Failed to load saved model")
    return model


def _load_model(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        npz = np.load(path, allow_pickle=False)
        if "meta" in npz:
            meta = json.loads(str(npz["meta"].item()))
        else:
            sidecar_meta = metadata_for_artifact(path)
            meta = {
                "model_name": sidecar_meta.model_name,
                "model_type": sidecar_meta.model_type,
                "version": sidecar_meta.version,
                "artifact_file": sidecar_meta.artifact_file,
                "interval": sidecar_meta.supported_intervals[0] if sidecar_meta.supported_intervals else None,
                "window": sidecar_meta.lookback,
                "horizon": sidecar_meta.horizon,
                "symbols": sidecar_meta.asset_universe,
                "trained_at": sidecar_meta.created_at,
                "training_cutoff": sidecar_meta.training_cutoff,
                "feature_version": sidecar_meta.feature_set,
                "target_mode": sidecar_meta.target,
                **sidecar_meta.metrics,
            }
        model = {
            "w1": np.asarray(npz["w1"], dtype=np.float32),
            "b1": np.asarray(npz["b1"], dtype=np.float32),
            "w2": np.asarray(npz["w2"], dtype=np.float32),
            "b2": np.asarray(npz["b2"], dtype=np.float32),
            "w3": np.asarray(npz["w3"], dtype=np.float32),
            "b3": np.asarray(npz["b3"], dtype=np.float32),
            "x_mean": np.asarray(npz["x_mean"], dtype=np.float32),
            "x_std": np.asarray(npz["x_std"], dtype=np.float32),
            "y_mean": np.asarray(npz["y_mean"], dtype=np.float32),
            "y_std": np.asarray(npz["y_std"], dtype=np.float32),
            "cum_resid_std": np.asarray(npz["cum_resid_std"], dtype=np.float32),
            "meta": meta,
        }
        return model
    except Exception:
        return None


def _predict_step_returns(model: dict, x_window: np.ndarray) -> np.ndarray:
    x = np.asarray(x_window, dtype=np.float32)
    x_n = (x - model["x_mean"]) / model["x_std"]
    z1 = x_n @ model["w1"] + model["b1"]
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ model["w2"] + model["b2"]
    a2 = np.maximum(z2, 0.0)
    y_n = a2 @ model["w3"] + model["b3"]
    y = y_n * model["y_std"] + model["y_mean"]
    return y.astype(np.float64)


def _model_input_vector(model: dict, returns: np.ndarray, window: int, interval: str) -> np.ndarray:
    feature = _pattern_feature_vector(returns[-window:], window, interval)
    expected_dim = int(np.asarray(model["x_mean"]).shape[0])
    if len(feature) == expected_dim:
        return feature

    # Compatibility for old raw-window weight files until they are retrained.
    padded = np.zeros(window, dtype=np.float64)
    tail = np.asarray(returns, dtype=np.float64)[-window:]
    if len(tail):
        padded[-len(tail) :] = tail
    if len(padded) == expected_dim:
        return padded.astype(np.float32)

    raise ValueError(
        f"Model input dimension mismatch: expected {expected_dim}, "
        f"got feature_dim={len(feature)} and raw_dim={len(padded)}. Retrain model weights."
    )


def _clean_close(close: np.ndarray) -> np.ndarray:
    arr = np.asarray(close, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(arr) == 0:
        raise ValueError("Forecast requires at least one positive close price")
    return arr


def _recent_step_volatility(returns: np.ndarray, window: int) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 4:
        return 0.0

    short_lookback = min(len(returns), max(12, window // 3))
    medium_lookback = min(len(returns), max(24, window))
    short_vol = float(np.std(returns[-short_lookback:]))
    medium_vol = float(np.std(returns[-medium_lookback:]))
    return max(short_vol, medium_vol * 0.75, 0.0)


def _recent_drift_curve(
    returns: np.ndarray,
    horizon: int,
    interval: str,
    return_clip: float,
) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 6:
        return np.zeros(horizon, dtype=np.float64)

    short = float(np.mean(returns[-min(len(returns), 8) :]))
    medium = float(np.mean(returns[-min(len(returns), 34) :]))
    drift = 0.65 * short + 0.35 * medium
    drift = float(np.clip(drift, -return_clip * 0.2, return_clip * 0.2))

    blend_by_interval = {
        "1d": 0.05,
        "1h": 0.14,
        "30m": 0.18,
        "15m": 0.18,
    }
    blend = float(blend_by_interval.get(interval, 0.1))
    decay = np.exp(-np.arange(horizon, dtype=np.float64) / max(horizon * 0.45, 1.0))
    return blend * drift * decay


def _calibrated_cum_std(
    model_cum_std: np.ndarray,
    returns: np.ndarray,
    horizon: int,
    window: int,
) -> tuple[np.ndarray, dict]:
    steps = np.arange(1, horizon + 1, dtype=np.float64)
    model_cum_std = np.asarray(model_cum_std, dtype=np.float64)
    recent_step_vol = _recent_step_volatility(returns, window)

    if recent_step_vol <= 0.0:
        return model_cum_std, {
            "band_calibration": "model_validation_residual",
            "recent_step_vol": None,
            "band_scale": 1.0,
        }

    realized_cum_std = recent_step_vol * np.sqrt(steps)
    calibrated = np.minimum(model_cum_std, realized_cum_std * 2.25)
    calibrated = np.maximum(calibrated, model_cum_std * 0.35)

    scale = float(np.mean(calibrated / np.maximum(model_cum_std, 1e-8)))
    return calibrated, {
        "band_calibration": "validation_residual_capped_by_recent_volatility",
        "recent_step_vol": recent_step_vol,
        "band_scale": scale,
    }


def _calibrate_path_amplitude(
    cum_pred: np.ndarray,
    returns: np.ndarray,
    horizon: int,
    interval: str,
    return_clip: float,
) -> tuple[np.ndarray, dict]:
    cum_pred = np.asarray(cum_pred, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    if len(cum_pred) < 4 or len(returns) < 12:
        return cum_pred, {"path_gain": 1.0, "target_path_range": None}

    lookback_by_interval = {
        "1d": 45,
        "1h": 120,
        "30m": 120,
        "15m": 160,
    }
    min_range_by_interval = {
        "1d": 0.06,
        "1h": 0.018,
        "30m": 0.018,
        "15m": 0.014,
    }
    ratio_by_interval = {
        "1d": 0.45,
        "1h": 0.28,
        "30m": 0.28,
        "15m": 0.25,
    }

    lookback = min(len(returns), lookback_by_interval.get(interval, horizon))
    recent_path = np.cumsum(returns[-lookback:])
    recent_range = float(np.max(recent_path) - np.min(recent_path))
    target_range = max(
        min_range_by_interval.get(interval, 0.015),
        recent_range * ratio_by_interval.get(interval, 0.25),
    )

    trend = np.linspace(0.0, float(cum_pred[-1]), len(cum_pred))
    residual = cum_pred - trend
    residual_range = float(np.max(residual) - np.min(residual))
    if residual_range <= 1e-8:
        return cum_pred, {"path_gain": 1.0, "target_path_range": target_range}

    current_range = float(np.max(cum_pred) - np.min(cum_pred))
    if current_range >= target_range:
        return cum_pred, {"path_gain": 1.0, "target_path_range": target_range}

    gain = min(target_range / max(residual_range, 1e-8), 8.0)
    adjusted = trend + residual * gain
    cap = return_clip * np.sqrt(np.arange(1, horizon + 1, dtype=np.float64))
    adjusted = np.clip(adjusted, -cap, cap)
    return adjusted, {"path_gain": float(gain), "target_path_range": target_range}


def train_and_save_pretrained_model(
    interval: str,
    horizon: int,
    *,
    force: bool = False,
    symbols: list[str] | None = None,
    fallback_close: np.ndarray | None = None,
) -> dict:
    cache_key = f"{interval}|{horizon}"
    path = _model_path(interval, horizon)
    if not force and path.exists():
        loaded = _load_model(path)
        if loaded is None:
            raise RuntimeError(f"Failed to read model file: {path}")
        _MODEL_CACHE[cache_key] = loaded
        return loaded

    cfg = INTERVAL_CFG.get(interval, INTERVAL_CFG["1d"])
    window = int(cfg["window"])
    max_samples = int(cfg["max_samples"])
    epochs = int(cfg["epochs"])

    if symbols:
        series, used_symbols = _collect_series_from_symbols(interval, symbols)
    else:
        series, used_symbols = _collect_series(interval)

    if not series and fallback_close is not None and len(fallback_close) > 200:
        series = [np.asarray(fallback_close, dtype=float)]
        used_symbols = ["request-fallback-close"]
    if not series:
        raise RuntimeError("No data available to train pretrained model")

    X, Y, y_scale, val_mask = _build_dataset(
        series,
        window=window,
        horizon=horizon,
        max_samples=max_samples,
        interval=interval,
    )
    bundle = _train_mlp(X, Y, y_scale=y_scale, val_mask=val_mask, epochs=epochs)
    model = _save_model(interval, horizon, bundle, window=window, symbols=used_symbols)
    _MODEL_CACHE[cache_key] = model
    return model


def train_and_save_pretrained_model_from_series(
    interval: str,
    horizon: int,
    *,
    series: list[np.ndarray],
    symbols: list[str],
    force: bool = False,
) -> dict:
    cache_key = f"{interval}|{horizon}"
    path = _model_path(interval, horizon)
    if not force and path.exists():
        loaded = _load_model(path)
        if loaded is None:
            raise RuntimeError(f"Failed to read model file: {path}")
        _MODEL_CACHE[cache_key] = loaded
        return loaded

    if not series:
        raise RuntimeError("No local series available to train pretrained model")

    cfg = INTERVAL_CFG.get(interval, INTERVAL_CFG["1d"])
    window = int(cfg["window"])
    max_samples = int(cfg["max_samples"])
    epochs = int(cfg["epochs"])
    X, Y, y_scale, val_mask = _build_dataset(
        series,
        window=window,
        horizon=horizon,
        max_samples=max_samples,
        interval=interval,
    )
    bundle = _train_mlp(X, Y, y_scale=y_scale, val_mask=val_mask, epochs=epochs)
    model = _save_model(interval, horizon, bundle, window=window, symbols=symbols)
    _MODEL_CACHE[cache_key] = model
    return model


def load_pretrained_model(interval: str, horizon: int) -> dict:
    cache_key = f"{interval}|{horizon}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    path = _model_path(interval, horizon)
    loaded = _load_model(path)
    if loaded is not None:
        _MODEL_CACHE[cache_key] = loaded
        return loaded

    raise PretrainedModelNotFoundError(
        f"Pretrained model file not found: {path}. "
        "Run offline training first (python scripts/train/train_pretrained_models.py)."
    )


def forecast_with_global_model(
    close: np.ndarray,
    interval: str,
    horizon: int,
    z_value: float,
    return_clip: float,
    max_log_band: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    model = load_pretrained_model(interval, horizon)
    window = int(model["meta"]["window"])

    close = _clean_close(close)
    if len(close) < 4:
        flat = np.repeat(close[-1], horizon)
        info = {
            "model_name": "Global DL MLP (insufficient data)",
            "val_mae_ret": None,
            "val_rmse_ret": None,
            "val_mape_pct": None,
            "band_calibration": "none",
        }
        return flat, flat, flat, info

    returns = np.diff(np.log(close))
    x_input = _model_input_vector(model, returns, window, interval)
    model_out = _predict_step_returns(model, x_input)[:horizon]
    if model["meta"].get("target_mode") == "volatility_scaled_cumulative_returns":
        cum_pred = model_out * max(_recent_step_volatility(returns, window), 1e-5)
        cum_pred = cum_pred + np.cumsum(_recent_drift_curve(returns, horizon, interval, return_clip))
        cum_pred = np.clip(cum_pred, -return_clip * np.sqrt(np.arange(1, horizon + 1)), return_clip * np.sqrt(np.arange(1, horizon + 1)))
        cum_pred, path_info = _calibrate_path_amplitude(cum_pred, returns, horizon, interval, return_clip)
        step_pred = np.diff(np.concatenate([[0.0], cum_pred]))
    else:
        step_pred = model_out
        if model["meta"].get("target_mode") == "volatility_scaled_returns":
            step_pred = step_pred * max(_recent_step_volatility(returns, window), 1e-5)
        step_pred = step_pred + _recent_drift_curve(returns, horizon, interval, return_clip)
        step_pred = np.clip(step_pred, -return_clip, return_clip)
        cum_pred = np.cumsum(step_pred)
        path_info = {"path_gain": 1.0, "target_path_range": None}
    pattern_returns = np.zeros(horizon, dtype=np.float64)

    cum_std = np.asarray(model["cum_resid_std"][:horizon], dtype=np.float64)
    if len(cum_std) < horizon:
        last = float(cum_std[-1]) if len(cum_std) else 1e-3
        cum_std = np.pad(cum_std, (0, horizon - len(cum_std)), constant_values=last)
    cum_std, band_info = _calibrated_cum_std(cum_std, returns, horizon, window)
    band = np.minimum(z_value * cum_std, max_log_band)

    base_price = float(close[-1])
    pred_mean = base_price * np.exp(cum_pred)
    pred_low = base_price * np.exp(cum_pred - band)
    pred_high = base_price * np.exp(cum_pred + band)

    info = {
        "model_name": "Pattern-Aware Global MLP",
        "val_mae_ret": model["meta"].get("val_mae_ret"),
        "val_rmse_ret": model["meta"].get("val_rmse_ret"),
        "val_mape_pct": model["meta"].get("val_mape_pct"),
        "trained_at": model["meta"].get("trained_at"),
        "train_symbols": model["meta"].get("symbols", []),
        "n_train": model["meta"].get("n_train"),
        "n_val": model["meta"].get("n_val"),
        "feature_version": model["meta"].get("feature_version", "raw_window_v1"),
        "input_dim": model["meta"].get("input_dim"),
        "target_mode": model["meta"].get("target_mode", "raw_step_returns"),
        "mean_step_return": float(np.mean(step_pred)),
        "step_return_std": float(np.std(step_pred)),
        "pattern_step_std": float(np.std(pattern_returns)),
        "band_last_log": float(band[-1]) if len(band) else None,
        **path_info,
        **band_info,
    }
    return pred_mean, pred_low, pred_high, info
