from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
BASELINE_OHLC_PATH = APP_DIR.parent.parent / "oil-price-baseline" / "outputs" / "ohlc.csv"

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
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


class PretrainedModelNotFoundError(RuntimeError):
    pass


def _model_path(interval: str, horizon: int) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR / f"global_dl_{interval}_h{horizon}.npz"


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
    if not series and BASELINE_OHLC_PATH.exists():
        try:
            ohlc = pd.read_csv(BASELINE_OHLC_PATH)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
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
            x_rows.append(returns[t - window : t].astype(np.float32))
            y_rows.append(returns[t : t + horizon].astype(np.float32))
            val_rows.append(sample_idx >= val_start)

    if not x_rows:
        raise ValueError("No training samples available")

    X = np.vstack(x_rows)
    Y = np.vstack(y_rows)
    val_mask = np.asarray(val_rows, dtype=bool)

    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(X), size=max_samples, replace=False)
        X = X[sel]
        Y = Y[sel]
        val_mask = val_mask[sel]
    return X, Y, val_mask


def _init_params(in_dim: int, out_dim: int, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    h1, h2 = 96, 96
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


def _train_mlp(X: np.ndarray, Y: np.ndarray, val_mask: np.ndarray, epochs: int = 80) -> dict:
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

    x_mean = X_train.mean(axis=0).astype(np.float32)
    x_std = (X_train.std(axis=0) + 1e-6).astype(np.float32)
    y_mean = Y_train.mean(axis=0).astype(np.float32)
    y_std = (Y_train.std(axis=0) + 1e-6).astype(np.float32)

    Xn = ((X_train - x_mean) / x_std).astype(np.float32)
    Yn = ((Y_train - y_mean) / y_std).astype(np.float32)
    Xv = ((X_val - x_mean) / x_std).astype(np.float32)
    Yv = ((Y_val - y_mean) / y_std).astype(np.float32)

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
            grad_y = (2.0 / len(xb)) * diff
            grads = _backward(params, cache, grad_y)
            step += 1
            _adam_update(params, grads, m, v, step, lr=1e-3)

        val_pred_n, _ = _forward(params, Xv)
        val_loss = float(np.mean((val_pred_n - Yv) ** 2))
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
    cum_pred = np.cumsum(val_pred, axis=1)
    cum_true = np.cumsum(Y_val, axis=1)
    cum_resid_std = np.std(cum_true - cum_pred, axis=0).astype(np.float32)
    cum_resid_std = np.maximum(cum_resid_std, 1e-4)

    # Model-level validation metrics for dashboard.
    one_step_pred = val_pred[:, 0]
    one_step_true = Y_val[:, 0]
    val_mae = float(np.mean(np.abs(one_step_true - one_step_pred)))
    val_rmse = float(np.sqrt(np.mean((one_step_true - one_step_pred) ** 2)))

    return {
        "params": params,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "cum_resid_std": cum_resid_std,
        "val_mae_ret": val_mae,
        "val_rmse_ret": val_rmse,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
    }


def _save_model(interval: str, horizon: int, bundle: dict, window: int, symbols: list[str]) -> dict:
    path = _model_path(interval, horizon)
    meta = {
        "interval": interval,
        "window": int(window),
        "horizon": int(horizon),
        "symbols": symbols,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "val_mae_ret": bundle["val_mae_ret"],
        "val_rmse_ret": bundle["val_rmse_ret"],
        "n_train": bundle["n_train"],
        "n_val": bundle["n_val"],
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
    model = _load_model(path)
    if model is None:
        raise RuntimeError("Failed to load saved model")
    return model


def _load_model(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        npz = np.load(path, allow_pickle=False)
        meta = json.loads(str(npz["meta"].item()))
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

    X, Y, val_mask = _build_dataset(series, window=window, horizon=horizon, max_samples=max_samples)
    bundle = _train_mlp(X, Y, val_mask=val_mask, epochs=epochs)
    model = _save_model(interval, horizon, bundle, window=window, symbols=used_symbols)
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
        "Run offline training first (train_pretrained_models.py)."
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

    close = np.asarray(close, dtype=np.float64)
    if len(close) < 4:
        flat = np.repeat(close[-1], horizon)
        info = {"model_name": "Global DL MLP (insufficient data)", "val_mae_ret": None, "val_rmse_ret": None}
        return flat, flat, flat, info

    returns = np.diff(np.log(close))
    if len(returns) >= window:
        x_window = returns[-window:]
    else:
        pad = np.zeros(window, dtype=np.float64)
        pad[-len(returns) :] = returns
        x_window = pad

    step_pred = _predict_step_returns(model, x_window)[:horizon]
    step_pred = np.clip(step_pred, -return_clip, return_clip)
    cum_pred = np.cumsum(step_pred)

    cum_std = np.asarray(model["cum_resid_std"][:horizon], dtype=np.float64)
    if len(cum_std) < horizon:
        last = float(cum_std[-1]) if len(cum_std) else 1e-3
        cum_std = np.pad(cum_std, (0, horizon - len(cum_std)), constant_values=last)
    band = np.minimum(z_value * cum_std, max_log_band)

    base_price = float(close[-1])
    pred_mean = base_price * np.exp(cum_pred)
    pred_low = base_price * np.exp(cum_pred - band)
    pred_high = base_price * np.exp(cum_pred + band)

    info = {
        "model_name": "Global DL MLP (pretrained, multi-horizon)",
        "val_mae_ret": model["meta"].get("val_mae_ret"),
        "val_rmse_ret": model["meta"].get("val_rmse_ret"),
        "trained_at": model["meta"].get("trained_at"),
        "train_symbols": model["meta"].get("symbols", []),
        "n_train": model["meta"].get("n_train"),
        "n_val": model["meta"].get("n_val"),
    }
    return pred_mean, pred_low, pred_high, info
