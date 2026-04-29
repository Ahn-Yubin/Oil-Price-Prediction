from __future__ import annotations

import numpy as np
import pandas as pd

from market_ai.modeling.forecasters.neural_npz import forecast_with_global_model

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

_SEQ_MODEL_CACHE: dict[tuple[str, str, int], object] = {}
_WINDOW_BY_INTERVAL = {"1d": 64, "1h": 96, "30m": 120, "15m": 144}


if nn is not None:

    class _LiveLSTM(nn.Module):
        def __init__(self, in_dim: int, hidden: int, horizon: int):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, horizon))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])


    class _LiveTCN(nn.Module):
        def __init__(self, in_dim: int, hidden: int, horizon: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=9, padding=4),
                nn.GELU(),
            )
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, horizon))

        def forward(self, x):
            return self.head(self.net(x.transpose(1, 2)).mean(dim=-1))


def _clean_close(values) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    return arr[np.isfinite(arr) & (arr > 0.0)]


def _returns(close: np.ndarray) -> np.ndarray:
    return np.diff(np.log(_clean_close(close)))


def _safe_std(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    val = float(np.std(values))
    return val if np.isfinite(val) else 0.0


def _recent_vol(returns: np.ndarray, window: int) -> float:
    lookback = min(len(returns), max(12, window // 2))
    return max(_safe_std(returns[-lookback:]), 1e-5)


def _window_signature(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    vol = max(_safe_std(values), 1e-6)
    norm_ret = np.clip((values - float(np.mean(values))) / vol, -6.0, 6.0)
    path = np.cumsum(values)
    path = path - float(path[-1])
    path_std = max(_safe_std(path), vol * np.sqrt(len(values)), 1e-6)
    norm_path = np.clip(path / path_std, -6.0, 6.0)
    return np.concatenate([norm_ret, norm_path])


def _sequence_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    vol = max(_safe_std(values), 1e-6)
    norm_ret = np.clip((values - float(np.mean(values))) / vol, -6.0, 6.0)
    path = np.cumsum(values)
    path = path - float(path[-1])
    path_std = max(_safe_std(path), vol * np.sqrt(len(values)), 1e-6)
    norm_path = np.clip(path / path_std, -6.0, 6.0)
    short = np.zeros_like(values)
    medium = np.zeros_like(values)
    for i in range(len(values)):
        short[i] = np.mean(values[max(0, i - 4) : i + 1]) / vol
        medium[i] = np.mean(values[max(0, i - 12) : i + 1]) / vol
    return np.stack([norm_ret, norm_path, np.clip(short, -6.0, 6.0), np.clip(medium, -6.0, 6.0)], axis=1).astype(np.float32)


def _live_sequence_path(close: np.ndarray, interval: str, horizon: int, kind: str) -> np.ndarray | None:
    if torch is None or nn is None:
        return None
    returns = _returns(close)
    window = _WINDOW_BY_INTERVAL.get(interval, 96)
    if len(returns) < window + horizon + 16:
        return None

    cache_key = (kind, interval, horizon)
    model = _SEQ_MODEL_CACHE.get(cache_key)
    if model is None:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for t in range(window, len(returns) - horizon + 1):
            hist = returns[t - window : t]
            fut = returns[t : t + horizon]
            scale = max(_recent_vol(hist, window), 1e-5)
            xs.append(_sequence_features(hist))
            ys.append(np.clip(np.cumsum(fut) / scale, -12.0, 12.0).astype(np.float32))
        if not xs:
            return None
        X = np.stack(xs).astype(np.float32)
        Y = np.stack(ys).astype(np.float32)
        if len(X) > 700:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(len(X), size=700, replace=False))
            X, Y = X[idx], Y[idx]

        torch.manual_seed(42)
        torch.set_num_threads(1)
        model = _LiveLSTM(X.shape[-1], 18, horizon) if kind == "lstm" else _LiveTCN(X.shape[-1], 24, horizon)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(Y)), batch_size=256, shuffle=True)
        weights = torch.tensor(1.0 / np.sqrt(np.arange(1, horizon + 1, dtype=np.float32)))
        weights = weights / weights.mean()
        model.train()
        for _epoch in range(3):
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = (((pred - yb) ** 2) * weights).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        _SEQ_MODEL_CACHE[cache_key] = model

    recent_scale = max(_recent_vol(returns, window), 1e-5)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(_sequence_features(returns[-window:])[None, :, :])).cpu().numpy()[0]
    path = pred_scaled.astype(np.float64) * recent_scale
    path, _gain = _calibrate_amplitude(path, returns, interval, horizon)
    return path


def _target_range(returns: np.ndarray, interval: str, horizon: int) -> float:
    lookbacks = {"1d": 45, "1h": 120, "30m": 120, "15m": 160}
    ratios = {"1d": 0.45, "1h": 0.28, "30m": 0.28, "15m": 0.25}
    floors = {"1d": 0.06, "1h": 0.018, "30m": 0.018, "15m": 0.014}
    lookback = min(len(returns), lookbacks.get(interval, horizon))
    recent_path = np.cumsum(returns[-lookback:])
    recent_range = float(np.max(recent_path) - np.min(recent_path)) if len(recent_path) else 0.0
    return max(floors.get(interval, 0.015), recent_range * ratios.get(interval, 0.25))


def _calibrate_amplitude(path: np.ndarray, returns: np.ndarray, interval: str, horizon: int) -> tuple[np.ndarray, float]:
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 4:
        return path, 1.0
    target_range = _target_range(returns, interval, horizon)
    current_range = float(np.max(path) - np.min(path))
    if current_range >= target_range:
        return path, 1.0
    trend = np.linspace(0.0, float(path[-1]), len(path))
    residual = path - trend
    residual_range = float(np.max(residual) - np.min(residual))
    if residual_range <= 1e-8:
        return path, 1.0
    gain = min(target_range / residual_range, 8.0)
    return trend + residual * gain, float(gain)


def _motif_cum_path(close: np.ndarray, interval: str, horizon: int, k: int = 12) -> tuple[np.ndarray, dict]:
    returns = _returns(close)
    window = {"1d": 64, "1h": 96, "30m": 120, "15m": 144}.get(interval, 96)
    if len(returns) < window + horizon + 10:
        return np.zeros(horizon, dtype=np.float64), {"motif_matches": 0, "motif_distance": None, "path_gain": 1.0}

    current = _window_signature(returns[-window:])
    recent_vol = _recent_vol(returns, window)
    candidates: list[tuple[float, np.ndarray]] = []
    last_start = len(returns) - window - horizon + 1
    for start in range(0, max(0, last_start)):
        hist = returns[start : start + window]
        fut = returns[start + window : start + window + horizon]
        if len(fut) != horizon:
            continue
        sig = _window_signature(hist)
        dist = float(np.mean((current - sig) ** 2))
        fut_scale = max(_safe_std(hist[-max(12, window // 2) :]), 1e-5)
        path = np.cumsum(fut) / fut_scale * recent_vol
        candidates.append((dist, path))

    if not candidates:
        return np.zeros(horizon, dtype=np.float64), {"motif_matches": 0, "motif_distance": None, "path_gain": 1.0}

    candidates.sort(key=lambda item: item[0])
    top = candidates[: min(k, len(candidates))]
    dists = np.asarray([d for d, _ in top], dtype=np.float64)
    weights = np.exp(-dists / max(float(np.median(dists)), 1e-6))
    weights = weights / max(float(np.sum(weights)), 1e-8)
    path = np.sum([w * p for w, (_, p) in zip(weights, top)], axis=0)
    path, gain = _calibrate_amplitude(path, returns, interval, horizon)
    return path, {
        "motif_matches": len(top),
        "motif_distance": float(np.mean(dists)),
        "path_gain": gain,
    }


def _cycle_cum_path(close: np.ndarray, interval: str, horizon: int) -> np.ndarray:
    returns = _returns(close)
    if len(returns) < 24:
        return np.zeros(horizon, dtype=np.float64)
    lookback = min(len(returns), max(48, horizon))
    hist = returns[-lookback:]
    vol = _recent_vol(returns, lookback)
    demeaned = hist - float(np.mean(hist))
    spectrum = np.fft.rfft(demeaned)
    if len(spectrum) <= 2:
        return np.zeros(horizon, dtype=np.float64)
    idx = int(np.argmax(np.abs(spectrum[1:])) + 1)
    amp = min(float(np.abs(spectrum[idx]) / len(hist)) * 2.0, vol * 1.25)
    phase = float(np.angle(spectrum[idx]))
    t = np.arange(len(hist), len(hist) + horizon, dtype=np.float64)
    step = float(np.mean(hist)) + amp * np.cos(2.0 * np.pi * idx * t / len(hist) + phase)
    path, _gain = _calibrate_amplitude(np.cumsum(step), returns, interval, horizon)
    return path


def _drift_cum_path(close: np.ndarray, horizon: int) -> np.ndarray:
    returns = _returns(close)
    lookback = min(len(returns), max(12, horizon // 2))
    drift = float(np.mean(returns[-lookback:])) if lookback else 0.0
    return np.cumsum(np.repeat(drift, horizon))


def _to_prices(base: float, cum_path: np.ndarray) -> np.ndarray:
    return base * np.exp(np.asarray(cum_path, dtype=np.float64))


def forecast_model_comparison(
    close: np.ndarray,
    interval: str,
    horizon: int,
    z_value: float,
    return_clip: float,
    max_log_band: float,
) -> tuple[list[dict], dict]:
    close = _clean_close(close)
    base = float(close[-1])
    motif_path, motif_info = _motif_cum_path(close, interval, horizon)
    cycle_path = _cycle_cum_path(close, interval, horizon)
    drift_path = _drift_cum_path(close, horizon)
    flat_path = np.zeros(horizon, dtype=np.float64)
    lstm_path = _live_sequence_path(close, interval, horizon, "lstm")
    tcn_path = _live_sequence_path(close, interval, horizon, "tcn")

    mlp_mean, mlp_low, mlp_high, mlp_info = forecast_with_global_model(
        close=close,
        interval=interval,
        horizon=horizon,
        z_value=z_value,
        return_clip=return_clip,
        max_log_band=max_log_band,
    )
    mlp_path = np.log(np.asarray(mlp_mean, dtype=np.float64) / base)
    band = np.maximum(
        np.abs(np.log(np.asarray(mlp_high, dtype=np.float64) / np.asarray(mlp_mean, dtype=np.float64))),
        np.abs(np.log(np.asarray(mlp_mean, dtype=np.float64) / np.asarray(mlp_low, dtype=np.float64))),
    )
    band = np.minimum(band, max_log_band)
    if motif_info["motif_matches"] > 0:
        ensemble_path = 0.60 * motif_path + 0.20 * cycle_path + 0.20 * mlp_path
        ensemble_path, _gain = _calibrate_amplitude(ensemble_path, _returns(close), interval, horizon)
    else:
        ensemble_path = 0.60 * mlp_path + 0.25 * cycle_path + 0.15 * drift_path

    models = [
        {
            "id": "motif",
            "label": "Motif",
            "description": "Historical motif analogue",
            "color": "#d29922",
            "values": _to_prices(base, motif_path if motif_info["motif_matches"] > 0 else mlp_path),
        },
        {
            "id": "ensemble",
            "label": "Ensemble",
            "description": "Motif + cycle + MLP",
            "color": "#3fb950",
            "values": _to_prices(base, ensemble_path),
        },
        {
            "id": "pattern_mlp",
            "label": "Pattern MLP",
            "description": "Pattern-aware MLP",
            "color": "#58a6ff",
            "values": np.asarray(mlp_mean, dtype=np.float64),
        },
        {
            "id": "cycle",
            "label": "Cycle",
            "description": "Dominant cycle extrapolation",
            "color": "#bc8cff",
            "values": _to_prices(base, cycle_path),
        },
        {
            "id": "lstm",
            "label": "LSTM",
            "description": "Live cached LSTM path model",
            "color": "#f778ba",
            "values": _to_prices(base, lstm_path) if lstm_path is not None else np.asarray(mlp_mean, dtype=np.float64),
        },
        {
            "id": "tcn",
            "label": "TCN",
            "description": "Live cached temporal CNN path model",
            "color": "#a5d6ff",
            "values": _to_prices(base, tcn_path) if tcn_path is not None else np.asarray(mlp_mean, dtype=np.float64),
        },
        {
            "id": "drift",
            "label": "Drift",
            "description": "Recent drift baseline",
            "color": "#ff7b72",
            "values": _to_prices(base, drift_path),
        },
        {
            "id": "flat",
            "label": "Flat",
            "description": "No-change baseline",
            "color": "#8b949e",
            "values": _to_prices(base, flat_path),
        },
    ]
    info = {
        **mlp_info,
        **motif_info,
        "model_name": "Historical Motif Pattern Model",
        "base_model": mlp_info.get("model_name"),
        "pattern_engine": "historical_motif_analogue",
        "_ci_band_values": band,
    }
    return models, info


def forecast_market_pattern(
    close: np.ndarray,
    interval: str,
    horizon: int,
    z_value: float,
    return_clip: float,
    max_log_band: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    close = _clean_close(close)
    base = float(close[-1])
    models, info = forecast_model_comparison(
        close=close,
        interval=interval,
        horizon=horizon,
        z_value=z_value,
        return_clip=return_clip,
        max_log_band=max_log_band,
    )
    primary = next((m for m in models if m["id"] == "motif"), models[0])
    pred_mean = np.asarray(primary["values"], dtype=np.float64)
    band = np.asarray(info.get("_ci_band_values"), dtype=np.float64)
    pred_low = pred_mean * np.exp(-band)
    pred_high = pred_mean * np.exp(band)
    return pred_mean, pred_low, pred_high, info
