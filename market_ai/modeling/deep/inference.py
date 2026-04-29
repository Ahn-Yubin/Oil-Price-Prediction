from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from market_ai.data.deep_dataset import build_deep_dataset_from_frame
from market_ai.data.event_providers import FileEventProvider, NullEventProvider
from market_ai.modeling.deep.artifacts import load_deep_artifact
from market_ai.modeling.deep.lstm_tcn_fusion import QUANTILE_LEVELS
from market_ai.schemas.deep_learning import DeepDatasetConfig


def predict_deep_quantiles(
    *,
    artifact_path: Path,
    candles: pd.DataFrame,
    symbol: str,
    interval: str,
    horizon: int,
    event_provider: FileEventProvider | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    model, metadata = load_deep_artifact(artifact_path, map_location=device)
    lookback = int(metadata.get("lookback") or 128)
    config = DeepDatasetConfig(
        interval=interval,
        lookback=lookback,
        horizon=horizon,
        min_history=lookback,
        max_samples=1,
        event_context_enabled=event_provider is not None,
        validation_ratio=0.0,
        test_ratio=0.0,
    )
    dataset = build_deep_dataset_from_frame(
        symbol=symbol,
        interval=interval,
        candles=candles,
        config=config,
        event_provider=event_provider or NullEventProvider(),
    )
    if not dataset.samples:
        raise RuntimeError("Not enough candles for deep model inference")
    sample = dataset.samples[-1]
    tensors = dataset.tensors([len(dataset.samples) - 1])
    resolved_device = torch.device(device)
    model = model.to(resolved_device)
    with torch.no_grad():
        out = model(
            torch.tensor(tensors["x_price"], dtype=torch.float32, device=resolved_device),
            torch.tensor(tensors["x_cross_asset"], dtype=torch.float32, device=resolved_device),
            torch.tensor(tensors["x_event_context"], dtype=torch.float32, device=resolved_device),
            torch.tensor(tensors["x_static"], dtype=torch.float32, device=resolved_device),
        )
    quantiles_scaled = out["quantiles"].detach().cpu().numpy()[0]
    recent_vol = max(float(sample.recent_realized_volatility), 1e-8)
    quantiles_log = quantiles_scaled * recent_vol
    quantile_prices = {
        f"p{int(level * 100):02d}": float(sample.current_price) * np.exp(quantiles_log[:, idx])
        for idx, level in enumerate(QUANTILE_LEVELS)
    }
    return {
        "model_name": metadata.get("model_name"),
        "metadata": metadata,
        "current_price": sample.current_price,
        "recent_realized_volatility": recent_vol,
        "quantiles_log": quantiles_log,
        "quantile_prices": quantile_prices,
        "prob_up": out["prob_up"].detach().cpu().numpy()[0],
        "expected_volatility": out["expected_volatility"].detach().cpu().numpy()[0] * recent_vol,
        "confidence": out["confidence"].detach().cpu().numpy()[0],
        "extra": {key: value.detach().cpu().numpy()[0] for key, value in out.items() if key not in {"quantiles", "prob_up", "expected_volatility", "confidence"}},
    }
