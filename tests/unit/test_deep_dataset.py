import numpy as np
import pandas as pd

from market_ai.data.deep_dataset import build_deep_dataset_from_frame, synthetic_ohlcv
from market_ai.data.event_providers import FileEventProvider
from market_ai.schemas.deep_learning import DeepDatasetConfig


def test_synthetic_deep_dataset_shapes_and_split():
    config = DeepDatasetConfig(interval="1d", lookback=24, horizon=5, min_history=24, max_samples=40)
    dataset = build_deep_dataset_from_frame(symbol="CL=F", interval="1d", candles=synthetic_ohlcv(90), config=config)
    assert dataset.samples
    tensors = dataset.tensors()
    assert tensors["x_price"].shape[1:] == (24, len(dataset.price_feature_names))
    assert tensors["y_vol_scaled_cum_return"].shape[1] == 5
    assert max(dataset.train_indices) < min(dataset.validation_indices)
    assert max(dataset.validation_indices) < min(dataset.test_indices)


def test_deep_dataset_zero_context_without_events():
    config = DeepDatasetConfig(interval="1d", lookback=20, horizon=3, min_history=20, max_samples=5, event_context_enabled=False)
    dataset = build_deep_dataset_from_frame(symbol="CL=F", interval="1d", candles=synthetic_ohlcv(60), config=config)
    assert np.asarray(dataset.samples[-1].x_event_context).sum() == 1.0


def test_deep_dataset_event_no_lookahead(tmp_path):
    events = tmp_path / "events.csv"
    events.write_text(
        "\n".join(
            [
                "timestamp,symbol,event_type,directional_bias,impact_strength,uncertainty,source_quality_score,summary,source",
                "2020-01-25T00:00:00Z,CL=F,energy_supply,bullish,0.7,0.2,1.0,past,past",
                "2030-01-01T00:00:00Z,CL=F,energy_supply,bearish,1.0,0.1,1.0,future,future",
            ]
        ),
        encoding="utf-8",
    )
    frame = synthetic_ohlcv(70, start="2020-01-01")
    config = DeepDatasetConfig(interval="1d", lookback=20, horizon=3, min_history=20, max_samples=20)
    dataset = build_deep_dataset_from_frame(
        symbol="CL=F",
        interval="1d",
        candles=frame,
        config=config,
        event_provider=FileEventProvider([events]),
    )
    contexts = pd.DataFrame([sample.x_event_context for sample in dataset.samples])
    assert contexts.iloc[-1, 0] >= 0
    assert contexts.iloc[-1, 7] >= 0
