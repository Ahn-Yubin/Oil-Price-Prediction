import numpy as np
import pandas as pd

from market_ai.data.deep_dataset import build_deep_dataset_from_frame, build_synthetic_deep_dataset, synthetic_ohlcv
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


def test_multi_symbol_deep_dataset_split_is_chronological_not_symbol_blocked():
    config = DeepDatasetConfig(
        interval="1d",
        symbols=["CL=F", "BZ=F"],
        lookback=20,
        horizon=4,
        min_history=20,
        max_samples=60,
    )
    dataset = build_synthetic_deep_dataset(config)

    def split_times(indices):
        return [pd.Timestamp(dataset.samples[int(idx)].as_of_time) for idx in indices]

    train_times = split_times(dataset.train_indices)
    validation_times = split_times(dataset.validation_indices)
    test_times = split_times(dataset.test_indices)
    assert max(train_times) <= min(validation_times)
    assert max(validation_times) <= min(test_times)
    assert {dataset.samples[int(idx)].symbol for idx in dataset.train_indices} == {"CL=F", "BZ=F"}
    assert {dataset.samples[int(idx)].symbol for idx in dataset.validation_indices} == {"CL=F", "BZ=F"}
    assert {dataset.samples[int(idx)].symbol for idx in dataset.test_indices} == {"CL=F", "BZ=F"}


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


def test_deep_dataset_aggregates_event_context_over_lookback_without_future_rows():
    frame = synthetic_ohlcv(32, start="2020-01-01")
    event_context = pd.DataFrame(
        [
            {
                "date": "2020-01-22T00:00:00Z",
                "symbol": "CL=F",
                "directional_bias_score": 1.0,
                "impact_score": 0.2,
                "uncertainty": 0.1,
                "time_decay": 0.3,
                "event_count_1d": 1.0,
            },
            {
                "date": "2020-01-25T00:00:00Z",
                "symbol": "CL=F",
                "directional_bias_score": -1.0,
                "impact_score": 0.8,
                "uncertainty": 0.4,
                "time_decay": 0.9,
                "event_count_1d": 2.0,
            },
            {
                "date": "2030-01-01T00:00:00Z",
                "symbol": "CL=F",
                "directional_bias_score": 1.0,
                "impact_score": 99.0,
                "uncertainty": 0.0,
                "time_decay": 1.0,
                "event_count_1d": 99.0,
            },
        ]
    )
    config = DeepDatasetConfig(interval="1d", lookback=10, horizon=2, min_history=10, max_samples=1)
    dataset = build_deep_dataset_from_frame(
        symbol="CL=F",
        interval="1d",
        candles=frame,
        config=config,
        event_context_frame=event_context,
    )

    latest = dataset.samples[-1].x_event_context
    end_ts = pd.Timestamp(dataset.samples[-1].as_of_time)
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    event_dates = pd.to_datetime(["2020-01-22T00:00:00Z", "2020-01-25T00:00:00Z"])
    recency = np.asarray(np.exp(-((end_ts - event_dates).total_seconds() / 86_400.0) / 14.0), dtype=float)
    weights = np.array([0.2, 0.8]) * recency
    expected_bias = ((np.array([1.0, -1.0]) * weights).sum()) / weights.sum()
    expected_uncertainty = ((np.array([0.1, 0.4]) * weights).sum()) / weights.sum()
    assert np.isclose(latest[0], expected_bias)
    assert np.isclose(latest[2], expected_uncertainty)
    assert np.isclose(latest[3], recency.max())
    assert np.isclose(latest[4], 3.0)
