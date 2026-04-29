from market_ai.data.deep_dataset import build_synthetic_deep_dataset
from market_ai.data.symbol_universe import resolve_universe
from market_ai.schemas.deep_learning import DeepDatasetConfig


def test_deep_dataset_universe_smoke():
    symbols = resolve_universe("oil_core")[:2]
    dataset = build_synthetic_deep_dataset(
        DeepDatasetConfig(interval="1d", symbols=symbols, lookback=20, horizon=4, min_history=20, max_samples=32)
    )
    assert dataset.samples
    assert {sample.symbol for sample in dataset.samples}.issubset(set(symbols))
