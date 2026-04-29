from market_ai.data.deep_dataset import build_synthetic_deep_dataset
from market_ai.modeling.deep.training import train_deep_model
from market_ai.schemas.deep_learning import DeepDatasetConfig


def test_train_deep_fusion_quick_synthetic_smoke():
    dataset = build_synthetic_deep_dataset(
        DeepDatasetConfig(interval="1d", symbols=["CL=F"], lookback=16, horizon=3, min_history=16, max_samples=24)
    )
    result = train_deep_model("deep_lstm_tcn_fusion", dataset, epochs=1, batch_size=8, device="cpu")
    assert result.epochs_ran == 1
    assert result.train_loss == result.train_loss
