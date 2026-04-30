from argparse import Namespace
from pathlib import Path

import pytest

from market_ai.data.deep_dataset import synthetic_ohlcv
from market_ai.schemas.deep_learning import DeepDatasetConfig
from scripts.train import train_deep_fusion_models as trainer


def _args(**overrides):
    values = {
        "synthetic": False,
        "quick_test": False,
        "allow_synthetic_fallback": False,
        "events_path": "",
        "interval": "1d",
    }
    values.update(overrides)
    return Namespace(**values)


def _config() -> DeepDatasetConfig:
    return DeepDatasetConfig(
        interval="1d",
        symbols=["CL=F"],
        lookback=16,
        horizon=3,
        min_history=16,
        max_samples=12,
        event_context_enabled=True,
    )


def test_events_path_is_passed_to_file_event_provider(tmp_path: Path, monkeypatch):
    events_path = tmp_path / "events.csv"
    events_path.write_text(
        "\n".join(
            [
                "timestamp,symbol,event_type,directional_bias,impact_strength,uncertainty,source_quality_score,summary",
                "2020-01-20T00:00:00Z,CL=F,energy_supply,bullish,0.9,0.2,0.9,test event",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(trainer, "_download_frame", lambda symbol, interval: synthetic_ohlcv(80, seed=7))

    dataset, report = trainer.build_dataset(_args(events_path=str(events_path)), ["CL=F"], _config())

    assert report["events_path"] == [str(events_path)]
    assert dataset.samples
    assert max(abs(value) for value in dataset.samples[-1].x_event_context) > 0.0


def test_yfinance_failure_without_synthetic_fallback_fails(monkeypatch):
    monkeypatch.setattr(trainer, "_download_frame", lambda symbol, interval: None)

    with pytest.raises(RuntimeError, match="Production training does not use synthetic fallback"):
        trainer.build_dataset(_args(), ["CL=F"], _config())


def test_allow_synthetic_fallback_is_explicit(monkeypatch):
    monkeypatch.setattr(trainer, "_download_frame", lambda symbol, interval: None)

    dataset, report = trainer.build_dataset(_args(allow_synthetic_fallback=True), ["CL=F"], _config())

    assert report["source"] == "synthetic_fallback"
    assert report["synthetic_used"] is True
    assert dataset.samples
