from datetime import datetime, timezone

from market_ai.config import Settings
from market_ai.data.event_providers import FileEventProvider
from market_ai.llm.event_encoder import LocalEventContextEncoder, OpenAICompatibleLLMEventEncoder, encoder_from_settings
from market_ai.schemas.llm_context import MarketContextInput


def test_local_event_context_encoder_uses_file_events(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(
        "\n".join(
            [
                "timestamp,symbol,event_type,directional_bias,impact_strength,uncertainty,source_quality_score,summary,source",
                "2025-01-01T00:00:00Z,CL=F,energy_supply,bullish,0.8,0.2,0.9,supply,sample",
            ]
        ),
        encoding="utf-8",
    )
    encoder = LocalEventContextEncoder(FileEventProvider([path]))
    output = encoder.encode_events(MarketContextInput(symbol="CL=F", interval="1d", generated_at=datetime(2025, 1, 2, tzinfo=timezone.utc)))
    assert output.events
    assert output.overall_bias == "bullish"
    assert output.event_embedding


def test_external_llm_disabled_falls_back_without_api_key():
    encoder = OpenAICompatibleLLMEventEncoder(api_key=None, model="test", enabled=False)
    output = encoder.encode_events(MarketContextInput(symbol="CL=F", interval="1d"))
    assert output.warnings
    assert "target_price" not in output.model_dump_json().lower()


def test_encoder_from_settings_does_not_enable_external_calls_by_default():
    encoder = encoder_from_settings(Settings(enable_llm_context=True, app_env="production", llm_api_key="dummy"))
    assert isinstance(encoder, OpenAICompatibleLLMEventEncoder)
    assert encoder.enabled is False
