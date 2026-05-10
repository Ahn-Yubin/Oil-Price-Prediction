from datetime import datetime, timezone

from market_ai.config import Settings
from market_ai.data.event_providers import FileEventProvider
from market_ai.llm.event_encoder import (
    GoogleGenerativeLLMEventEncoder,
    LocalEventContextEncoder,
    LocalHTTPLLMEventEncoder,
    OpenAICompatibleLLMEventEncoder,
    encoder_from_settings,
)
from market_ai.llm.event_encoder import _default_https_context
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
    assert any("External LLM disabled" in warning for warning in output.warnings)
    assert "target_price" not in output.model_dump_json().lower()


def test_external_llm_enabled_without_key_warns():
    encoder = OpenAICompatibleLLMEventEncoder(api_key=None, model="test", enabled=True)
    output = encoder.encode_events(MarketContextInput(symbol="CL=F", interval="1d"))
    assert any("API key missing" in warning for warning in output.warnings)


def test_encoder_from_settings_does_not_enable_external_calls_by_default():
    encoder = encoder_from_settings(Settings(enable_llm_context=True, app_env="production", llm_api_key="dummy"))
    assert isinstance(encoder, GoogleGenerativeLLMEventEncoder)
    assert encoder.enabled is False


def test_local_http_llm_disabled_uses_local_rules(tmp_path):
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
    encoder = LocalHTTPLLMEventEncoder(
        api_base="http://localhost:1",
        model="local",
        enabled=False,
        fallback_provider=FileEventProvider([path]),
    )
    output = encoder.encode_events(MarketContextInput(symbol="CL=F", interval="1d", generated_at=datetime(2025, 1, 2, tzinfo=timezone.utc)))
    assert output.events
    assert any("dry-run" in warning for warning in output.warnings)


def test_external_llm_uses_https_context():
    context = _default_https_context()
    assert context.verify_mode.name == "CERT_REQUIRED"


def test_google_generative_url_uses_native_gemma_endpoint():
    encoder = GoogleGenerativeLLMEventEncoder(
        api_key="secret",
        model="gemma-3-27b-it",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        enabled=True,
    )

    url = encoder._url()

    assert "/models/gemma-3-27b-it:generateContent" in url
    assert "/openai/" not in url
