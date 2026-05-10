from datetime import datetime, timezone

from market_ai.config import Settings
from market_ai.schemas.llm_context import MarketContextInput, RawNewsItem
from market_ai.llm.event_encoder import (
    MockLLMEventEncoder,
    NullLLMEventEncoder,
    deterministic_explanation,
    encoder_from_settings,
    parse_llm_context_json,
)


def test_llm_schema_parse_valid_json():
    raw = """
    {
      "events": [
        {
          "event_type": "macro",
          "affected_assets": ["CL=F"],
          "directional_bias": "mixed",
          "impact_strength": 0.4,
          "uncertainty": 0.6,
          "time_decay": 0.5,
          "summary": "Inventory event",
          "risk_factors": ["headline risk"]
        }
      ],
      "overall_bias": "mixed",
      "impact_score": 0.4,
      "uncertainty": 0.6,
      "event_embedding": [0.4, 0.6, 0.5],
      "explanation": "Context only",
      "warnings": []
    }
    """
    parsed = parse_llm_context_json(raw)
    assert parsed.events[0].affected_assets == ["CL=F"]
    assert parsed.events[0].impact_strength == 0.4


def test_invalid_json_falls_back_safely():
    parsed = parse_llm_context_json("{bad json")
    assert parsed.events == []
    assert parsed.warnings


def test_json_fence_is_parsed_safely():
    parsed = parse_llm_context_json(
        """```json
        {"events": [], "overall_bias": "neutral", "impact_score": 0, "uncertainty": 1, "event_embedding": [], "explanation": "ok", "warnings": []}
        ```"""
    )
    assert parsed.overall_bias == "neutral"


def test_json_can_be_extracted_from_model_preface():
    parsed = parse_llm_context_json(
        'Here is the JSON: {"events": [], "overall_bias": "mixed", "impact_score": 0, "uncertainty": 1, "event_embedding": [], "explanation": "ok", "warnings": []}'
    )
    assert parsed.overall_bias == "mixed"


def test_llm_qualitative_scores_are_sanitized():
    parsed = parse_llm_context_json(
        """
        {
          "events": [
            {
              "event_type": "energy_news",
              "affected_assets": ["CL=F"],
              "directional_bias": "Bullish",
              "impact_strength": "Medium",
              "uncertainty": "Low",
              "time_decay": "Short-term",
              "summary": "Inventory draw",
              "risk_factors": []
            }
          ],
          "overall_bias": "Bullish",
          "impact_score": "High",
          "uncertainty": "Low",
          "event_embedding": [],
          "explanation": "ok",
          "warnings": []
        }
        """
    )

    assert parsed.overall_bias == "bullish"
    assert parsed.events[0].directional_bias == "bullish"
    assert parsed.events[0].impact_strength == 0.5
    assert parsed.uncertainty == 0.25


def test_single_item_json_array_is_accepted():
    parsed = parse_llm_context_json(
        '[{"events": [], "overall_bias": "neutral", "impact_score": 0, "uncertainty": 1, "event_embedding": [], "explanation": "ok", "warnings": []}]'
    )
    assert parsed.overall_bias == "neutral"


def test_mock_encoder_does_not_emit_numeric_price_forecast():
    context = MarketContextInput(
        symbol="CL=F",
        interval="1d",
        news=[RawNewsItem(title="OPEC meeting", published_at=datetime.now(timezone.utc))],
    )
    output = MockLLMEventEncoder().encode_events(context)
    dumped = output.model_dump()
    assert "price" not in dumped
    assert output.events[0].affected_assets == ["CL=F"]


def test_disabled_encoder_and_deterministic_explanation():
    encoder = encoder_from_settings(Settings(enable_llm_context=False))
    assert isinstance(encoder, NullLLMEventEncoder)
    context = encoder.encode_events(MarketContextInput(symbol="CL=F", interval="1d"))
    explanation = deterministic_explanation(
        symbol="CL=F",
        interval="1d",
        forecast_summary={"confidence": 0.3, "regime": "range"},
        data_status={"status": "real"},
        llm_context=context,
    )
    assert explanation.mode == "deterministic"
    assert explanation.confidence_warning
