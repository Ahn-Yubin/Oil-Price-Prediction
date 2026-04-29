from market_ai.llm.event_encoder import parse_llm_context_json


def test_llm_prompt_safety_rejects_price_override_fields():
    raw = """
    {
      "target_price": 100,
      "p90": 120,
      "events": [],
      "overall_bias": "bullish",
      "impact_score": 0.2,
      "uncertainty": 0.5,
      "event_embedding": [0.2, 0.5, 0.1],
      "explanation": "target_price should be ignored",
      "warnings": []
    }
    """
    output = parse_llm_context_json(raw)
    assert any("forbidden numeric forecast" in warning for warning in output.warnings)
    assert "target_price" not in output.explanation
