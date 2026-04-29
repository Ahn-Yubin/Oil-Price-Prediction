from datetime import datetime, timezone

from market_ai.features.context_features import build_event_context_vector


def test_event_context_zero_without_events():
    vector = build_event_context_vector([], as_of_time=datetime(2025, 1, 10, tzinfo=timezone.utc))
    assert vector.event_count_7d == 0
    assert vector.directional_bias_score == 0


def test_event_context_is_point_in_time():
    as_of = datetime(2025, 1, 10, tzinfo=timezone.utc)
    events = [
        {
            "timestamp": "2025-01-09T00:00:00Z",
            "event_type": "energy_supply",
            "directional_bias": "bullish",
            "impact_strength": 0.8,
            "uncertainty": 0.3,
            "source_quality_score": 0.9,
        },
        {
            "timestamp": "2025-01-11T00:00:00Z",
            "event_type": "macro_policy",
            "directional_bias": "bearish",
            "impact_strength": 1.0,
        },
    ]
    vector = build_event_context_vector(events, as_of_time=as_of)
    assert vector.event_count_3d == 1
    assert vector.directional_bias_score > 0
    assert vector.energy_event_score > 0
