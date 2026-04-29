from pathlib import Path

from market_ai.data.event_providers import FileEventProvider, load_events_from_file


def test_sample_event_file_schema():
    path = Path("data/external/events/sample_market_events.csv")
    events = load_events_from_file(path)
    assert events
    assert events[0].timestamp.tzinfo is not None
    assert 0 <= events[0].impact_strength <= 1


def test_file_event_provider_symbol_filter_and_as_of():
    provider = FileEventProvider(["data/external/events/sample_market_events.csv"])
    events = provider.events_as_of(symbol="CL=F", as_of_time=events_time("2025-01-10T00:00:00Z"))
    assert events
    assert all(event.symbol in {"CL=F", "ALL"} for event in events)


def events_time(value: str):
    import pandas as pd

    return pd.Timestamp(value).to_pydatetime()
