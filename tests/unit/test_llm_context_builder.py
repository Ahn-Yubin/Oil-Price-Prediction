from __future__ import annotations

from datetime import datetime, timezone

import pytest

from market_ai.data.event_providers import FileEventProvider
from market_ai.features.context_features import EVENT_CONTEXT_DIM
from market_ai.llm.context_builder import _recent_news_items, build_event_context_daily
from market_ai.llm.event_encoder import LocalEventContextEncoder
from market_ai.schemas.llm_context import LLMContextOutput


def test_recent_news_items_passes_point_in_time_news_to_llm_context(tmp_path):
    path = tmp_path / "news.csv"
    path.write_text(
        "\n".join(
            [
                "published_at,symbol,headline,source",
                "2026-01-01T00:00:00Z,CL=F,old headline,rss",
                "2026-01-09T00:00:00Z,CL=F,recent headline,rss",
                "2026-01-10T00:00:00Z,BZ=F,wrong symbol,rss",
                "2026-01-11T00:00:00Z,CL=F,future headline,rss",
            ]
        ),
        encoding="utf-8",
    )
    provider = FileEventProvider([path])

    news = _recent_news_items(
        provider,
        symbol="CL=F",
        as_of_time=datetime(2026, 1, 10, tzinfo=timezone.utc),
        lookback_days=7,
    )

    assert [item.title for item in news] == ["recent headline"]


def test_event_context_cache_is_written_incrementally_and_reused(tmp_path, monkeypatch):
    news_path = tmp_path / "news.csv"
    news_path.write_text(
        "\n".join(
            [
                "published_at,symbol,headline,source",
                "2026-01-10T00:00:00Z,CL=F,recent headline,rss",
            ]
        ),
        encoding="utf-8",
    )
    cache_path = tmp_path / "cache.jsonl"

    first, _ = build_event_context_daily(
        symbols=["CL=F"],
        news_paths=[news_path],
        mode="local_rules",
        start="2026-01-10",
        end="2026-01-10",
        cache_path=cache_path,
    )

    assert len(first) == 1
    assert EVENT_CONTEXT_DIM == 27
    assert first.iloc[0]["news_volume_7d"] > 0.0
    assert first.iloc[0]["news_selection_coverage"] == 1.0
    assert first.iloc[0]["source_diversity_score"] > 0.0
    assert cache_path.exists()
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 1

    def fail_encode(self, context):  # noqa: ANN001
        raise AssertionError("cache miss")

    monkeypatch.setattr(LocalEventContextEncoder, "encode_events", fail_encode)
    second, _ = build_event_context_daily(
        symbols=["CL=F"],
        news_paths=[news_path],
        mode="local_rules",
        start="2026-01-10",
        end="2026-01-10",
        cache_path=cache_path,
    )

    assert len(second) == 1
    assert second.iloc[0]["event_count"] == first.iloc[0]["event_count"]


def test_raw_news_pool_features_cover_full_recent_news_window(tmp_path):
    news_path = tmp_path / "news.csv"
    news_path.write_text(
        "\n".join(
            [
                "published_at,symbol,headline,source",
                "2026-01-09T00:00:00Z,CL=F,OPEC output cut supports crude supply,rss-a",
                "2026-01-10T00:00:00Z,CL=F,China demand weak as oil prices fall,rss-b",
            ]
        ),
        encoding="utf-8",
    )

    frame, _ = build_event_context_daily(
        symbols=["CL=F"],
        news_paths=[news_path],
        mode="local_rules",
        start="2026-01-10",
        end="2026-01-10",
        news_limit_per_context=1,
    )

    row = frame.iloc[0]
    assert row["llm_input_news_count"] == 1
    assert 0.0 < row["news_selection_coverage"] < 1.0
    assert row["raw_bullish_pressure"] > 0.0
    assert row["raw_bearish_pressure"] > 0.0
    assert row["raw_energy_pressure"] > 0.0
    assert row["raw_supply_pressure"] > 0.0
    assert row["raw_demand_pressure"] > 0.0


def test_geopolitical_supply_shock_news_is_bullish_supply_context(tmp_path):
    news_path = tmp_path / "news.csv"
    news_path.write_text(
        "\n".join(
            [
                "published_at,symbol,headline,source",
                "2026-06-05T00:00:00Z,CL=F,Oil flat as US Iran war threatens Strait of Hormuz supply,rss-a",
            ]
        ),
        encoding="utf-8",
    )

    frame, raw_events = build_event_context_daily(
        symbols=["CL=F"],
        news_paths=[news_path],
        mode="local_rules",
        start="2026-06-05",
        end="2026-06-05",
    )

    event = raw_events.iloc[0]
    row = frame.iloc[0]
    assert event["event_type"] == "geopolitical_supply_shock"
    assert event["directional_bias"] == "bullish"
    assert event["impact_strength"] >= 0.70
    assert row["raw_geopolitical_pressure"] > 0.0
    assert row["raw_supply_pressure"] > 0.0
    assert row["raw_net_pressure"] > 0.0


def test_external_fallback_is_not_cached(tmp_path, monkeypatch):
    news_path = tmp_path / "news.csv"
    news_path.write_text(
        "\n".join(
            [
                "published_at,symbol,headline,source",
                "2026-01-10T00:00:00Z,CL=F,recent headline,rss",
            ]
        ),
        encoding="utf-8",
    )
    cache_path = tmp_path / "cache.jsonl"

    class FallbackEncoder:
        def encode_events(self, context):  # noqa: ANN001
            return LLMContextOutput(
                events=[],
                overall_bias="neutral",
                impact_score=0.0,
                uncertainty=1.0,
                event_embedding=[0.0] * 13,
                explanation="fallback",
                warnings=["External LLM fallback: quota exceeded"],
            )

    monkeypatch.setattr("market_ai.llm.context_builder.encoder_for_mode", lambda *args, **kwargs: FallbackEncoder())
    frame, _ = build_event_context_daily(
        symbols=["CL=F"],
        news_paths=[news_path],
        mode="google_generative",
        start="2026-01-10",
        end="2026-01-10",
        cache_path=cache_path,
    )

    assert len(frame) == 1
    assert not cache_path.exists()


def test_strict_external_context_rejects_fallback_rows(tmp_path, monkeypatch):
    news_path = tmp_path / "news.csv"
    news_path.write_text(
        "\n".join(
            [
                "published_at,symbol,headline,source",
                "2026-01-10T00:00:00Z,CL=F,recent headline,rss",
            ]
        ),
        encoding="utf-8",
    )
    cache_path = tmp_path / "cache.jsonl"

    class FallbackEncoder:
        def encode_events(self, context):  # noqa: ANN001
            return LLMContextOutput(
                events=[],
                overall_bias="neutral",
                impact_score=0.0,
                uncertainty=1.0,
                event_embedding=[0.0] * 13,
                explanation="fallback",
                warnings=["External LLM fallback: quota exceeded"],
            )

    monkeypatch.setattr("market_ai.llm.context_builder.encoder_for_mode", lambda *args, **kwargs: FallbackEncoder())

    with pytest.raises(RuntimeError, match="prevents local_rules fallback"):
        build_event_context_daily(
            symbols=["CL=F"],
            news_paths=[news_path],
            mode="google_generative",
            start="2026-01-10",
            end="2026-01-10",
            cache_path=cache_path,
            strict_external_llm=True,
        )

    assert not cache_path.exists()


def test_external_contexts_are_batched(tmp_path, monkeypatch):
    news_path = tmp_path / "news.csv"
    news_path.write_text(
        "\n".join(
            [
                "published_at,symbol,headline,source",
                "2026-01-10T00:00:00Z,CL=F,recent headline,rss",
            ]
        ),
        encoding="utf-8",
    )
    cache_path = tmp_path / "cache.jsonl"
    calls: list[int] = []

    class BatchEncoder:
        def encode_event_batch(self, contexts):  # noqa: ANN001
            calls.append(len(contexts))
            return [
                LLMContextOutput(
                    events=[],
                    overall_bias="neutral",
                    impact_score=0.0,
                    uncertainty=1.0,
                    event_embedding=[0.0] * 13,
                    explanation="batched",
                    warnings=[],
                )
                for _ in contexts
            ]

        def encode_events(self, context):  # noqa: ANN001
            raise AssertionError("single encode should not be used")

    monkeypatch.setattr("market_ai.llm.context_builder.encoder_for_mode", lambda *args, **kwargs: BatchEncoder())
    frame, _ = build_event_context_daily(
        symbols=["CL=F"],
        news_paths=[news_path],
        mode="google_generative",
        start="2026-01-10",
        end="2026-01-12",
        cache_path=cache_path,
        llm_batch_size=3,
    )

    assert len(frame) == 3
    assert calls == [3]
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 3
