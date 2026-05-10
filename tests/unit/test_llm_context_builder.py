from __future__ import annotations

from datetime import datetime, timezone

from market_ai.data.event_providers import FileEventProvider
from market_ai.llm.context_builder import _recent_news_items


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
