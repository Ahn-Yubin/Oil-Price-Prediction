import pandas as pd

from market_ai.llm.live_context import _filter_relevant_news


def test_oil_live_news_filter_keeps_price_relevant_headlines():
    frame = pd.DataFrame(
        [
            {
                "published_at": "2026-05-19T00:00:00Z",
                "headline": "Bay Area climate fight erupts over bid to delay natural gas water heater ban",
                "body": "",
                "source": "google_news",
            },
            {
                "published_at": "2026-05-19T01:00:00Z",
                "headline": "Oil prices rise as OPEC supply cuts tighten crude market",
                "body": "",
                "source": "google_news",
            },
        ]
    )

    filtered, warnings = _filter_relevant_news("CL=F", frame)

    assert warnings == []
    assert filtered["headline"].tolist() == ["Oil prices rise as OPEC supply cuts tighten crude market"]


def test_non_oil_symbols_do_not_use_oil_filter():
    frame = pd.DataFrame(
        [
            {
                "published_at": "2026-05-19T00:00:00Z",
                "headline": "S&P 500 futures edge higher before Fed minutes",
                "body": "",
                "source": "google_news",
            }
        ]
    )

    filtered, warnings = _filter_relevant_news("DX-Y.NYB", frame)

    assert warnings == []
    assert filtered.equals(frame)
