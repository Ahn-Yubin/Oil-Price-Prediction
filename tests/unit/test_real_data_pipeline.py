from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

from market_ai.data.deep_dataset import build_deep_dataset_from_frame
from market_ai.data.manifests import entry_from_file, manifest_schema
from market_ai.data.providers.cftc_provider import cot_weekly_to_daily_point_in_time, normalize_cftc_cot_frame
from market_ai.data.providers.cme_provider import normalize_cme_settlements_frame
from market_ai.data.providers.eia_provider import normalize_eia_manual_frame, weekly_to_daily_point_in_time
from market_ai.data.providers.fred_provider import build_fred_wide_panel, normalize_fred_frame
from market_ai.data.providers.market_price_provider import _stooq_daily_url, missing_bars_report, normalize_market_price_frame
from market_ai.data.providers.public_news_provider import gdelt_doc_url, google_news_rss_url, normalize_public_news
from market_ai.schemas.deep_learning import DeepDatasetConfig


def test_manual_csv_ingest_scripts_fail_fast_on_placeholder_paths():
    commands = [
        ["scripts/data/fetch_eia_petroleum.py", "--manual-csv", "path/to/eia.csv", "EIA manual CSV not found"],
        ["scripts/data/fetch_cftc_cot.py", "--manual-csv", "path/to/cftc.csv", "CFTC COT manual CSV not found"],
        ["scripts/data/fetch_cme_settlements.py", "--manual-csv", "path/to/cme.csv", "CME settlements manual CSV not found"],
    ]
    for script, flag, csv_path, expected in commands:
        result = subprocess.run(
            [sys.executable, script, flag, csv_path],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert expected in result.stderr
        assert "Traceback" not in result.stderr


def test_manifest_schema_and_inventory_entry(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("timestamp,symbol,close\n2025-01-01T00:00:00Z,CL=F,70\n", encoding="utf-8")
    schema = manifest_schema()
    assert "dataset_name" in schema["properties"]
    entry = entry_from_file(path, dataset_name="sample", source="unit", point_in_time_safe=True)
    assert entry.rows == 1
    assert entry.point_in_time_safe is True


def test_market_price_normalization_removes_duplicates_and_reports_missing():
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-01", "2025-01-01", "2025-01-03"],
            "open": [70, 71, 73],
            "high": [72, 72, 74],
            "low": [69, 70, 72],
            "close": [71, 72, 73],
            "volume": [1, 2, 3],
        }
    )
    normalized = normalize_market_price_frame(frame, symbol="CL=F", provider="unit")
    assert len(normalized) == 2
    report = missing_bars_report(normalized, interval="1d", symbol="CL=F")
    assert report["missing_timestamp"].str.contains("2025-01-02").any()


def test_stooq_url_uses_daily_csv_params():
    url = _stooq_daily_url("spy.us", period="1y")
    assert "stooq.com/q/d/l/" in url
    assert "s=spy.us" in url
    assert "i=d" in url
    assert "d1=" in url


def test_fred_macro_normalization_and_wide_panel():
    raw = pd.DataFrame(
        {
            "observation_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "DGS10": ["4.2", ".", "4.3"],
        }
    )
    normalized = normalize_fred_frame(raw, series_id="DGS10", label="10-year Treasury")
    assert normalized["series_id"].unique().tolist() == ["DGS10"]
    assert normalized["value"].isna().sum() == 1
    wide = build_fred_wide_panel(normalized)
    assert "DGS10" in wide.columns


def test_public_news_normalization_deduplicates_rows():
    frame = pd.DataFrame(
        {
            "published_at": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "symbol": ["CL=F", "CL=F"],
            "headline": ["OPEC supply cut lifts crude", "OPEC supply cut lifts crude"],
            "body": ["", ""],
            "source": ["unit", "unit"],
            "url": ["https://example.com/a", "https://example.com/a"],
            "retrieved_at": ["2025-01-01T00:05:00Z", "2025-01-01T00:05:00Z"],
        }
    )
    normalized = normalize_public_news([frame])
    assert len(normalized) == 1
    assert normalized.iloc[0]["symbol"] == "CL=F"


def test_public_news_normalization_accepts_mixed_datetime_formats():
    first = pd.DataFrame(
        {
            "published_at": ["2026-03-13 10:40:36+00:00"],
            "symbol": ["CL=F"],
            "headline": ["Yahoo-style timestamp"],
            "body": [""],
            "source": ["yahoo_finance_rss"],
            "url": ["https://example.com/y"],
            "retrieved_at": ["2026-03-13 10:41:00+00:00"],
        }
    )
    second = pd.DataFrame(
        {
            "published_at": ["2026-05-11T05:50:17+00:00"],
            "symbol": ["CL=F"],
            "headline": ["RSS ISO timestamp"],
            "body": [""],
            "source": ["google_news_rss"],
            "url": ["https://example.com/g"],
            "retrieved_at": ["2026-05-11T05:51:00+00:00"],
        }
    )
    normalized = normalize_public_news([first, second])
    assert len(normalized) == 2


def test_gdelt_doc_url_supports_windowed_queries_and_caps_records():
    url = gdelt_doc_url(
        '"crude oil"',
        start_datetime="20260501000000",
        end_datetime="20260508235959",
        maxrecords=999,
    )
    assert "startdatetime=20260501000000" in url
    assert "enddatetime=20260508235959" in url
    assert "timespan=" not in url
    assert "maxrecords=250" in url


def test_google_news_rss_url_uses_public_rss_search_endpoint():
    url = google_news_rss_url('"crude oil"')
    assert url.startswith("https://news.google.com/rss/search?")
    assert "ceid=US%3Aen" in url


def test_fundamental_parsers_and_point_in_time_fill():
    eia = normalize_eia_manual_frame(
        pd.DataFrame(
            {
                "report_date": ["2025-01-03", "2025-01-10"],
                "crude_stocks": [400, 410],
                "crude_imports": [6.0, 6.2],
                "crude_exports": [4.0, 4.1],
            }
        )
    )
    daily = weekly_to_daily_point_in_time(eia, end="2025-01-20")
    available = pd.to_datetime(daily["feature_available_at"], utc=True)
    timestamps = pd.to_datetime(daily["timestamp"], utc=True)
    assert (available[available.notna()] <= timestamps[available.notna()]).all()

    cot = normalize_cftc_cot_frame(
        pd.DataFrame(
            {
                "report_date": ["2025-01-07", "2025-01-14"],
                "market": ["CRUDE OIL", "CRUDE OIL"],
                "managed_money_long": [100, 120],
                "managed_money_short": [80, 70],
                "commercial_long": [50, 50],
                "commercial_short": [90, 100],
                "open_interest": [1000, 1100],
            }
        )
    )
    cot_daily = cot_weekly_to_daily_point_in_time(cot, end="2025-01-25")
    assert "managed_money_net" in cot_daily.columns

    cme = normalize_cme_settlements_frame(
        pd.DataFrame(
            {
                "trade_date": ["2025-01-02"] * 6,
                "contract": ["CLG25", "CLH25", "CLJ25", "CLK25", "CLM25", "CLN25"],
                "settle": [70, 69.5, 69, 68.8, 68.4, 68.0],
            }
        )
    )
    assert cme.loc[0, "m1_m2_spread"] > 0


def test_deep_dataset_processed_features_do_not_look_ahead():
    candles = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=80, freq="D", tz="UTC"),
            "open": [70 + i * 0.1 for i in range(80)],
            "high": [71 + i * 0.1 for i in range(80)],
            "low": [69 + i * 0.1 for i in range(80)],
            "close": [70.5 + i * 0.1 for i in range(80)],
            "volume": [1000] * 80,
        }
    )
    aux = pd.DataFrame(
        {
            "feature_available_at": [
                datetime(2025, 1, 10, tzinfo=timezone.utc),
                datetime(2025, 4, 15, tzinfo=timezone.utc),
            ],
            "m1_m2_spread": [1.0, 99.0],
        }
    )
    config = DeepDatasetConfig(interval="1d", lookback=20, horizon=3, min_history=20, max_samples=3, event_context_enabled=False)
    dataset = build_deep_dataset_from_frame(symbol="CL=F", interval="1d", candles=candles, config=config, auxiliary_frame=aux)
    spread_idx = dataset.cross_asset_feature_names.index("spread")
    assert dataset.samples
    assert max(window[-1][spread_idx] for window in [sample.x_cross_asset for sample in dataset.samples]) < 99.0
