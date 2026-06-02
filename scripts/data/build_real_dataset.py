#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_ai.env import load_project_env

load_project_env()

from market_ai.config import PROJECT_DIR
from market_ai.data.manifests import entry_from_file, upsert_inventory_entries
from market_ai.data.market_panel import build_missing_bars_report, save_market_panel
from market_ai.data.providers.fred_provider import DEFAULT_FRED_SERIES, build_fred_wide_panel, fetch_fred_series
from market_ai.data.providers.market_price_provider import (
    StooqMarketPriceProvider,
    YFinanceMarketPriceProvider,
    combine_market_frames,
    write_market_cache,
)
from market_ai.data.providers.public_news_provider import (
    NEWS_TOPIC_QUERIES,
    fetch_gdelt_articles,
    fetch_yahoo_finance_rss,
    normalize_public_news,
)
from market_ai.data.storage import DATA_ROOT, ensure_data_lake, write_table
from market_ai.data.symbol_universe import load_symbol_universe
from market_ai.llm.context_builder import build_event_context_daily


UNIVERSE_PATH = PROJECT_DIR / "configs" / "symbol_universe.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a real multi-source oil forecasting dataset.")
    parser.add_argument("--universe", default="oil_core")
    parser.add_argument("--symbols", default="", help="Comma-separated override for the symbol universe.")
    parser.add_argument("--interval", choices=["1d", "1h", "30m", "15m"], default="1d")
    parser.add_argument("--period", default="10y")
    parser.add_argument("--news-timespan", default="30d", help="GDELT DOC API timespan, e.g. 7d, 30d, 3m.")
    parser.add_argument("--news-maxrecords", type=int, default=75)
    parser.add_argument("--skip-stooq-secondary", action="store_true")
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-fred", action="store_true")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    return parser.parse_args()


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _timespan_days(timespan: str) -> int | None:
    value = timespan.strip().lower()
    if not value:
        return None
    try:
        count = int(value[:-1])
    except ValueError:
        return None
    unit = value[-1]
    if unit == "d":
        return count
    if unit == "w":
        return count * 7
    if unit == "m":
        return count * 31
    return None


def resolve_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return _ordered_unique(_split(args.symbols))
    universe = load_symbol_universe(UNIVERSE_PATH)
    return universe.get(args.universe, universe.get("default_global", ["CL=F"]))


def fetch_prices(
    *,
    symbols: list[str],
    interval: str,
    period: str,
    data_root: Path,
    universe_name: str,
    include_secondary_stooq: bool,
) -> tuple[Path, list[Path]]:
    yfinance = YFinanceMarketPriceProvider()
    stooq = StooqMarketPriceProvider()
    provider_chain = [yfinance, stooq]
    panel_frames: list[pd.DataFrame] = []
    written_paths: list[Path] = []
    status_rows: list[dict[str, object]] = []

    for symbol in symbols:
        selected_provider = ""
        for provider in provider_chain:
            try:
                frame = provider.fetch(symbol, interval=interval, period=period)
                path = write_market_cache(frame, root=data_root, provider=provider.provider_name, interval=interval, symbol=symbol)
                written_paths.append(path)
                panel_frames.append(frame)
                selected_provider = provider.provider_name
                status_rows.append(
                    {
                        "symbol": symbol,
                        "provider": provider.provider_name,
                        "interval": interval,
                        "role": "panel",
                        "status": "ok",
                        "rows": len(frame),
                        "path": str(path.relative_to(PROJECT_DIR)) if path.is_relative_to(PROJECT_DIR) else str(path),
                        "error": "",
                    }
                )
                print(f"{symbol}: panel source={provider.provider_name} rows={len(frame)} path={path}")
                break
            except Exception as exc:
                status_rows.append(
                    {
                        "symbol": symbol,
                        "provider": provider.provider_name,
                        "interval": interval,
                        "role": "panel",
                        "status": "failed",
                        "rows": 0,
                        "path": "",
                        "error": str(exc),
                    }
                )

        if include_secondary_stooq and interval == "1d" and selected_provider != "stooq":
            try:
                secondary = stooq.fetch(symbol, interval=interval, period=period)
                secondary_path = write_market_cache(
                    secondary,
                    root=data_root,
                    provider=stooq.provider_name,
                    interval=interval,
                    symbol=symbol,
                )
                written_paths.append(secondary_path)
                status_rows.append(
                    {
                        "symbol": symbol,
                        "provider": stooq.provider_name,
                        "interval": interval,
                        "role": "secondary_raw",
                        "status": "ok",
                        "rows": len(secondary),
                        "path": str(secondary_path.relative_to(PROJECT_DIR))
                        if secondary_path.is_relative_to(PROJECT_DIR)
                        else str(secondary_path),
                        "error": "",
                    }
                )
                print(f"{symbol}: secondary source=stooq rows={len(secondary)} path={secondary_path}")
            except Exception as exc:
                status_rows.append(
                    {
                        "symbol": symbol,
                        "provider": stooq.provider_name,
                        "interval": interval,
                        "role": "secondary_raw",
                        "status": "failed",
                        "rows": 0,
                        "path": "",
                        "error": str(exc),
                    }
                )

    if not panel_frames:
        raise SystemExit("No real market price data was fetched; refusing to create synthetic fallback.")

    panel = combine_market_frames(panel_frames)
    panel_path = save_market_panel(panel, interval=interval, data_root=data_root)
    missing = build_missing_bars_report(panel, interval=interval)
    missing_path = data_root / "interim" / "market" / f"missing_bars_{universe_name}_{interval}.csv"
    status_path = data_root / "interim" / "market" / f"fetch_status_{universe_name}_{interval}.csv"
    write_table(missing, missing_path)
    write_table(pd.DataFrame(status_rows), status_path)
    print(f"price panel: rows={len(panel)} path={panel_path}")
    print(f"price status: rows={len(status_rows)} path={status_path}")
    return panel_path, [missing_path, status_path, *written_paths]


def fetch_news(*, symbols: list[str], data_root: Path, timespan: str, maxrecords: int) -> tuple[Path | None, Path]:
    frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []
    for symbol in symbols:
        try:
            frame = fetch_yahoo_finance_rss(symbol)
            frames.append(frame)
            status_rows.append({"provider": "yahoo_finance_rss", "topic_or_symbol": symbol, "status": "ok", "rows": len(frame), "error": ""})
            print(f"{symbol}: yahoo rss news rows={len(frame)}")
        except Exception as exc:
            status_rows.append({"provider": "yahoo_finance_rss", "topic_or_symbol": symbol, "status": "failed", "rows": 0, "error": str(exc)})

    for topic, query in NEWS_TOPIC_QUERIES.items():
        try:
            frame = fetch_gdelt_articles(topic, query, timespan=timespan, maxrecords=maxrecords)
            frames.append(frame)
            status_rows.append({"provider": "gdelt_doc_api", "topic_or_symbol": topic, "status": "ok", "rows": len(frame), "error": ""})
            print(f"{topic}: gdelt news rows={len(frame)}")
        except Exception as exc:
            status_rows.append({"provider": "gdelt_doc_api", "topic_or_symbol": topic, "status": "failed", "rows": 0, "error": str(exc)})

    news = normalize_public_news(frames)
    days = _timespan_days(timespan)
    if days is not None and not news.empty:
        cutoff = pd.Timestamp(datetime.now(timezone.utc) - pd.Timedelta(days=days))
        news = news[pd.to_datetime(news["published_at"], errors="coerce", utc=True) >= cutoff].reset_index(drop=True)
    status_path = data_root / "interim" / "events" / "public_news_fetch_status.csv"
    write_table(pd.DataFrame(status_rows), status_path)
    if news.empty:
        print("news: no public news rows fetched")
        return None, status_path
    news_path = data_root / "raw" / "news" / "public_market_news.csv"
    write_table(news, news_path)
    print(f"news: rows={len(news)} path={news_path}")
    return news_path, status_path


def build_event_context_from_news(*, symbols: list[str], news_path: Path, data_root: Path) -> list[Path]:
    context_path = data_root / "processed" / "event_context" / "event_context_daily.csv"
    cache_path = data_root / "processed" / "event_context" / "llm_context_cache.jsonl"
    raw_path = data_root / "interim" / "events" / "combined_market_events.csv"
    context_frame, raw_events = build_event_context_daily(
        symbols=symbols,
        news_paths=[news_path],
        mode="local_rules",
        end=pd.Timestamp(datetime.now(timezone.utc)).floor("D").isoformat(),
        cache_path=cache_path,
    )
    write_table(raw_events, raw_path)
    write_table(context_frame, context_path)
    print(f"event context: rows={len(context_frame)} path={context_path}")
    return [raw_path, context_path, cache_path]


def fetch_fred_macro(*, data_root: Path) -> list[Path]:
    frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []
    for series_id, label in DEFAULT_FRED_SERIES.items():
        try:
            frame = fetch_fred_series(series_id, label=label)
            frames.append(frame)
            status_rows.append({"series_id": series_id, "status": "ok", "rows": len(frame), "error": ""})
            print(f"{series_id}: fred rows={len(frame)}")
        except Exception as exc:
            status_rows.append({"series_id": series_id, "status": "failed", "rows": 0, "error": str(exc)})

    status_path = data_root / "interim" / "fundamentals" / "fred_fetch_status.csv"
    write_table(pd.DataFrame(status_rows), status_path)
    if not frames:
        print("fred: no macro rows fetched")
        return [status_path]

    long_frame = pd.concat(frames, ignore_index=True).sort_values(["series_id", "date"])
    wide_frame = build_fred_wide_panel(long_frame)
    long_path = data_root / "raw" / "macro" / "fred_daily_long.csv"
    wide_path = data_root / "processed" / "macro_panel" / "fred_daily_wide.csv"
    write_table(long_frame, long_path)
    write_table(wide_frame, wide_path)
    print(f"fred long: rows={len(long_frame)} path={long_path}")
    print(f"fred wide: rows={len(wide_frame)} path={wide_path}")
    return [status_path, long_path, wide_path]


def write_summary(*, data_root: Path, universe: str, symbols: list[str], sources: list[str], paths: list[Path]) -> Path:
    summary_path = data_root / "manifests" / "real_dataset_build_summary.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe": universe,
        "symbols": symbols,
        "sources": sources,
        "paths": [str(path.relative_to(PROJECT_DIR)) if path.is_relative_to(PROJECT_DIR) else str(path) for path in paths],
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def _providers_for_manifest_path(path: Path, *, stooq_enabled: bool) -> str:
    parts = set(path.parts)
    name = path.name
    if "market" in parts or "market_panel" in parts:
        providers = ["yfinance"]
        if stooq_enabled:
            providers.append("stooq")
        return ",".join(providers)
    if "macro" in parts or "fred" in name:
        return "fred"
    if "news" in parts or name == "public_news_fetch_status.csv":
        return "yahoo_finance_rss,gdelt_doc_api"
    if "events" in parts or "event_context" in parts:
        return "yahoo_finance_rss,gdelt_doc_api,local_rules"
    return "unknown"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    data_root = Path(args.data_root)
    ensure_data_lake(data_root)
    symbols = resolve_symbols(args)
    stooq_enabled = bool(os.environ.get("STOOQ_API_KEY", "").strip())
    sources = ["yfinance", "yahoo_finance_rss", "gdelt_doc_api", "fred"]
    if stooq_enabled:
        sources.insert(1, "stooq")
    elif not args.skip_stooq_secondary:
        print("stooq secondary: skipped because STOOQ_API_KEY is not set")
    print(f"selected universe={args.universe} symbols={','.join(symbols)}")

    paths: list[Path] = []
    panel_path, price_paths = fetch_prices(
        symbols=symbols,
        interval=args.interval,
        period=args.period,
        data_root=data_root,
        universe_name=args.universe,
        include_secondary_stooq=not args.skip_stooq_secondary and stooq_enabled,
    )
    paths.extend([panel_path, *price_paths])

    if not args.skip_news:
        news_path, news_status_path = fetch_news(
            symbols=symbols,
            data_root=data_root,
            timespan=args.news_timespan,
            maxrecords=args.news_maxrecords,
        )
        paths.append(news_status_path)
        if news_path is not None:
            paths.append(news_path)
            paths.extend(build_event_context_from_news(symbols=symbols, news_path=news_path, data_root=data_root))

    if not args.skip_fred:
        paths.extend(fetch_fred_macro(data_root=data_root))

    summary_path = write_summary(data_root=data_root, universe=args.universe, symbols=symbols, sources=sources, paths=paths)
    manifest_entries = []
    for path in paths:
        if path.exists() and path.suffix.lower() in {".csv", ".json", ".jsonl", ".parquet"}:
            source = "macro" if "macro" in path.parts or "fred" in path.name else "news" if "news" in path.parts else "market"
            if "event" in path.parts:
                source = "events"
            relative_parts = path.relative_to(data_root).parts if path.is_relative_to(data_root) else path.parts
            point_in_time_safe = (
                relative_parts[:3] == ("processed", "market_panel", args.interval)
                or relative_parts[:2] == ("processed", "event_context")
            )
            manifest_entries.append(
                entry_from_file(
                    path,
                    dataset_name="_".join(path.relative_to(data_root).with_suffix("").parts),
                    source=source,
                    source_url_or_provider=_providers_for_manifest_path(path, stooq_enabled=stooq_enabled),
                    point_in_time_safe=point_in_time_safe,
                )
            )
    upsert_inventory_entries(manifest_entries)
    print(f"summary: path={summary_path}")


if __name__ == "__main__":
    main()
