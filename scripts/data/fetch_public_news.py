#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_ai.config import PROJECT_DIR
from market_ai.data.manifests import entry_from_file, upsert_inventory_entries
from market_ai.data.providers.public_news_provider import (
    GOOGLE_NEWS_TOPIC_QUERIES,
    NEWS_TOPIC_QUERIES,
    fetch_gdelt_articles,
    fetch_google_news_rss,
    fetch_yahoo_finance_rss,
    normalize_public_news,
)
from market_ai.data.storage import DATA_ROOT, ensure_data_lake, read_table, write_table
from market_ai.data.symbol_universe import load_symbol_universe


UNIVERSE_PATH = PROJECT_DIR / "configs" / "symbol_universe.yaml"

TOPIC_SYMBOLS = {
    "energy": ("CL=F", "BZ=F", "NG=F", "RB=F", "HO=F", "USO", "XLE"),
    "metals": ("GC=F", "SI=F", "HG=F"),
    "fx_macro": ("DX-Y.NYB", "EURUSD=X", "USDKRW=X", "JPY=X"),
    "equities_vol": ("SPY", "QQQ", "^GSPC", "^VIX", "XLE"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch public oil-market news from Yahoo RSS and GDELT DOC API.")
    parser.add_argument("--universe", default="oil_core")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol override.")
    parser.add_argument("--output", default=str(DATA_ROOT / "raw" / "news" / "public_market_news.csv"))
    parser.add_argument("--status-output", default=str(DATA_ROOT / "interim" / "events" / "public_news_fetch_status.csv"))
    parser.add_argument("--skip-yahoo", action="store_true")
    parser.add_argument("--skip-google-news", action="store_true")
    parser.add_argument("--skip-gdelt", action="store_true")
    parser.add_argument("--merge-existing", action="store_true", default=True)
    parser.add_argument("--replace", dest="merge_existing", action="store_false")
    parser.add_argument("--gdelt-start", default="", help="UTC start date/datetime. Defaults to --gdelt-days ago.")
    parser.add_argument("--gdelt-end", default="", help="UTC end date/datetime. Defaults to now.")
    parser.add_argument("--gdelt-days", type=int, default=90)
    parser.add_argument("--gdelt-window-days", type=int, default=7)
    parser.add_argument("--gdelt-maxrecords", type=int, default=250)
    parser.add_argument("--gdelt-sleep", type=float, default=6.0)
    parser.add_argument("--gdelt-retries", type=int, default=4)
    parser.add_argument("--topics", default=",".join(NEWS_TOPIC_QUERIES), help="Comma-separated GDELT topics.")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    return parser.parse_args()


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return _split(args.symbols)
    universe = load_symbol_universe(UNIVERSE_PATH)
    return universe.get(args.universe, universe.get("default_global", ["CL=F"]))


def _gdelt_datetime(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y%m%d%H%M%S")


def _date_windows(args: argparse.Namespace) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    end = pd.to_datetime(args.gdelt_end, utc=True) if args.gdelt_end else pd.Timestamp(datetime.now(timezone.utc))
    start = pd.to_datetime(args.gdelt_start, utc=True) if args.gdelt_start else end - pd.Timedelta(days=max(1, args.gdelt_days))
    step = pd.Timedelta(days=max(1, args.gdelt_window_days))
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        next_cursor = min(cursor + step, end)
        windows.append((cursor, next_cursor))
        cursor = next_cursor
    return windows


def _expand_topic_symbols(frame: pd.DataFrame, topic: str, selected_symbols: set[str]) -> pd.DataFrame:
    symbols = [symbol for symbol in TOPIC_SYMBOLS.get(topic, ("ALL",)) if symbol in selected_symbols]
    if not symbols:
        symbols = ["ALL"]
    expanded = []
    for symbol in symbols:
        copy = frame.copy()
        copy["symbol"] = symbol
        expanded.append(copy)
    return pd.concat(expanded, ignore_index=True) if expanded else frame


def fetch_news(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = _resolve_symbols(args)
    selected_symbols = set(symbols)
    frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []
    output = Path(args.output)

    if args.merge_existing and output.exists():
        try:
            existing = read_table(output)
            frames.append(existing)
            status_rows.append({"provider": "existing_file", "topic_or_symbol": str(output), "status": "ok", "rows": len(existing), "error": ""})
            print(f"[news] existing rows={len(existing)} path={output}", flush=True)
        except Exception as exc:
            status_rows.append({"provider": "existing_file", "topic_or_symbol": str(output), "status": "failed", "rows": 0, "error": str(exc)})

    if not args.skip_yahoo:
        for symbol in symbols:
            try:
                frame = fetch_yahoo_finance_rss(symbol)
                frames.append(frame)
                status_rows.append({"provider": "yahoo_finance_rss", "topic_or_symbol": symbol, "status": "ok", "rows": len(frame), "error": ""})
                print(f"[news] yahoo symbol={symbol} rows={len(frame)}", flush=True)
            except Exception as exc:
                status_rows.append({"provider": "yahoo_finance_rss", "topic_or_symbol": symbol, "status": "failed", "rows": 0, "error": str(exc)})
                print(f"[news] yahoo symbol={symbol} failed={exc}", flush=True)

    if not args.skip_google_news:
        for topic in _split(args.topics):
            if topic not in NEWS_TOPIC_QUERIES:
                continue
            try:
                frame = fetch_google_news_rss(topic, GOOGLE_NEWS_TOPIC_QUERIES[topic])
                frame = _expand_topic_symbols(frame, topic, selected_symbols)
                frames.append(frame)
                status_rows.append({"provider": "google_news_rss", "topic_or_symbol": topic, "status": "ok", "rows": len(frame), "error": ""})
                print(f"[news] google topic={topic} rows={len(frame)}", flush=True)
            except Exception as exc:
                status_rows.append({"provider": "google_news_rss", "topic_or_symbol": topic, "status": "failed", "rows": 0, "error": str(exc)})
                print(f"[news] google topic={topic} failed={exc}", flush=True)

    if not args.skip_gdelt:
        windows = _date_windows(args)
        topics = [topic for topic in _split(args.topics) if topic in NEWS_TOPIC_QUERIES]
        total = max(1, len(windows) * len(topics))
        done = 0
        for start, end in windows:
            for topic in topics:
                done += 1
                query = NEWS_TOPIC_QUERIES[topic]
                try:
                    frame = fetch_gdelt_articles(
                        topic,
                        query,
                        start_datetime=_gdelt_datetime(start),
                        end_datetime=_gdelt_datetime(end),
                        maxrecords=args.gdelt_maxrecords,
                        sleep_seconds=args.gdelt_sleep,
                        retries=args.gdelt_retries,
                    )
                    frame = _expand_topic_symbols(frame, topic, selected_symbols)
                    frames.append(frame)
                    status_rows.append(
                        {
                            "provider": "gdelt_doc_api",
                            "topic_or_symbol": topic,
                            "window_start": start.isoformat(),
                            "window_end": end.isoformat(),
                            "status": "ok",
                            "rows": len(frame),
                            "error": "",
                        }
                    )
                    print(f"[news] gdelt {done}/{total} topic={topic} rows={len(frame)} window={start.date()}..{end.date()}", flush=True)
                except Exception as exc:
                    status_rows.append(
                        {
                            "provider": "gdelt_doc_api",
                            "topic_or_symbol": topic,
                            "window_start": start.isoformat(),
                            "window_end": end.isoformat(),
                            "status": "failed",
                            "rows": 0,
                            "error": str(exc),
                        }
                    )
                    print(f"[news] gdelt {done}/{total} topic={topic} failed={exc}", flush=True)

    normalized = normalize_public_news(frames)
    source_counts = normalized["source"].astype(str).str.split(":").str[0].value_counts().head(8).to_dict() if not normalized.empty else {}
    print(f"[news] normalized rows={len(normalized)} source_counts={source_counts}", flush=True)
    return normalized, pd.DataFrame(status_rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    ensure_data_lake(Path(args.data_root))
    started = time.monotonic()
    news, status = fetch_news(args)
    output = Path(args.output)
    status_output = Path(args.status_output)
    write_table(news, output)
    write_table(status, status_output)
    upsert_inventory_entries(
        [
            entry_from_file(output, dataset_name="raw_news_public_market_news", source="news", source_url_or_provider="yahoo_finance_rss,gdelt_doc_api"),
            entry_from_file(status_output, dataset_name="interim_events_public_news_fetch_status", source="events", source_url_or_provider="yahoo_finance_rss,gdelt_doc_api"),
        ]
    )
    elapsed = int(time.monotonic() - started)
    print(f"[news] wrote rows={len(news)} output={output} status={status_output} elapsed={elapsed}s")


if __name__ == "__main__":
    main()
