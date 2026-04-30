#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_ai.data.manifests import entry_from_file, upsert_inventory_entries
from market_ai.data.storage import DATA_ROOT, ensure_data_lake, write_table
from market_ai.llm.context_builder import build_event_context_daily


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build point-in-time event/LLM context features.")
    parser.add_argument("--events-path", default=os.environ.get("MARKET_EVENTS_PATH", ""))
    parser.add_argument("--news-path", default=os.environ.get("NEWS_EVENTS_PATH", ""))
    parser.add_argument("--symbols", default="CL=F,BZ=F,NG=F")
    parser.add_argument("--mode", choices=["none", "local_rules", "openai_compatible", "local_http", "offline_file"], default="local_rules")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--live", action="store_true", help="Allow actual external/local LLM calls.")
    parser.add_argument("--offline-file", default="")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    return parser.parse_args()


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    ensure_data_lake(data_root)
    context_path = data_root / "processed" / "event_context" / "event_context_daily.csv"
    cache_path = data_root / "processed" / "event_context" / "llm_context_cache.jsonl"
    context_frame, raw_events = build_event_context_daily(
        symbols=_split(args.symbols),
        events_paths=_split(args.events_path),
        news_paths=_split(args.news_path),
        mode=args.mode,
        start=args.start or None,
        end=args.end or None,
        api_key=os.environ.get("LLM_API_KEY"),
        api_base=os.environ.get("LLM_API_BASE") if args.mode == "openai_compatible" else os.environ.get("LOCAL_LLM_API_BASE"),
        model=os.environ.get("LLM_MODEL") if args.mode == "openai_compatible" else os.environ.get("LOCAL_LLM_MODEL"),
        live=args.live and os.environ.get("ENABLE_EXTERNAL_LLM_CALLS", "").lower() in {"1", "true", "yes", "on"},
        offline_file=args.offline_file or None,
        cache_path=cache_path,
    )
    raw_path = data_root / "interim" / "events" / "combined_market_events.csv"
    write_table(raw_events, raw_path)
    write_table(context_frame, context_path)
    upsert_inventory_entries(
        [
            entry_from_file(raw_path, dataset_name="interim_combined_market_events", source="events", point_in_time_safe=True),
            entry_from_file(context_path, dataset_name="processed_event_context_daily", source="events", point_in_time_safe=True, notes=f"llm_mode={args.mode}; live={args.live}"),
        ]
    )
    print(f"Wrote event context rows={len(context_frame)} to {context_path}")
    print(f"Wrote LLM context cache to {cache_path}")


if __name__ == "__main__":
    main()
