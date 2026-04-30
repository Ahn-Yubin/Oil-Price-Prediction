#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_ai.data.event_providers import FileEventProvider
from market_ai.llm.context_builder import encoder_for_mode
from market_ai.schemas.llm_context import MarketContextInput, RawNewsItem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LLM context encoder modes without numeric forecasting.")
    parser.add_argument("--mode", choices=["none", "local_rules", "openai_compatible", "local_http", "offline_file"], default="local_rules")
    parser.add_argument("--events-path", default=os.environ.get("MARKET_EVENTS_PATH", "data/external/events/sample_market_events.csv"))
    parser.add_argument("--offline-file", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    live = bool(args.live and not args.dry_run)
    provider = FileEventProvider([args.events_path] if args.events_path else [])
    api_base = os.environ.get("LLM_API_BASE") if args.mode == "openai_compatible" else os.environ.get("LOCAL_LLM_API_BASE")
    model = os.environ.get("LLM_MODEL") if args.mode == "openai_compatible" else os.environ.get("LOCAL_LLM_MODEL")
    encoder = encoder_for_mode(
        args.mode,
        provider=provider,
        api_key=os.environ.get("LLM_API_KEY"),
        api_base=api_base,
        model=model,
        live=live and os.environ.get("ENABLE_EXTERNAL_LLM_CALLS", "").lower() in {"1", "true", "yes", "on"},
        offline_file=args.offline_file or None,
    )
    context = MarketContextInput(
        symbol="CL=F",
        interval="1d",
        generated_at=datetime.now(timezone.utc),
        news=[RawNewsItem(title="Oil inventory draw raises supply risk", published_at=datetime.now(timezone.utc), source="dry_run")],
    )
    output = encoder.encode_events(context)
    dumped = output.model_dump(mode="json")
    text = json.dumps(dumped, ensure_ascii=False).lower()
    forbidden = any(token in text for token in ["target_price", "p50", "p90", "future_price_path", "return_path"])
    print(
        json.dumps(
            {
                "mode": args.mode,
                "live": live,
                "events": len(output.events),
                "overall_bias": output.overall_bias,
                "embedding_dim": len(output.event_embedding),
                "warnings": output.warnings,
                "safety_check_passed": not forbidden,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if forbidden:
        raise SystemExit("LLM context safety check failed.")


if __name__ == "__main__":
    main()
