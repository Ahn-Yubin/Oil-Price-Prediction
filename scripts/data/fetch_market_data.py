#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_ai.data.providers.yfinance_provider import load_market_data_window


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a market data window and print data status.")
    parser.add_argument("--symbol", default="CL=F")
    parser.add_argument("--interval", default="1d")
    args = parser.parse_args()
    window = load_market_data_window(args.symbol, args.interval)
    print(json.dumps(window.data_status.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
