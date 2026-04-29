#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from market_ai.features.price_features import build_price_features


def main() -> int:
    parser = argparse.ArgumentParser(description="Build price features from an OHLC CSV.")
    parser.add_argument("csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    features = build_price_features(frame)
    print(features.tail().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
