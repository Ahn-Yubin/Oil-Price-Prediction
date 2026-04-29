#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app
from market_ai.data.providers import yfinance_provider

EXPECTED_503_HINTS = (
    "artifact",
    "model",
    "market data",
    "external data",
    "unavailable",
    "not found",
    "download",
)


def _smoke_ohlc(_provider_symbol, timeframe) -> pd.DataFrame:
    rows = 320
    step = timedelta(seconds=timeframe.seconds)
    end = datetime.now(timezone.utc)
    dates = [end - step * (rows - idx - 1) for idx in range(rows)]
    base = np.arange(rows, dtype=float)
    close = 76.0 + 0.03 * base + 1.5 * np.sin(base / 12.0)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    span = np.maximum(np.abs(close - open_) * 0.4, 0.35)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) + span,
            "low": np.minimum(open_, close) - span,
            "close": close,
            "volume": np.zeros(rows),
        }
    )


def main() -> int:
    yfinance_provider._download_ohlc = _smoke_ohlc
    client = TestClient(app)
    endpoints = [
        "/api/health",
        "/api/models",
        "/api/data-status?symbol=CL=F&interval=1d",
        "/api/forecast?symbol=CL=F&interval=1d",
        "/api/chart?symbol=CL=F&interval=1d",
    ]
    failed = []
    expected_503 = []
    for endpoint in endpoints:
        response = client.get(endpoint)
        body = response.text[:400]
        if response.status_code == 503 and any(
            hint in body.lower() for hint in EXPECTED_503_HINTS
        ):
            expected_503.append((endpoint, body))
            print(f"{endpoint}: 503 expected dependency unavailable")
            continue
        print(f"{endpoint}: {response.status_code}")
        if response.status_code >= 400:
            failed.append((endpoint, response.status_code, response.text[:400]))
    if expected_503:
        print("Expected 503 responses:")
        for endpoint, body in expected_503:
            print(f"- {endpoint}: {body}")
    if failed:
        for endpoint, status, body in failed:
            print(f"FAILED {endpoint}: {status} {body}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
