from __future__ import annotations

from typing import Protocol

import pandas as pd

from market_ai.schemas.market import Timeframe


class MarketDataProvider(Protocol):
    def download_ohlc(self, provider_symbol: str, timeframe: Timeframe) -> pd.DataFrame:
        ...
