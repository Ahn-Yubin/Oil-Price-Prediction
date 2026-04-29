from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ohlc_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OHLC CSV missing columns: {sorted(missing)}")
    return frame
