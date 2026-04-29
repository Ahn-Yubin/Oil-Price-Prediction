from __future__ import annotations

import pandas as pd


def build_calendar_features(dates) -> pd.DataFrame:
    dt = pd.to_datetime(pd.Series(dates), errors="coerce", utc=True)
    out = pd.DataFrame(index=dt.index)
    out["day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)
    out["month"] = dt.dt.month.fillna(0).astype(int)
    out["quarter"] = dt.dt.quarter.fillna(0).astype(int)
    out["session"] = "regular"
    out["futures_expiry_placeholder"] = False
    out["economic_event_placeholder"] = False
    return out
