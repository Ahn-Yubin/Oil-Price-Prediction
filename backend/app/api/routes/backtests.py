from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Query

from market_ai.config import PROJECT_DIR, get_settings


router = APIRouter()


@router.get("/api/backtests")
def backtests(symbol: str = Query(default=None), interval: str = Query(default=None)) -> dict[str, Any]:
    current_settings = get_settings()
    requested_symbol = (symbol or current_settings.default_symbol).replace("=", "_")
    requested_interval = interval or current_settings.default_interval
    output_dir = PROJECT_DIR / "outputs" / "backtests"
    candidates = [
        output_dir / f"{requested_symbol}_{requested_interval}_leaderboard.csv",
        output_dir / f"{requested_symbol}_leaderboard.csv",
        output_dir / f"{requested_symbol}_{requested_interval}_summary.csv",
        output_dir / f"{requested_symbol}_summary.csv",
    ]
    availability_candidates = [
        output_dir / f"{requested_symbol}_{requested_interval}_model_availability.csv",
        output_dir / "latest_model_availability.csv",
    ]

    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            availability_path = next((candidate for candidate in availability_candidates if candidate.exists()), None)
            availability = pd.read_csv(availability_path).to_dict(orient="records") if availability_path else []
            return {
                "status": "available",
                "path": str(path.relative_to(PROJECT_DIR)),
                "rows": len(frame),
                "leaderboard": frame.head(25).to_dict(orient="records"),
                "model_availability": availability,
            }
    availability_path = next((candidate for candidate in availability_candidates if candidate.exists()), None)
    availability = pd.read_csv(availability_path).to_dict(orient="records") if availability_path else []
    return {"status": "missing", "path": None, "rows": 0, "leaderboard": [], "model_availability": availability}
