from __future__ import annotations

from typing import Any
import json

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
    latest_leaderboard = output_dir / "leaderboards" / "latest.json"
    if latest_leaderboard.exists():
        meta = json.loads(latest_leaderboard.read_text(encoding="utf-8"))
        latest_dir = PROJECT_DIR / meta.get("output_dir", "")
        leaderboard_path = latest_dir / "leaderboard.csv"
        availability_path = latest_dir / "model_availability.csv"
        if leaderboard_path.exists():
            frame = pd.read_csv(leaderboard_path)
            if "symbol" in frame.columns:
                frame = frame[frame["symbol"].astype(str).str.replace("=", "_") == requested_symbol]
            if "interval" in frame.columns:
                frame = frame[frame["interval"].astype(str) == requested_interval]
            availability = pd.read_csv(availability_path).to_dict(orient="records") if availability_path.exists() else []
            return {
                "status": "available",
                "path": str(leaderboard_path.relative_to(PROJECT_DIR)),
                "rows": len(frame),
                "leaderboard": frame.head(25).to_dict(orient="records"),
                "model_availability": availability,
                "latest_run": meta,
            }
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
