from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_ai.config import PROJECT_DIR
from market_ai.data.storage import safe_symbol
from market_ai.schemas.market import ForecastPoint


CALIBRATION_DIR = PROJECT_DIR / "artifacts" / "calibration"


@dataclass(frozen=True)
class CalibrationArtifact:
    model: str
    symbol: str
    interval: str
    calibration_status: str
    adjustment_80: list[float]
    adjustment_90: list[float]
    adjustment_95: list[float]
    coverage_80: float | None
    coverage_90: float | None
    coverage_95: float | None
    calibration_start: str | None
    calibration_end: str | None
    n_origins: int
    generated_at: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def calibration_artifact_path(model: str, symbol: str, interval: str, root: Path = CALIBRATION_DIR) -> Path:
    return root / f"{model}_{safe_symbol(symbol)}_{interval}.json"


def _quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    return float(np.quantile(values, min(max(q, 0.0), 1.0), method="higher"))


def compute_conformal_adjustment(details: pd.DataFrame, *, model: str, symbol: str, interval: str) -> CalibrationArtifact:
    frame = details.copy()
    if "model" in frame.columns:
        frame = frame[frame["model"].astype(str) == model]
    if frame.empty:
        return CalibrationArtifact(
            model=model,
            symbol=symbol,
            interval=interval,
            calibration_status="uncalibrated",
            adjustment_80=[],
            adjustment_90=[],
            adjustment_95=[],
            coverage_80=None,
            coverage_90=None,
            coverage_95=None,
            calibration_start=None,
            calibration_end=None,
            n_origins=0,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
    max_step = int(pd.to_numeric(frame["step"], errors="coerce").max())
    adj80: list[float] = []
    adj90: list[float] = []
    adj95: list[float] = []
    coverage80 = []
    coverage90 = []
    for step in range(1, max_step + 1):
        step_frame = frame[pd.to_numeric(frame["step"], errors="coerce") == step]
        actual = pd.to_numeric(step_frame["actual_log_return"], errors="coerce").to_numpy(dtype=np.float64)
        p10 = pd.to_numeric(step_frame.get("p10_log_return"), errors="coerce").to_numpy(dtype=np.float64)
        p90 = pd.to_numeric(step_frame.get("p90_log_return"), errors="coerce").to_numpy(dtype=np.float64)
        p05 = pd.to_numeric(step_frame.get("p05_log_return"), errors="coerce").to_numpy(dtype=np.float64)
        p95 = pd.to_numeric(step_frame.get("p95_log_return"), errors="coerce").to_numpy(dtype=np.float64)
        miss80 = np.maximum(p10 - actual, actual - p90)
        miss90 = np.maximum(p05 - actual, actual - p95)
        adj80.append(max(0.0, _quantile(miss80, 0.80)))
        adj90.append(max(0.0, _quantile(miss90, 0.90)))
        adj95.append(max(0.0, _quantile(miss90, 0.95)))
        if len(actual):
            coverage80.append(float(np.mean((actual >= p10) & (actual <= p90))))
            coverage90.append(float(np.mean((actual >= p05) & (actual <= p95))))
    origins = pd.to_numeric(frame["origin"], errors="coerce").dropna().astype(int)
    return CalibrationArtifact(
        model=model,
        symbol=symbol,
        interval=interval,
        calibration_status="calibrated" if len(origins.unique()) >= 20 else "uncalibrated",
        adjustment_80=adj80,
        adjustment_90=adj90,
        adjustment_95=adj95,
        coverage_80=float(np.mean(coverage80)) if coverage80 else None,
        coverage_90=float(np.mean(coverage90)) if coverage90 else None,
        coverage_95=None,
        calibration_start=str(int(origins.min())) if not origins.empty else None,
        calibration_end=str(int(origins.max())) if not origins.empty else None,
        n_origins=int(origins.nunique()),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def save_calibration_artifact(artifact: CalibrationArtifact, path: Path | None = None) -> Path:
    resolved = path or calibration_artifact_path(artifact.model, artifact.symbol, artifact.interval)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(artifact.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved


def load_calibration_artifact(model: str, symbol: str, interval: str, root: Path = CALIBRATION_DIR) -> CalibrationArtifact | None:
    path = calibration_artifact_path(model, symbol, interval, root)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationArtifact(**data)


def _adjustment(values: list[float], idx: int) -> float:
    if not values:
        return 0.0
    return float(values[min(idx, len(values) - 1)])


def apply_calibration_to_points(points: list[ForecastPoint], *, current_price: float, artifact: CalibrationArtifact | None) -> list[ForecastPoint]:
    if artifact is None or artifact.calibration_status != "calibrated":
        return points
    calibrated: list[ForecastPoint] = []
    for idx, point in enumerate(points):
        mid = np.log(max(point.p50, 1e-8) / max(current_price, 1e-8))
        p10 = min(np.log(max(point.p10, 1e-8) / current_price), mid)
        p90 = max(np.log(max(point.p90, 1e-8) / current_price), mid)
        p05 = min(np.log(max(point.p05, 1e-8) / current_price), p10)
        p95 = max(np.log(max(point.p95, 1e-8) / current_price), p90)
        adj80 = _adjustment(artifact.adjustment_80, idx)
        adj90 = _adjustment(artifact.adjustment_90, idx)
        calibrated.append(
            point.model_copy(
                update={
                    "p05": float(current_price * np.exp(p05 - adj90)),
                    "p10": float(current_price * np.exp(p10 - adj80)),
                    "p90": float(current_price * np.exp(p90 + adj80)),
                    "p95": float(current_price * np.exp(p95 + adj90)),
                    "confidence": min(max(point.confidence + 0.05, 0.0), 1.0),
                }
            )
        )
    return calibrated
