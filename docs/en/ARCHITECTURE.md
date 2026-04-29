# Architecture

The project separates `backend`, `market_ai`, `frontend`, and `scripts` by responsibility. The goal is not an oil dashboard, but a Universal Market Forecasting Dashboard that can expand across market assets.

## Responsibilities

- `backend/`: FastAPI routes, HTTP error handling, static frontend serving, and service adapters only.
- `market_ai/`: data ingestion, feature engineering, forecasting, modeling, calibration, regimes, backtesting, and LLM context logic.
- `frontend/`: TradingView overlay UI, controls, panels, and API clients.
- `scripts/`: human-run CLI entrypoints only.
- `artifacts/`: `.npz` model artifacts and metadata JSON, separated from source code.
- `outputs/`: generated forecasts, backtests, plots, and reports.

## Dependency Direction

`backend` may call `market_ai`. `market_ai` must not import `backend`. This keeps models, backtests, and data pipelines testable without the API server.

## Compatibility Layer

The new uvicorn entrypoint is `backend.app.main:app`. The old `app.main:app` remains only as a thin wrapper for legacy run commands.
