# Architecture

The project separates `backend`, `market_ai`, `frontend`, and `scripts` to operate a WTI oil forecasting-only dashboard and single model.

## Responsibility Split

| Area | Responsibility |
| --- | --- |
| `backend/` | FastAPI routes, HTTP error handling, static frontend serving, service adapters |
| `market_ai/` | data ingestion/normalization, feature engineering, forecasting, modeling, calibration, regime, backtesting, LLM context |
| `frontend/` | chart overlay, controls, data quality panel, context marker/news panel |
| `scripts/` | human-run CLI entrypoints |
| `artifacts/` | model artifacts and metadata |
| `data/` | raw/interim/processed/external/features/manifests |
| `outputs/` | backtests, plots, generated output; stale Markdown reports are folded into canonical docs and removed |
| `docs/ko`, `docs/en` | Korean primary docs and English mirror docs |

## Dependency Direction

`backend` may call `market_ai`. `market_ai` must not import `backend`. This keeps data pipelines, models, and backtests testable without the API server.

```text
frontend -> backend API -> market_ai
scripts ---------------> market_ai
tests -----------------> backend / market_ai
```

## Runtime Flow

```text
provider raw data
-> data/raw, data/interim
-> data/processed market/fundamental/event context
-> training scripts
-> artifacts/models + artifacts/metadata
-> backend forecast/context APIs
-> frontend chart overlay
```

## LLM Flow

The LLM is used as a context encoder inside `market_ai.llm` and as a prose generator in the backend dashboard analysis route over already-computed forecast/news evidence.

```text
news/events
-> LocalEventContextEncoder or OpenAICompatibleLLMEventEncoder
-> event_context_daily.csv
-> oil_context_fusion x_event_context
```

The LLM does not generate numeric forecast paths.

## API Layer

- `/api/forecast`: new typed forecast contract
- `/api/chart`: legacy chart compatibility contract
- `/api/market-context`: news/context markers and scenario commentary
- `/api/dashboard-analysis`: one external LLM call for AI commentary, news interpretation, and forecast report prose
- `/api/backtests/visualization`: forecast overlay and realized future candles from a historical origin
- `/api/explanation`: forecast explanation
- `/api/models`, `/api/data-status`, `/api/backtests`: operational/diagnostic endpoints

`/api/chart` compatibility is preserved until explicitly removed.

## Compatibility Layer

The new uvicorn entrypoint is `backend.app.main:app`. The old `app.main:app` remains only as a thin wrapper for legacy commands.

## Operational Boundaries

- Do not silently use mock/synthetic fallback in production.
- Inject API keys only through environment variables or `.env`; do not commit them.
- Keep `.pt`/`.npz` artifacts separate from source code.
- Update Korean and English mirror documentation together.
