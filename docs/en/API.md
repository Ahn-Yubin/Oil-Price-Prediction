# API

The API preserves compatibility with the existing chart frontend while exposing new forecast/context contracts. New fields must be additive.

## Endpoints

| Endpoint | Role |
| --- | --- |
| `GET /api/health` | app settings, artifacts, provider status |
| `GET /api/models` | model registry, artifact availability, metadata |
| `GET /api/data-status?symbol=CL=F&interval=1d` | data source, resolved symbol, staleness |
| `GET /api/forecast?symbol=CL=F&interval=1d` | candles, quantile forecast, scenarios, regime, model metadata |
| `GET /api/chart?symbol=CL=F&interval=1d` | legacy chart payload; backward-compatibility target |
| `GET /api/market-context?symbol=NYMEX:CL1%21&interval=1d` | news, context markers, scenario commentary |
| `GET /api/features` | feature information |
| `GET /api/explanation` | forecast and optional LLM context explanation |
| `GET /api/backtests` | backtest output lookup |

## `/api/chart`

`/api/chart` must not be removed or changed in a breaking way. Existing keys are preserved:

- `candles`
- `predicted`
- `predicted_lower`
- `predicted_upper`
- `forecast_models`
- `metrics`
- `updated_at`

Data quality, warnings, and model metadata may only be added as additive fields.

## `/api/forecast`

The new typed forecast contract includes:

- `candles`: historical OHLCV
- `forecast`: horizon-level quantile paths
- `scenarios`: bull/base/bear scenarios
- `model_metadata`: model id, artifact status, training metadata
- `data_status`: real/stale/fallback/mock/error
- `warnings`, `warning_objects`: degraded status and actions

Forecast prices are reconstructed from volatility-scaled cumulative log returns.

## `/api/market-context`

This endpoint supports historical news and context interpretation on the chart.

Returned content:

- `news`: recent headline/source/time/url rows
- `context_points`: event count, bias, impact, uncertainty, and explanation by date
- `scenario_commentary`: deterministic commentary for model forecast scenarios
- `llm_context_summary`: summary of LLM/context state
- `calibration_status`: whether bands are calibrated

The LLM does not forecast numeric prices. Commentary is auxiliary human-readable explanation of forecast and context.

Example:

```bash
curl "http://127.0.0.1:8000/api/market-context?symbol=NYMEX:CL1%21&interval=1d&models=llm_context_seq_moe"
```

## Error Policy

Production must not silently use mock data when market data fails. APIs must surface degraded state through `data_status`, warnings, and explicit errors.
