# API

The API preserves compatibility with the existing chart frontend while exposing new forecast/context contracts. New fields must be additive.

## Endpoints

| Endpoint | Role |
| --- | --- |
| `GET /api/health` | app settings, artifacts, provider status |
| `GET /api/models` | model registry, artifact availability, metadata |
| `GET /api/data-status?symbol=CL=F&interval=1d` | data source, resolved symbol, staleness |
| `GET /api/forecast?symbol=CL=F&interval=1d` | candles, quantile forecast, scenarios, regime, model metadata |
| `GET /api/chart?symbol=CL=F&interval=1d&horizon=7` | legacy chart payload; backward-compatibility target |
| `GET /api/market-context?symbol=CL=F&interval=1d` | news, context markers, scenario commentary |
| `GET /api/dashboard-analysis?symbol=CL=F&interval=1d` | generates AI commentary, news interpretation, and forecast report with one external LLM call |
| `GET /api/features` | feature information |
| `GET /api/explanation` | forecast and optional LLM context explanation |
| `GET /api/backtests` | backtest output lookup |
| `GET /api/backtests/visualization` | chart backtest overlay payload from a historical origin |
| `GET /api/model-commentary` | compatibility endpoint for standalone market commentary |

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

Supported query parameters:

- `symbol`
- `interval`
- `horizon`: optional. If omitted, the interval default horizon of 30 steps is used. The current dashboard uses a fixed 30-day view and displays 1-week, 2-week, and 1-month segment markers. The backend runs the h30 artifact and returns a leading slice when a shorter horizon is requested.
- `models`: optional model selector

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
curl "http://127.0.0.1:8000/api/market-context?symbol=CL=F&interval=1d&models=oil_context_fusion"
```

## `/api/dashboard-analysis`

This is the combined endpoint for the dashboard's three AI panels. Instead of having the frontend call `/api/model-commentary`, `/api/market-context`, and `/api/report` separately, the server bundles the same forecast and news evidence, makes one external LLM request, and returns panel-specific payloads.

Returned content:

- `commentary`: payload for the AI market commentary panel
- `market_context`: payload for the news interpretation panel, including translated headlines and public context explanations
- `report`: payload for the forecast report panel
- `warnings`: warnings from combined generation
- `llm_used`: whether an external LLM was used

Main query parameters:

- `symbol`, `interval`, `models`, `horizon`, `language`
- `origin_time`: optional. When present, output is written as a point-in-time backtest report from that origin and avoids live relative wording such as current, recent, now, or today.

This endpoint still does not use the LLM as a numeric forecaster. Numeric values and paths come only from the forecast payload; the LLM writes prose and interprets news. The frontend uses request ids and payload keys to ignore stale responses and clears panel/marker state before language switches or backtest-origin changes so old news markers cannot leak onto the chart.

Example:

```bash
curl "http://127.0.0.1:8000/api/dashboard-analysis?symbol=CL=F&interval=1d&models=oil_context_fusion&horizon=30&language=en"
```

## `/api/backtests/visualization`

This endpoint builds a point-in-time forecast from the historical origin selected on the chart and returns the realized candles that followed that origin in the same payload. The `/api/chart` contract is unchanged; backtest-specific fields are additive and isolated to this endpoint.

Main query parameters:

- `symbol`: forecast target symbol
- `interval`: `1d` or `1h`
- `origin_time`: unix timestamp or ISO datetime. The last candle at or before this time becomes the forecast origin.
- `models`: optional model selector
- `horizon`: optional display horizon

Additional response fields:

- `mode`: `backtest_visualization`
- `origin_time`: actual candle time used as the origin
- `actual_future_candles`: realized OHLCV after the origin. The frontend renders these as a translucent candle series.
- `backtest`: origin index, history rows, future rows, and horizon metadata

Example:

```bash
curl "http://127.0.0.1:8000/api/backtests/visualization?symbol=CL=F&interval=1d&origin_time=2026-04-01T00:00:00Z&models=oil_context_fusion"
```

## `/api/model-commentary`

This compatibility endpoint turns the already-produced `oil_context_fusion` forecast path into analyst-style market commentary. The new dashboard prefers `/api/dashboard-analysis` to reduce external LLM calls. The LLM must not create new price targets or return paths; it explains why the path leans that way using news, chart action, regime state, supply/macro context, and risks. If external LLM calls are disabled or fail, the endpoint returns deterministic fallback commentary from the same inputs.

Main query parameters:

- `symbol`, `interval`, `models`, `horizon`
- `origin_time`: optional, used for commentary from the same historical origin as backtest visualization

Example:

```bash
curl "http://127.0.0.1:8000/api/model-commentary?symbol=CL=F&interval=1d&models=oil_context_fusion"
```

## Error Policy

Production must not silently use mock data when market data fails. APIs must surface degraded state through `data_status`, warnings, and explicit errors.
