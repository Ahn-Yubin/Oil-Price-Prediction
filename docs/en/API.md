# API

The API preserves compatibility with the existing chart frontend while exposing the newer forecast contract.

## Stable Endpoints

- `GET /api/health`
- `GET /api/models`
- `GET /api/data-status?symbol=CL=F&interval=1d`
- `GET /api/forecast?symbol=CL=F&interval=1d`
- `GET /api/chart?symbol=CL=F&interval=1d`
- `GET /api/features`
- `GET /api/explanation`
- `GET /api/backtests`

## `/api/chart`

`/api/chart` is backward-compatible. It preserves legacy payload keys: `candles`, `predicted`, `predicted_lower`, `predicted_upper`, `forecast_models`, `metrics`, and `updated_at`. Data quality information is added only through additive fields.

## `/api/forecast`

`/api/forecast` is the newer typed forecast contract. It returns candles, quantile forecasts, scenarios, regime probabilities, model metadata, and data status.

## Error Policy

Production must not silently use mock data when market data cannot be loaded. The API should expose degraded states through `data_status` and explicit errors.
