# API

API는 기존 chart frontend와의 호환성을 유지하면서, 신규 forecast contract를 함께 제공합니다.

## 안정 Endpoint

- `GET /api/health`
- `GET /api/models`
- `GET /api/data-status?symbol=CL=F&interval=1d`
- `GET /api/forecast?symbol=CL=F&interval=1d`
- `GET /api/chart?symbol=CL=F&interval=1d`
- `GET /api/features`
- `GET /api/explanation`
- `GET /api/backtests`

## `/api/chart`

`/api/chart`는 backward compatibility 대상입니다. 기존 payload key인 `candles`, `predicted`, `predicted_lower`, `predicted_upper`, `forecast_models`, `metrics`, `updated_at`을 유지합니다. 데이터 품질 정보는 additive field로만 추가합니다.

## `/api/forecast`

`/api/forecast`는 신규 typed forecast contract입니다. Candle, quantile forecast, scenario, regime probability, model metadata, data status를 함께 제공합니다.

## 오류 정책

Production에서 market data를 가져오지 못하면 mock data를 조용히 사용하지 않습니다. API는 `data_status`와 명시적 error로 degraded 상태를 드러내야 합니다.
