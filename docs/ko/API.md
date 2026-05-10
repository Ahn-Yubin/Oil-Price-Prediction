# API

API는 기존 chart frontend와의 호환성을 유지하면서 신규 forecast/context contract를 제공합니다. 새 field는 additive로만 추가합니다.

## Endpoint

| Endpoint | 역할 |
| --- | --- |
| `GET /api/health` | app 설정, artifact, provider 상태 |
| `GET /api/models` | model registry, artifact availability, metadata |
| `GET /api/data-status?symbol=CL=F&interval=1d` | 데이터 source, resolved symbol, stale 여부 |
| `GET /api/forecast?symbol=CL=F&interval=1d` | candle, quantile forecast, scenario, regime, model metadata |
| `GET /api/chart?symbol=CL=F&interval=1d` | 기존 chart payload. backward compatibility 대상 |
| `GET /api/market-context?symbol=NYMEX:CL1%21&interval=1d` | 뉴스, context marker, scenario commentary |
| `GET /api/features` | feature 관련 정보 |
| `GET /api/explanation` | forecast와 optional LLM context 기반 설명 |
| `GET /api/backtests` | backtest output 조회 |

## `/api/chart`

`/api/chart`는 제거하거나 breaking change를 만들면 안 됩니다. 기존 key를 유지합니다.

- `candles`
- `predicted`
- `predicted_lower`
- `predicted_upper`
- `forecast_models`
- `metrics`
- `updated_at`

데이터 품질, warning, model metadata는 additive field로만 추가합니다.

## `/api/forecast`

신규 typed forecast contract입니다. 주요 payload는 다음을 포함합니다.

- `candles`: historical OHLCV
- `forecast`: horizon별 quantile path
- `scenarios`: bull/base/bear scenario
- `model_metadata`: model id, artifact status, training metadata
- `data_status`: real/stale/fallback/mock/error
- `warnings`, `warning_objects`: degraded 상태와 조치

Forecast price는 volatility-scaled cumulative log return에서 복원된 값입니다.

## `/api/market-context`

차트 위에 과거 뉴스와 context 해석을 표시하기 위한 endpoint입니다.

반환 내용:

- `news`: 최근 뉴스 headline/source/time/url
- `context_points`: 날짜별 event count, bias, impact, uncertainty, explanation
- `scenario_commentary`: 모델 예측 시나리오에 대한 deterministic 해설
- `llm_context_summary`: LLM/context 상태 요약
- `calibration_status`: band가 calibrated인지 여부

LLM이 숫자 가격을 예측하지 않습니다. 이 endpoint의 해설은 forecast와 context를 사람이 읽기 쉽게 설명하는 보조 정보입니다.

예시:

```bash
curl "http://127.0.0.1:8000/api/market-context?symbol=NYMEX:CL1%21&interval=1d&models=llm_context_seq_moe"
```

## 오류 정책

Production에서 market data를 가져오지 못하면 mock data를 조용히 사용하지 않습니다. API는 `data_status`, warning, 명시적 error로 degraded 상태를 드러내야 합니다.
