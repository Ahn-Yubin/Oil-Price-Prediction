# API

API는 기존 chart frontend와의 호환성을 유지하면서 신규 forecast/context contract를 제공합니다. 새 field는 additive로만 추가합니다.

## Endpoint

| Endpoint | 역할 |
| --- | --- |
| `GET /api/health` | app 설정, artifact, provider 상태 |
| `GET /api/models` | model registry, artifact availability, metadata |
| `GET /api/data-status?symbol=CL=F&interval=1d` | 데이터 source, resolved symbol, stale 여부 |
| `GET /api/forecast?symbol=CL=F&interval=1d` | candle, quantile forecast, scenario, regime, model metadata |
| `GET /api/chart?symbol=CL=F&interval=1d&horizon=7` | 기존 chart payload. backward compatibility 대상 |
| `GET /api/market-context?symbol=CL=F&interval=1d` | 뉴스, context marker, scenario commentary |
| `GET /api/features` | feature 관련 정보 |
| `GET /api/explanation` | forecast와 optional LLM context 기반 설명 |
| `GET /api/backtests` | backtest output 조회 |
| `GET /api/backtests/visualization` | 과거 origin 기준 chart backtest overlay payload |
| `GET /api/model-commentary` | 단일 운영 모델 예측 경로에 대한 LLM/fallback 해설 |

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

지원 query:

- `symbol`
- `interval`
- `horizon`: optional. 지정하지 않으면 interval 기본 horizon인 30스텝을 씁니다. 프론트엔드는 7, 14, 30 중 하나를 전달합니다. 내부 모델은 h30 artifact를 실행하고 요청한 길이만큼 앞부분을 잘라 응답합니다.
- `models`: optional model selector

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
curl "http://127.0.0.1:8000/api/market-context?symbol=CL=F&interval=1d&models=oil_context_fusion"
```

## `/api/backtests/visualization`

차트에서 선택한 과거 시점을 기준으로 point-in-time forecast를 만들고, 같은 payload 안에 실제 이후 캔들을 함께 반환합니다. `/api/chart` contract는 변경하지 않고, 이 endpoint에만 backtest용 additive field를 둡니다.

주요 query:

- `symbol`: 예측 대상 symbol
- `interval`: `1d`, `1h`
- `origin_time`: unix timestamp 또는 ISO datetime. 이 시점 이하의 마지막 candle이 예측 기준점입니다.
- `models`: optional model selector
- `horizon`: optional 표시 horizon

추가 반환 field:

- `mode`: `backtest_visualization`
- `origin_time`: 실제 사용된 candle time
- `actual_future_candles`: origin 이후 실제 OHLCV. 프론트에서는 반투명 candle series로 표시합니다.
- `backtest`: origin index, history rows, future rows, horizon metadata

예시:

```bash
curl "http://127.0.0.1:8000/api/backtests/visualization?symbol=CL=F&interval=1d&origin_time=2026-04-01T00:00:00Z&models=oil_context_fusion"
```

## `/api/model-commentary`

단일 운영 모델 `oil_context_fusion`이 이미 낸 forecast path를 바탕으로 LLM이 애널리스트식 시황 해설을 작성합니다. LLM은 새 가격 target이나 새 return path를 만들지 않고, 뉴스, 차트 흐름, 국면, 수급/거시 리스크를 근거로 “왜 이런 방향으로 보는지”만 설명합니다. 외부 LLM 호출이 꺼져 있거나 실패하면 같은 입력을 사용한 deterministic fallback을 반환합니다.

주요 query:

- `symbol`, `interval`, `models`, `horizon`
- `origin_time`: optional. backtest visualization과 같은 과거 origin 기준 해설에 사용합니다.

예시:

```bash
curl "http://127.0.0.1:8000/api/model-commentary?symbol=CL=F&interval=1d&models=oil_context_fusion"
```

## 오류 정책

Production에서 market data를 가져오지 못하면 mock data를 조용히 사용하지 않습니다. API는 `data_status`, warning, 명시적 error로 degraded 상태를 드러내야 합니다.
