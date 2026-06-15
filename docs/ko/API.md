# API

API는 기존 chart frontend와의 호환성을 유지하면서 신규 forecast/context contract를 제공합니다. 새 field는 additive로만 추가합니다.

## Endpoint

| Endpoint | 역할 |
| --- | --- |
| `GET /api/health` | app 설정, artifact, provider 상태 |
| `GET /api/models` | model registry, artifact availability, metadata |
| `GET /api/data-status?symbol=CL=F&interval=1d` | 데이터 source, resolved symbol, stale 여부 |
| `GET /api/forecast?symbol=CL=F&interval=1d` | candle, quantile forecast, scenario, regime, model metadata |
| `POST /api/scenarios/forecast` | 사용자 입력 미래 사건 묶음을 horizon별 LLM event context로 변환한 뒤 시나리오 forecast path 생성 |
| `GET /api/chart?symbol=CL=F&interval=1d&horizon=7` | 기존 chart payload. backward compatibility 대상 |
| `GET /api/market-context?symbol=CL=F&interval=1d` | 뉴스, context marker, scenario commentary |
| `GET /api/dashboard-analysis?symbol=CL=F&interval=1d` | AI 시황 해설, 뉴스 해석, 예측 리포트를 한 번의 외부 LLM 호출로 생성 |
| `GET /api/features` | feature 관련 정보 |
| `GET /api/explanation` | forecast와 optional LLM context 기반 설명 |
| `GET /api/backtests` | backtest output 조회 |
| `GET /api/backtests/visualization` | 과거 origin 기준 chart backtest overlay payload |
| `GET /api/model-commentary` | 개별 시황 해설 호환 endpoint |

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
- `horizon`: optional. 지정하지 않으면 interval 기본 horizon인 30스텝을 씁니다. 현재 dashboard는 30일 고정 화면을 쓰고 1주/2주/한달 구간 표시만 나눕니다. 내부 모델은 h30 artifact를 실행하고 요청 길이가 더 짧으면 앞부분을 잘라 응답합니다.
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

## `/api/scenarios/forecast`

Scenario mode에서 만든 시나리오 폴더의 미래 사건 묶음을 처리하는 additive endpoint입니다. Request body는 `title`, `content`, optional `event_time`, additive `events[]`, `symbol`, `interval`, `horizon`, `models`를 받습니다. `events[]`의 각 항목은 `title`, `content`, `event_time`을 가집니다.

처리 방식:

- 사용자 입력은 외부 LLM context encoder에 전달됩니다.
- LLM은 이벤트 종류, 방향성, 영향도, 불확실성, event embedding만 반환해야 합니다.
- Backend는 여러 이벤트를 예측 horizon별 event-context schedule로 바꿉니다. 특정 horizon에서는 그 시점까지 발생한 이벤트만 활성 context로 들어갑니다.
- `oil_context_fusion`은 각 horizon 구간의 context vector를 입력 feature로 사용해 숫자 forecast path를 계산합니다. 시나리오 path는 모델 출력에 사후 보정 배열을 더하지 않습니다.
- LLM이 직접 `p50`, `p90`, 목표가, 미래 return path를 만들면 validator가 해당 구조화 출력을 거부합니다.

`event_time`은 미래 사건의 발생 시각 메타데이터이자 horizon별 context 활성화 기준입니다. 모델 입력에는 미래 가격 정보가 들어가지 않으며, 사건 시각은 LLM 입력, `llm_context_summary.scenario_event_time`, `llm_context_summary.model_context_schedule`에 남습니다. `event_time`이 없으면 본문 안의 “모레/다음 주” 같은 표현을 LLM이 `generated_at` 기준으로 해석하지만, API는 warning을 함께 반환합니다.

반환 내용:

- `points`: 차트 overlay용 시나리오 path. 첫 점은 현재 가격 anchor입니다.
- `forecast`: quantile forecast path
- `llm_context_summary`: scenario override 출처, bias, impact, uncertainty, horizon별 `model_context_schedule`
- `llm_context`: 검증된 구조화 event context
- `warning_objects`: LLM fallback, event_time 누락, artifact/data 품질 경고

예시:

```bash
curl -X POST "http://127.0.0.1:8000/api/scenarios/forecast" \
  -H "Content-Type: application/json" \
  -d '{"title":"공급 충격 후 증산","content":"호르무즈 봉쇄 후 OPEC 증산","events":[{"title":"호르무즈 봉쇄","content":"원유 수송 차질로 공급 우려가 커짐","event_time":"2026-06-21T00:00:00Z"},{"title":"OPEC 증산","content":"OPEC이 원유 생산량을 늘림","event_time":"2026-07-01T00:00:00Z"}],"symbol":"CL=F","interval":"1d","horizon":30}'
```

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

## `/api/dashboard-analysis`

Dashboard의 세 AI 패널을 위한 통합 endpoint입니다. 기존처럼 `/api/model-commentary`, `/api/market-context`, `/api/report`를 화면에서 따로 호출하지 않고, 서버가 같은 forecast와 news evidence를 묶어 외부 LLM에 한 번 요청한 뒤 아래 payload를 나눠 반환합니다.

반환 내용:

- `commentary`: AI 시황 해설 패널 payload
- `market_context`: 뉴스 해석 패널 payload. LLM 번역 headline과 공개용 context 설명만 포함합니다.
- `report`: 예측 리포트 패널 payload
- `warnings`: 통합 생성 중 발생한 경고
- `llm_used`: 외부 LLM 사용 여부

주요 query:

- `symbol`, `interval`, `models`, `horizon`, `language`
- `origin_time`: optional. 백테스트 기준점이 있으면 해당 시점 기준의 point-in-time 해설로 작성하며, “현재/최근/지금” 같은 live 표현 대신 절대 날짜와 시간을 사용합니다.

이 endpoint의 LLM도 숫자 예측기를 대신하지 않습니다. 숫자와 예측 경로는 이미 계산된 forecast payload에서만 가져오며, LLM은 문장 생성과 뉴스 해석만 담당합니다. Frontend는 request id와 payload key로 stale response를 무시하고, 언어 전환/백테스트 origin 변경 중 오래된 뉴스 marker가 차트에 섞이지 않도록 패널과 marker를 먼저 비웁니다.

예시:

```bash
curl "http://127.0.0.1:8000/api/dashboard-analysis?symbol=CL=F&interval=1d&models=oil_context_fusion&horizon=30&language=ko"
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

단일 운영 모델 `oil_context_fusion`이 이미 낸 forecast path를 바탕으로 애널리스트식 시황 해설을 작성하는 개별 호환 endpoint입니다. 신규 dashboard 화면은 LLM 호출 수를 줄이기 위해 `/api/dashboard-analysis`를 우선 사용합니다. LLM은 새 가격 target이나 새 return path를 만들지 않고, 뉴스, 차트 흐름, 국면, 수급/거시 리스크를 근거로 “왜 이런 방향으로 보는지”만 설명합니다. 외부 LLM 호출이 꺼져 있거나 실패하면 같은 입력을 사용한 deterministic fallback을 반환합니다.

주요 query:

- `symbol`, `interval`, `models`, `horizon`
- `origin_time`: optional. backtest visualization과 같은 과거 origin 기준 해설에 사용합니다.

예시:

```bash
curl "http://127.0.0.1:8000/api/model-commentary?symbol=CL=F&interval=1d&models=oil_context_fusion"
```

## 오류 정책

Production에서 market data를 가져오지 못하면 mock data를 조용히 사용하지 않습니다. API는 `data_status`, warning, 명시적 error로 degraded 상태를 드러내야 합니다.
