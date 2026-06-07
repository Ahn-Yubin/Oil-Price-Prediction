# 아키텍처

이 프로젝트는 `backend`, `market_ai`, `frontend`, `scripts`를 분리해 WTI 유가 예측 전용 dashboard와 단일 모델을 운영합니다.

## 책임 분리

| 영역 | 책임 |
| --- | --- |
| `backend/` | FastAPI route, HTTP error 처리, static frontend serving, service adapter |
| `market_ai/` | 데이터 수집/정규화, feature engineering, forecasting, modeling, calibration, regime, backtesting, LLM context |
| `frontend/` | chart overlay, controls, data quality panel, context marker/news panel |
| `scripts/` | 사람이 실행하는 CLI entrypoint |
| `artifacts/` | model artifact와 metadata 저장 |
| `data/` | raw/interim/processed/external/features/manifests |
| `outputs/` | backtest, plot, generated output. 오래된 Markdown report는 canonical docs로 합치고 삭제 |
| `docs/ko`, `docs/en` | 한국어 원본과 영어 mirror |

## 의존성 방향

`backend`는 `market_ai`를 호출할 수 있습니다. `market_ai`는 `backend`를 import하면 안 됩니다. 이렇게 해야 API server 없이도 data pipeline, model, backtest를 독립적으로 테스트할 수 있습니다.

```text
frontend -> backend API -> market_ai
scripts ---------------> market_ai
tests -----------------> backend / market_ai
```

## Runtime 흐름

```text
provider raw data
-> data/raw, data/interim
-> data/processed market/fundamental/event context
-> training scripts
-> artifacts/models + artifacts/metadata
-> backend forecast/context APIs
-> frontend chart overlay
```

## LLM 흐름

LLM은 `market_ai.llm` 내부에서 context encoder로 쓰이고, backend의 dashboard analysis route에서 이미 계산된 forecast/news evidence를 설명하는 prose generator로 쓰입니다.

```text
news/events
-> LocalEventContextEncoder 또는 OpenAICompatibleLLMEventEncoder
-> event_context_daily.csv
-> oil_context_fusion x_event_context
```

LLM은 numeric forecast path를 생성하지 않습니다.

## API 계층

- `/api/forecast`: 신규 typed forecast contract
- `/api/chart`: 기존 chart compatibility contract
- `/api/market-context`: 뉴스/context marker와 scenario commentary
- `/api/dashboard-analysis`: AI 시황 해설, 뉴스 해석, 예측 리포트를 한 번의 외부 LLM 호출로 생성
- `/api/backtests/visualization`: 과거 origin 기준 forecast overlay와 실제 이후 candle
- `/api/explanation`: forecast explanation
- `/api/models`, `/api/data-status`, `/api/backtests`: 운영/진단 endpoint

`/api/chart` compatibility는 명시적으로 제거하기 전까지 유지합니다.

## 호환성 계층

신규 uvicorn entrypoint는 `backend.app.main:app`입니다. 기존 `app.main:app`는 구형 실행 명령을 위한 얇은 wrapper로만 유지합니다.

## 운영 경계

- Production에서 mock/synthetic fallback을 조용히 쓰지 않습니다.
- API key는 환경변수 또는 `.env`로만 주입하고 커밋하지 않습니다.
- `.pt`/`.npz` artifact는 source code와 분리합니다.
- Documentation은 한국어/영어 mirror를 같이 갱신합니다.
