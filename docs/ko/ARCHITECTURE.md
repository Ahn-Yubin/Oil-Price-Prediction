# 아키텍처

이 프로젝트는 `backend`, `market_ai`, `frontend`, `scripts`를 명확히 분리합니다. 목표는 oil dashboard가 아니라 여러 시장 자산에 확장 가능한 Universal Market Forecasting Dashboard입니다.

## 책임 분리

- `backend/`: FastAPI route, HTTP error 처리, static frontend serving, service adapter만 둡니다.
- `market_ai/`: 데이터 수집, feature engineering, forecasting, modeling, calibration, regime, backtesting, LLM context logic을 둡니다.
- `frontend/`: TradingView overlay UI, control, panel, API client를 둡니다.
- `scripts/`: 사람이 직접 실행하는 CLI entrypoint만 둡니다.
- `artifacts/`: `.npz` model artifact와 metadata JSON을 source code와 분리합니다.
- `outputs/`: forecast, backtest, plot, report 같은 생성 산출물을 둡니다.

## 의존성 방향

`backend`는 `market_ai`를 호출할 수 있습니다. `market_ai`는 `backend`를 import하면 안 됩니다. 이렇게 해야 API server 없이도 모델, backtest, data pipeline을 독립적으로 테스트할 수 있습니다.

## 호환성 계층

신규 uvicorn entrypoint는 `backend.app.main:app`입니다. 기존 `app.main:app`는 구형 실행 명령을 위한 얇은 wrapper로만 유지합니다.
