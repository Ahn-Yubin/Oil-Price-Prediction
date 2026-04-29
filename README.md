# Universal Market Forecasting Dashboard

이 프로젝트는 **Universal Market Forecasting Dashboard, oil as first use case**입니다. 현재 유가 예측은 첫 번째 use case이며, 최종 목표는 가격 데이터, 시계열 모델, LLM context encoder, TradingView overlay를 갖춘 범용 시장 AI 플랫폼입니다.

영어 문서는 [README.en.md](README.en.md)를 참고하십시오.

처음 보는 팀원은 [프로젝트 현황](docs/ko/PROJECT_STATUS.md)을 먼저 읽으십시오. 폴더별 역할, 현재 구현 범위, LLM/모델/백테스트/차트 예측선 흐름, 다음 작업 순서를 한 문서에 정리했습니다.

## 실행 방법

```bash
uvicorn backend.app.main:app --reload --port 8000
python scripts/train/train_pretrained_models.py --interval 1d
python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 5 --models random_walk,drift,flat --no-plots
python scripts/maintenance/check_docs_i18n.py
python scripts/maintenance/smoke_test_api.py
```

로컬 shell에 `python`이 없으면 `.venv/bin/python`을 사용하십시오.

## 환경변수

- `APP_ENV`: runtime 환경입니다. production에서는 mock fallback을 조용히 사용하지 않습니다.
- `ALLOW_MOCK_DATA`: development fallback 허용 여부입니다. 기본 예시는 `false`입니다.
- `MODEL_DIR`: `.npz` 모델 artifact 위치입니다. 기본값은 `artifacts/models`입니다.
- `METADATA_DIR`: model metadata JSON 위치입니다. 기본값은 `artifacts/metadata`입니다.
- `DATA_DIR`: runtime data 위치입니다. 기본값은 `data`입니다.
- `DEFAULT_SYMBOL`, `DEFAULT_INTERVAL`: 기본 symbol과 interval입니다.
- `ENABLE_LLM_CONTEXT`, `LLM_API_KEY`, `LLM_MODEL`: LLM context encoder 설정입니다.

## API

주요 endpoint는 다음과 같습니다.

- `GET /api/health`
- `GET /api/models`
- `GET /api/data-status?symbol=CL=F&interval=1d`
- `GET /api/forecast?symbol=CL=F&interval=1d`
- `GET /api/chart?symbol=CL=F&interval=1d`
- `GET /api/features`
- `GET /api/explanation`
- `GET /api/backtests`

`GET /api/chart`는 기존 frontend와의 backward compatibility를 유지합니다. 신규 통합은 `GET /api/forecast`를 우선 사용하십시오.

## 모델 구조

숫자 예측은 LLM이 아니라 시계열 모델과 baseline이 담당합니다. Forecast target은 volatility-scaled cumulative log return distribution 구조를 유지하며, 예측 가격은 `current_price * exp(cumulative_log_return_h)`로 복원합니다.

`.npz` 모델 가중치는 `artifacts/models`에 두고, metadata JSON은 `artifacts/metadata`에 둡니다. Source code와 artifact를 섞지 않습니다.

## 백테스트

Backtest CLI는 `scripts/backtest/run_backtest.py`입니다. 재사용 가능한 로직은 `market_ai/backtesting`에 있습니다.

```bash
python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 5 --models random_walk,drift,flat --no-plots
```

## LLM Context Encoder

LLM은 숫자 가격 예측기가 아닙니다. LLM은 뉴스, 이벤트, 거시 맥락을 구조화된 context로 encode하고 설명을 생성하는 용도로만 사용합니다. LLM 기능이 꺼져 있어도 API와 dashboard는 동작해야 합니다.

## TradingView Overlay

Frontend는 `frontend/`에 있으며 TradingView Lightweight Charts 스타일 overlay를 제공합니다. 현재 UI는 `/api/forecast`를 우선 시도하고 필요하면 `/api/chart` 호환 payload를 사용합니다.

## 문서 정책

문서 정책은 **한국어 원본 + 영어 미러**입니다. `docs/ko`와 `docs/en`은 같은 상대경로 구조를 유지해야 하며, `scripts/maintenance/check_docs_i18n.py`로 검증합니다.
