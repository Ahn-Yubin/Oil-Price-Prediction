# Oil Price Forecasting Dashboard

이 프로젝트는 유가 예측 전용 FastAPI + market_ai + frontend 저장소입니다. 목표는 WTI 원유(`CL=F`) 차트, 에너지/거시 지표, 뉴스/LLM context, 단일 통합 딥러닝 예측 모델, chart overlay를 하나의 운영 흐름으로 묶는 것입니다.

영어 문서는 [README.en.md](README.en.md)를 참고하십시오. 처음 보는 팀원은 [프로젝트 현황](docs/ko/PROJECT_STATUS.md)을 먼저 읽으십시오.

## 핵심 원칙

- Production에서 mock/synthetic 데이터를 조용히 사용하지 않습니다.
- `/api/chart` backward compatibility를 유지합니다.
- LLM은 context/event encoder입니다. 숫자 가격 예측기는 아닙니다.
- Forecast target은 volatility-scaled cumulative log return distribution입니다.
- 예측 가격은 `price_t+h = current_price * exp(predicted_cumulative_log_return_h)`로 복원합니다.
- 사용자-facing 예측 모델은 `oil_context_fusion` 하나입니다.
- `.pt`/`.npz` artifact는 `artifacts/models`, metadata JSON은 `artifacts/metadata`에 둡니다.
- 한국어 문서와 영어 mirror는 같은 상대경로 구조를 유지합니다.

## 빠른 실행

```bash
.venv/bin/python -m uvicorn backend.app.main:app --reload --port 8000
```

주요 API:

- `GET /api/health`
- `GET /api/models`
- `GET /api/data-status?symbol=CL=F&interval=1d`
- `GET /api/forecast?symbol=CL=F&interval=1d`
- `GET /api/chart?symbol=CL=F&interval=1d`
- `GET /api/market-context?symbol=CL=F&interval=1d`
- `GET /api/explanation`
- `GET /api/backtests`

## 현재 데이터 상태

현재 학습에 사용할 수 있는 데이터는 다음과 같습니다.

- `data/processed/market_panel/{interval}/panel.csv`: `1d`, `1h`, `30m`, `15m` market panel
- `data/processed/oil_fundamentals/eia_weekly.csv`: EIA weekly petroleum 데이터
- `data/processed/oil_fundamentals/cftc_cot_weekly.csv`: CFTC COT weekly 포지셔닝 데이터
- `data/processed/macro_panel/fred_daily_wide.csv`: 금리, 환율, 달러, VIX 등 FRED macro 데이터
- `data/raw/news/public_market_news.csv`: 공개 뉴스 원문
- `data/processed/event_context/event_context_daily.csv`: 뉴스/이벤트 context vector
- `data/manifests/data_inventory.json`: 데이터 inventory

제한 사항은 CME futures curve 장기 데이터, 더 긴 뉴스 history, 충분한 calibration residual입니다. 자세한 내용은 [데이터 파이프라인](docs/ko/DATA_PIPELINE.md)을 참고하십시오.

## LLM Context

LLM 사용 흐름은 다음과 같습니다.

```text
뉴스/이벤트 -> LLM context encoder -> event context vector -> 딥러닝 모델 입력 -> 시계열 모델이 숫자 예측
```

Google Gemma hosted API를 쓸 때:

```bash
export ENABLE_LLM_CONTEXT=true
export ENABLE_EXTERNAL_LLM_CALLS=true
export LLM_CONTEXT_MODE=google_generative
export LLM_API_KEY="YOUR_GOOGLE_API_KEY"
export LLM_API_BASE="https://generativelanguage.googleapis.com/v1beta"
export LLM_MODEL="gemma-3-27b-it"
```

`export`는 현재 shell에만 남습니다. 터미널을 닫으면 사라집니다. 프로젝트별 지속 설정은 `.env`를 사용합니다. 이 저장소의 주요 server/script entrypoint는 프로젝트 루트 `.env`를 자동 로드합니다.

```bash
cp .env.example .env
```

shell의 `echo "$LLM_MODEL"`은 `.env`를 자동으로 보여주지 않습니다. 프로젝트가 읽는 설정은 Python으로 확인합니다.

```bash
.venv/bin/python - <<'PY'
from market_ai.config import get_settings
s = get_settings()
print("enable_llm_context:", s.enable_llm_context)
print("enable_external_llm_calls:", s.enable_external_llm_calls)
print("llm_model:", s.llm_model)
print("llm_api_base:", s.llm_api_base)
print("llm_api_key_set:", bool(s.llm_api_key))
PY
```

연결 검증:

```bash
.venv/bin/python scripts/llm/test_llm_context.py --mode openai_compatible --dry-run
.venv/bin/python scripts/llm/test_llm_context.py --mode google_generative --live
```

자세한 설명은 [LLM Context](docs/ko/LLM_CONTEXT.md)를 참고하십시오.

## 데이터 구축

```bash
.venv/bin/python scripts/data/fetch_market_prices.py --universe oil_core --interval 1d --period 10y
.venv/bin/python scripts/data/fetch_eia_petroleum.py
.venv/bin/python scripts/data/fetch_cftc_cot.py
.venv/bin/python scripts/data/build_event_context.py --news-path data/raw/news/public_market_news.csv --symbols CL=F,BZ=F,NG=F --mode local_rules
.venv/bin/python scripts/data/build_data_inventory.py
```

공개 데이터 orchestration:

```bash
.venv/bin/python scripts/data/build_real_dataset.py \
  --universe oil_core \
  --interval 1d \
  --period 10y \
  --news-timespan 3m \
  --news-maxrecords 30
```

## 학습

현재 processed data를 사용한 단일 통합 유가 모델 학습:

```bash
.venv/bin/python scripts/train/train_deep_fusion_models.py \
  --model oil_context_fusion \
  --interval 1d \
  --horizon 30 \
  --lookback 128 \
  --universe oil_core \
  --use-processed-data \
  --market-panel data/processed/market_panel/1d/panel.csv \
  --oil-fundamentals data/processed/oil_fundamentals/eia_weekly.csv \
  --cot data/processed/oil_fundamentals/cftc_cot_weekly.csv \
  --macro-panel data/processed/macro_panel/fred_daily_wide.csv \
  --event-context data/processed/event_context/event_context_daily.csv \
  --max-samples 0 \
  --epochs 5 \
  --batch-size 64 \
  --device mps \
  --force
```

`oil_context_fusion`은 기존 `deep_lstm_tcn_fusion`의 LSTM/TCN 가격 인코더와 `llm_context_seq_moe`의 context-gated expert 구조를 하나로 통합하고, attention/pattern/motif expert branch를 포함합니다. 가격 feature, 관련 에너지/거시 시장, EIA/CFTC fundamental, FRED macro, 뉴스/event context vector를 입력으로 사용합니다.

## 검증

```bash
.venv/bin/python -m pytest tests/integration/test_api.py tests/unit/test_real_data_pipeline.py tests/unit/test_deep_dataset.py tests/unit/test_train_deep_fusion_cli_policy.py
.venv/bin/python -m compileall backend market_ai scripts
.venv/bin/python scripts/maintenance/check_docs_i18n.py --check-legacy
```

## 문서

- [프로젝트 현황](docs/ko/PROJECT_STATUS.md)
- [아키텍처](docs/ko/ARCHITECTURE.md)
- [API](docs/ko/API.md)
- [데이터 파이프라인](docs/ko/DATA_PIPELINE.md)
- [모델 설계](docs/ko/MODEL_DESIGN.md)
- [LLM Context](docs/ko/LLM_CONTEXT.md)
- [운영](docs/ko/OPERATIONS.md)
- [프론트엔드](docs/ko/FRONTEND.md)
- [백테스트](docs/ko/BACKTESTING.md)
- [로드맵](docs/ko/ROADMAP.md)
