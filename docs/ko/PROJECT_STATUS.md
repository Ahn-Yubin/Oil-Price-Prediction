# 프로젝트 현황

이 문서는 현재 저장소의 기준 상태를 설명합니다. 과거 감사/작업 보고서는 canonical 문서에 흡수했고, 운영 판단은 이 문서와 `DATA_PIPELINE`, `LLM_CONTEXT`, `OPERATIONS`를 기준으로 합니다.

## 한 줄 요약

프로젝트는 다시 WTI 유가(`CL=F`) 예측 전용 dashboard와 단일 모델 운영 구조로 전환된 상태입니다. FastAPI backend, `market_ai` 도메인 로직, frontend chart overlay, 데이터 수집 CLI, deep learning 학습 CLI, model artifact/metadata가 분리되어 있습니다.

현재 학습 가능한 핵심 데이터는 가격 panel, EIA weekly petroleum, CFTC COT, 공개 뉴스, event context입니다. CME futures curve 장기 데이터와 더 긴 뉴스 history, calibration residual은 아직 보강 대상입니다.

## 현재 구현 범위

| 영역 | 상태 |
| --- | --- |
| Backend | `backend.app.main:app` 기준 FastAPI app. `/api/forecast`, `/api/chart`, `/api/market-context` 제공 |
| Frontend | 검색 심볼 입력 없이 항상 `CL=F` forecast overlay, context marker, 뉴스/context panel 표시 |
| Market data | yfinance 기반 `oil_core`와 보조 macro/related market panel 구축 가능 |
| Fundamentals | EIA bulk, CFTC ZIP/manual CSV ingest 가능 |
| CME | manual CSV ingest 가능. licensed CSV 확보 필요 |
| News/context | Yahoo RSS/GDELT public news 수집, local_rules 또는 external LLM context 생성 |
| Deep learning | 사용자-facing 단일 모델 `oil_context_fusion` 학습/metadata 저장 가능 |
| Backtest/calibration | rolling backtest와 calibration script 존재. 충분한 coverage 검증은 별도 실행 필요 |
| Docs | 한국어/영어 mirror 구조 유지 |

## 폴더 역할

| 경로 | 역할 |
| --- | --- |
| `backend/` | FastAPI route, static frontend serving, service adapter |
| `market_ai/` | data, feature, forecasting, modeling, calibration, regime, backtesting, LLM context core logic |
| `frontend/` | chart overlay UI, controls, panels |
| `scripts/` | 사람이 실행하는 data/train/evaluate/maintenance CLI |
| `artifacts/models/` | `.npz`, `.pt` model artifact |
| `artifacts/metadata/` | model metadata JSON |
| `data/` | raw/interim/processed/external/features/manifests |
| `outputs/` | 실행 산출물. 오래된 report성 Markdown은 canonical docs로 합치고 삭제 |
| `docs/ko`, `docs/en` | 한국어 원본과 영어 mirror |
| `tests/` | unit/integration tests |

## API 상태

| Endpoint | 역할 |
| --- | --- |
| `GET /api/health` | 설정, model artifact, provider 상태 |
| `GET /api/models` | model registry와 artifact availability |
| `GET /api/data-status` | symbol/interval별 데이터 상태 |
| `GET /api/forecast` | 신규 forecast contract |
| `GET /api/chart` | 기존 chart compatibility contract. 제거 금지 |
| `GET /api/market-context` | 뉴스, context point, 시나리오 해설 |
| `GET /api/explanation` | forecast와 optional LLM context 설명 |
| `GET /api/backtests` | backtest output 조회 |

새 field는 additive로만 추가합니다. `/api/chart`는 기존 frontend와의 backward compatibility 대상입니다.

## 예측 정책

숫자 예측은 LLM이 아니라 시계열 모델과 baseline이 담당합니다.

```text
target = future cumulative log return / recent realized volatility
price_t+h = current_price * exp(predicted_cumulative_log_return_h)
```

LLM은 뉴스/이벤트를 context vector로 변환하고, `oil_context_fusion`의 context expert/gating/confidence/uncertainty에 간접적으로 영향을 줍니다. LLM output이 price, target price, p50/p90, future return path를 생성하거나 덮어쓰면 안 됩니다.

## 모델 상태

| 모델 | 분류 | 상태 |
| --- | --- | --- |
| `oil_context_fusion` | Unified deep `.pt` | 사용자-facing 단일 운영 모델. LSTM, TCN, attention, context expert를 통합 |
| `motif`, `pattern_mlp` | Internal benchmark | 운영 선택지는 아니며 fallback/backtest 비교용 |
| `deep_lstm_tcn_fusion`, `llm_context_seq_moe` | Legacy merged | 구조가 `oil_context_fusion`으로 통합됨 |
| `random_walk`, `drift`, `seasonal_naive`, `volatility_scaled_naive` | Baseline | backtest/fallback 비교용 |
| `flat`, `simple_moving_average_path`, `regime_ensemble` | Backtest-only | 운영 forecast default 아님 |
| `cycle`, `lstm`, `tcn`, `ensemble` | Removed/deprecated | active model 아님 |

현재 deep 학습 산출물은 `artifacts/models`와 `artifacts/metadata`에 저장됩니다. Smoke/quick artifact는 production 성능을 의미하지 않습니다.

## 현재 데이터와 크기

정확한 row count와 기간은 `data/manifests/data_inventory.json`이 기준입니다. 현재 파이프라인에서 확인된 주요 데이터는 다음과 같습니다.

| 데이터 | 경로 | 설명 |
| --- | --- | --- |
| Market panel | `data/processed/market_panel/{interval}/panel.csv` | `1d`, `1h`, `30m`, `15m` 가격 panel |
| EIA weekly | `data/processed/oil_fundamentals/eia_weekly.csv` | 장기 petroleum weekly series |
| CFTC COT | `data/processed/oil_fundamentals/cftc_cot_weekly.csv` | weekly positioning series |
| FRED macro | `data/processed/macro_panel/fred_daily_wide.csv` | 금리, 환율, 달러, VIX 등 macro series |
| News | `data/raw/news/public_market_news.csv` | 공개 뉴스 원문 |
| Event context | `data/processed/event_context/event_context_daily.csv` | daily LLM/local context vector |
| Inventory | `data/manifests/data_inventory.json` | 데이터 품질/기간/row count 기록 |

부족한 데이터는 CME curve, 더 긴 뉴스 history, 충분한 calibration residual, sub-daily release timestamp입니다.

## LLM 설정 상태 판단

현재 repository의 주요 server/script entrypoint는 프로젝트 루트 `.env`를 자동 로드합니다. `export`는 해당 shell에만 적용되고 터미널을 닫으면 사라집니다. shell의 `echo`가 비어 있어도 `.env`를 읽는 Python process에서는 값이 보일 수 있습니다.

검증:

```bash
.venv/bin/python - <<'PY'
from market_ai.config import get_settings
s = get_settings()
print("llm_model:", s.llm_model)
print("llm_api_key_set:", bool(s.llm_api_key))
PY
.venv/bin/python scripts/llm/test_llm_context.py --mode google_generative --live
```

성공 기준은 `safety_check_passed=true`이고 `External LLM fallback` warning이 없는 것입니다.

## 현재 학습 가능 여부

현재 보유 데이터로 학습은 가능합니다. 단, 해석은 다음처럼 제한해야 합니다.

- `horizon=30`: 운영 기준 artifact입니다. 화면의 7/14/30 선택지는 h30 경로의 앞부분을 잘라 표시합니다.
- `horizon=7`, `horizon=14`: 별도 모델을 만들지 않고 h30 결과의 앞부분을 표시합니다. 같은 판단에서 나온 경로라 기간을 바꿔도 일관성이 높습니다.
- 30보다 긴 기간은 반복 호출로 억지 연결하지 않고, 필요하면 별도 h60/h90 artifact를 학습해야 합니다.
- CME curve가 없으면 term structure 관련 edge는 아직 빠집니다.
- Calibration residual이 충분하지 않으면 band를 검증된 confidence interval이라고 부르면 안 됩니다.

2026-06-02 기준 `oil_context_fusion` 1D/1H h30 artifact는 `oil_core` processed panel, EIA/CFTC/FRED macro, event context를 사용해 재학습되었습니다. 단일 artifact 내부 expert system은 `lstm`, `tcn`, `attention`, `context`, `pattern`, `motif`입니다. 1D h30은 train 8,252 / validation 1,768 / test 1,768 샘플, validation RMSE 3.1088, test RMSE 3.5934입니다. 1H h30은 train 45,962 / validation 9,849 / test 9,849 샘플, validation RMSE 0.4963, test RMSE 2.1368입니다.

대표 학습 명령은 `docs/ko/OPERATIONS.md`의 학습 섹션을 기준으로 합니다.

## 완료된 주요 개선

- EIA bulk download/normalize 경로 추가
- CFTC ZIP/CSV/manual CSV normalize 경로 추가
- CME manual/URL CSV normalize 경로 정리
- Public news 수집과 event context 생성 경로 추가
- Deep dataset recent-origin sampling 최적화
- `train_deep_fusion_models.py --use-processed-data` 처리 수정
- `/api/market-context` route 추가
- Frontend context marker/news/scenario panel 추가
- LLM context 설정과 Google OpenAI-compatible endpoint 문서화
- 오래된 report성 Markdown 삭제 및 canonical docs 업데이트

## 다음 우선순위

1. `.env` 또는 shell export로 Google LLM live call 성공 여부 확인
2. LLM context를 live로 재생성
3. `oil_context_fusion` 1D/1H h30 rolling backtest와 calibration
4. CME settlement/curve CSV 확보 후 ingest
5. rolling backtest와 quantile calibration 실행
6. longer news history 적재
7. `/api/market-context`를 dashboard에서 실제 사용하며 UX/성능 점검
