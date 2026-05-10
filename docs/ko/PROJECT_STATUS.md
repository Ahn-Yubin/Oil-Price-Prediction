# 프로젝트 현황

이 문서는 현재 저장소의 기준 상태를 설명합니다. 과거 감사/작업 보고서는 canonical 문서에 흡수했고, 운영 판단은 이 문서와 `DATA_PIPELINE`, `LLM_CONTEXT`, `OPERATIONS`를 기준으로 합니다.

## 한 줄 요약

프로젝트는 유가 전용 dashboard에서 범용 시장 예측 플랫폼으로 전환된 상태입니다. FastAPI backend, `market_ai` 도메인 로직, frontend chart overlay, 데이터 수집 CLI, deep learning 학습 CLI, model artifact/metadata가 분리되어 있습니다.

현재 학습 가능한 핵심 데이터는 가격 panel, EIA weekly petroleum, CFTC COT, 공개 뉴스, event context입니다. CME futures curve 장기 데이터와 더 긴 뉴스 history, calibration residual은 아직 보강 대상입니다.

## 현재 구현 범위

| 영역 | 상태 |
| --- | --- |
| Backend | `backend.app.main:app` 기준 FastAPI app. `/api/forecast`, `/api/chart`, `/api/market-context` 제공 |
| Frontend | forecast overlay, context marker, 뉴스/context panel, scenario commentary 표시 |
| Market data | yfinance 기반 `research_core` panel 구축 가능 |
| Fundamentals | EIA bulk, CFTC ZIP/manual CSV ingest 가능 |
| CME | manual CSV ingest 가능. licensed CSV 확보 필요 |
| News/context | Yahoo RSS/GDELT public news 수집, local_rules 또는 external LLM context 생성 |
| Deep learning | `deep_lstm_tcn_fusion`, `llm_context_seq_moe` 학습/metadata 저장 가능 |
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

LLM은 뉴스/이벤트를 context vector로 변환하고, `llm_context_seq_moe`의 gating/confidence/uncertainty에 간접적으로 영향을 줍니다. LLM output이 price, target price, p50/p90, future return path를 생성하거나 덮어쓰면 안 됩니다.

## 모델 상태

| 모델 | 분류 | 상태 |
| --- | --- | --- |
| `motif` | Classical | 사용 가능 |
| `pattern_mlp` | Deep `.npz` | interval별 artifact 사용 |
| `deep_lstm_tcn_fusion` | Deep `.pt` | processed data 학습 가능 |
| `llm_context_seq_moe` | Deep `.pt` + event context | LLM/event context input 사용 가능 |
| `random_walk`, `drift`, `seasonal_naive`, `volatility_scaled_naive` | Baseline | 사용 가능 |
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

- `horizon=8`: 최근 뉴스 context와 겹치는 구간이 있어 LLM/event context 효과를 smoke 수준으로 확인하기 좋습니다.
- `horizon=45`: 장기 예측 artifact는 만들 수 있지만, 뉴스 context history가 짧으면 event context 효과는 제한적입니다.
- CME curve가 없으면 term structure 관련 edge는 아직 빠집니다.
- Calibration residual이 충분하지 않으면 band를 검증된 confidence interval이라고 부르면 안 됩니다.

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
3. `llm_context_seq_moe`와 `deep_lstm_tcn_fusion` h8/h45 재학습
4. CME settlement/curve CSV 확보 후 ingest
5. rolling backtest와 quantile calibration 실행
6. longer news history 적재
7. `/api/market-context`를 dashboard에서 실제 사용하며 UX/성능 점검
