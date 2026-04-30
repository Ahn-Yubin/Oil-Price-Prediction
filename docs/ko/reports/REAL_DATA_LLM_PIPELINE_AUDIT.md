# 실전 데이터/LLM 파이프라인 감사

작성일: 2026-04-30

## 요약

현재 저장소는 Universal Market Forecasting Dashboard 구조와 deep 모델 inference 경로를 갖추고 있다. `artifacts/models`에는 production deep `.pt` artifact와 legacy `.npz` artifact가 있으며, `/api/forecast`는 deep 모델을 선택 모델로 사용할 수 있다. 다만 감사 시작 시점의 `data/` 영역에는 `sample_market_events.csv` 외 실전 event/news/fundamental 원천 데이터가 없었고, deep 학습은 yfinance 가격과 샘플 event context 중심이었다.

## 감사 항목

| 항목 | 상태 | 근거 |
| --- | --- | --- |
| `artifacts/models` 실제 `.pt`/`.npz` 파일 | 사용 가능 | `deep_lstm_tcn_fusion_1d_h45.pt`, `llm_context_seq_moe_1d_h45.pt`, interval별 `global_dl_*.npz`가 존재한다. |
| `artifacts/metadata` deep metadata | 사용 가능 | `deep_lstm_tcn_fusion_1d_h45.json`, `llm_context_seq_moe_1d_h45.json`이 있고 둘 다 `status=available`이다. |
| `/api/models` deep status | 사용 가능 | `deep_artifact_availability()` 결과를 `user_facing_models[].status`와 training command로 반환한다. 현재 `1d/h45` deep 모델은 `available`이다. |
| deep 모델의 `/api/forecast` 사용 | 사용 가능 | `market_ai.forecasting.service._deep_comparison_models()`가 선택된 deep 모델을 `forecast_with_deep_model()`로 호출한다. |
| `llm_context_seq_moe` inference event provider | 부분 구현 | `ENABLE_LLM_CONTEXT=true`일 때만 `FileEventProvider.from_env()`가 inference에 들어간다. 기본값은 off라 event context가 0 context가 될 수 있다. |
| `sample_market_events.csv` 외 실전 event 데이터 | 미구현 | 감사 시작 시점 `data/`에는 샘플 CSV 외 raw news/event/fundamental dataset이 없었다. |
| `train_deep_fusion_models.py` 실전 데이터/context 입력 | 부분 구현 | yfinance 다운로드와 `--events-path`는 연결되어 있었지만 EIA/CFTC/CME/processed market panel 입력은 없었다. |
| cross-asset feature | placeholder | `empty_cross_asset_window()`가 missing indicator 중심 matrix를 만든다. 실제 related asset panel 값은 학습 dataset에 연결되지 않았다. |
| backtest 결과 저장 위치 | 사용 가능 | `outputs/backtests`에 summary, details, horizon metrics, probabilistic metrics, leaderboard, model availability CSV가 저장된다. |
| `PROJECT_STATUS.md` 최신성 | 불일치 | 기존 문서는 deep `1d/h45` artifact를 missing으로 설명했지만 실제 artifact와 metadata는 존재한다. |

## 사용 가능

- `/api/forecast`와 `/api/chart` contract는 유지되어 있다.
- LLM safety guardrail은 price target, p50/p90, future return path 관련 output을 warning 처리하고 숫자 forecast를 덮어쓰지 않는다.
- Forecast target은 volatility-scaled cumulative log return distribution으로 유지된다.
- Deep `.pt` artifact와 metadata sidecar는 source code와 분리되어 있다.

## 미구현 또는 부족

- EIA/CFTC/CME/manual CSV ingest와 point-in-time daily feature store.
- yfinance market panel의 재현 가능한 raw/cache/processed 저장 파이프라인.
- 뉴스 headline CSV와 event context daily dataset 생성.
- 외부 API/local HTTP/offline file LLM context 운영 모드 검증 script.
- processed data 기반 deep 학습 CLI.
- rolling leaderboard의 latest snapshot 구조.
- conformal quantile calibration artifact와 API 연결.

## Placeholder

- cross-asset feature matrix는 실제 related market panel이 아니라 missing indicator가 중심이다.
- `llm_context_seq_moe` metadata의 event input은 샘플 event CSV 기준이며 실전 뉴스/event source가 아니다.
- 기존 quantile band는 residual-volatility adapter이며 검증된 confidence interval로 볼 수 없다.

## 결론

기존 forecast/model API 표면은 재사용 가능하다. 이번 작업은 기존 모델/API를 갈아엎기보다 `data/raw`, `data/processed`, manifest, manual/live provider, LLM context cache, processed-data training, leaderboard, calibration을 추가하는 방식으로 진행한다.
