# 실전 데이터/LLM 파이프라인 구현 보고서

작성일: 2026-04-30

## 1. 구현 요약

- 데이터 lake 디렉터리와 manifest 계층을 추가했습니다.
- yfinance market panel 저장/정리 CLI를 추가했습니다.
- EIA/CFTC/CME manual CSV ingest와 point-in-time daily 변환 경로를 추가했습니다.
- event/news CSV를 daily event context와 `llm_context_cache.jsonl`로 변환하는 pipeline을 추가했습니다.
- LLM 운영 모드 `none`, `local_rules`, `openai_compatible`, `local_http`, `offline_file`을 지원하고 live call은 명시 옵션에서만 허용합니다.
- deep dataset과 training CLI가 processed market panel, fundamentals, COT, CME curve, event context를 받을 수 있게 했습니다.
- rolling leaderboard latest output, conformal calibration artifact, `/api/backtests`, `/api/forecast` calibration status, frontend diagnostics badge/panel을 연결했습니다.

## 2. 새 데이터 파이프라인

추가 구조:

- `data/raw/market`, `data/raw/eia`, `data/raw/cftc`, `data/raw/cme`, `data/raw/events`, `data/raw/news`
- `data/interim/market`, `data/interim/fundamentals`, `data/interim/events`
- `data/processed/market_panel`, `data/processed/oil_fundamentals`, `data/processed/event_context`
- `data/features/deep_training`
- `data/manifests/data_inventory.json`, `data/manifests/latest_snapshot.json`

실행 결과:

- `fetch_market_prices.py --universe oil_core --interval 1d --period 5y` 성공
- processed market panel: 6,294 rows, symbols `CL=F,BZ=F,NG=F,RB=F,HO=F`
- event context daily: 39 rows for `CL=F,BZ=F,NG=F`
- data inventory: 11 dataset entries

## 3. 사용 가능한 실전 데이터 Source

- yfinance market prices: 동작 확인됨.
- EIA petroleum: API key 또는 manual CSV.
- CFTC COT: official/manual CSV URL 또는 manual CSV.
- CME settlements: manual CSV 또는 licensed URL. Fake scraping 없음.
- events/news: manual event CSV, news headline CSV, offline LLM cache.

## 4. LLM 운영 모드

검증 명령 결과:

- `local_rules`: 통과, embedding dim 13.
- `openai_compatible --dry-run`: 통과, 외부 호출 없음.
- `local_http --dry-run`: 통과, local_rules fallback 사용.

LLM output safety check는 price target, p50/p90, future return path를 허용하지 않습니다. LLM은 context/event encoder와 explanation 역할만 합니다.

## 5. 학습 결과

- Quick-test smoke training:
  - `deep_lstm_tcn_fusion`, `1d/h8`, synthetic smoke, 1 epoch 통과.
  - `llm_context_seq_moe`, `1d/h8`, synthetic smoke, 1 epoch 통과.
- Existing production deep artifacts:
  - `deep_lstm_tcn_fusion_1d_h45.pt`: available.
  - `llm_context_seq_moe_1d_h45.pt`: available.
- 장기 processed-data 재학습은 production artifact를 덮어쓰지 않기 위해 이번 실행에서는 수행하지 않았습니다. CLI와 dataset path는 구현/테스트했습니다.

## 6. Artifact/Metadata 상태

- `.pt`와 `.npz` artifact는 `artifacts/models`에 유지됩니다.
- deep metadata는 `artifacts/metadata`에 유지됩니다.
- quick-test artifact는 `artifacts/smoke` 아래이며 production default 후보가 아닙니다.
- calibration artifact 예시: `artifacts/calibration/motif_CL_F_1d.json`.

## 7. Backtest 결과

`CL=F`, `1d`, 5 origins smoke leaderboard:

| Rank | Model | RMSE | MAE | Pinball | Coverage 80 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `drift` | 13.115 | 9.970 | 0.044 | 0.600 |
| 2 | `pattern_mlp` | 14.487 | 11.110 | 0.054 | 0.578 |
| 3 | `random_walk` | 15.210 | 11.730 | 0.057 | 0.560 |
| 4 | `deep_lstm_tcn_fusion` | 16.379 | 13.299 | 0.068 | 0.369 |
| 5 | `llm_context_seq_moe` | 16.976 | 13.832 | 0.071 | 0.324 |
| 6 | `motif` | 17.080 | 13.534 | 0.069 | 0.413 |

이 smoke run에서는 deep 모델이 baseline보다 나빴습니다. 과장하지 않고 다음 단계는 processed fundamentals/news context로 장기 재학습 후 50+ origins leaderboard를 다시 보는 것입니다.

## 8. Calibration 상태

- `calibrate_quantiles.py --model motif --symbol CL=F --interval 1d` 실행됨.
- 현재 artifact는 `n_origins=5`라 `calibration_status=uncalibrated`.
- `/api/forecast`는 calibration artifact가 calibrated일 때만 band를 조정하고, 아니면 unvalidated warning을 유지합니다.

## 9. Dashboard 변화

- `/api/models`에 data pipeline status와 LLM context status를 추가했습니다.
- `/api/backtests`가 latest leaderboard snapshot을 우선 반환합니다.
- `/api/forecast`에 `calibration_status` additive field를 추가했습니다.
- Frontend에 band calibration badge와 Model Diagnostics/Leaderboard panel을 추가했습니다.

## 10. 테스트 결과

통과:

- `python scripts/maintenance/check_docs_i18n.py --check-legacy`
- `python -m compileall backend market_ai scripts`
- `python scripts/maintenance/smoke_test_api.py`
- `python -m pytest` -> 87 passed
- `node --check frontend/src/main.js`
- yfinance oil_core 1d 5y fetch
- event context build
- LLM local/openai-compatible/local-http dry-run
- quick-test deep training 2개
- backtest smoke

스킵:

- `npm run build`: `frontend/node_modules`가 없어 실행하지 않았습니다.
- 실제 EIA/CFTC/CME live/API ingest: API key나 licensed/manual source 파일이 제공되지 않아 parser/unit test와 CLI 구현으로 검증했습니다.

## 11. 실패/스킵 사유

- Long-horizon production retraining은 기존 production artifact를 덮어쓰는 위험이 있어 수행하지 않았습니다.
- Calibration은 5 origins smoke 결과라 calibrated로 승격하지 않았습니다.

## 12. 다음 작업

1. 실제 EIA/CFTC/CME/manual CSV 파일 적재.
2. news headline CSV 또는 외부 news provider 연동.
3. processed-data `1d/h45` 장기 재학습.
4. CL/BZ/NG 50+ origins leaderboard 실행.
5. 충분한 residual origins로 model별 conformal calibration 생성.
6. API/frontend에서 source별 stale/coverage detail 확장.
