# Deep + LLM Context 구현 보고서

작성일: 2026-04-30

## 1. 구현 요약

LSTM + TCN + event/LLM context 기반 deep sequence 모델 경로를 추가하고, `/api/forecast`의 `models` query를 실제 모델 선택에 반영했다. 구형 `cycle`, live `lstm`, live `tcn`, fixed `ensemble`은 기본 API, frontend selector, backtest default에서 제거했다.

## 2. 삭제/대체한 모델

- `cycle`: standalone forecast 제거. Cycle signal은 `cycle_strength`, `cycle_phase_sin`, `cycle_phase_cos` feature로 이동.
- `lstm`: live cached request-time model 제거. `deep_lstm_tcn_fusion` LSTM encoder로 대체.
- `tcn`: live cached request-time model 제거. `deep_lstm_tcn_fusion` TCN encoder로 대체.
- `ensemble`: fixed-weight mix 제거. `llm_context_seq_moe` learned MoE로 대체.

## 3. 유지한 모델

- `motif`: historical analogue and explainability model.
- `pattern_mlp`: legacy `.npz` artifact fallback.
- `random_walk`, `drift`, `seasonal_naive`, `volatility_scaled_naive`: user-facing baselines.
- `flat`, `simple_moving_average_path`, optional `regime_ensemble`: backtest-only.

## 4. 추가된 신규 모델

- `deep_lstm_tcn_fusion`: causal LSTM encoder, causal TCN encoder, fusion gate, FiLM-style context conditioning, quantile/direction/volatility/confidence heads.
- `llm_context_seq_moe`: LSTM expert, TCN expert, baseline adapter, motif adapter, event/static-context gating network.

## 5. 추가된 주요 파일

- `market_ai/modeling/deep/*`
- `market_ai/modeling/forecasters/deep_fusion.py`
- `market_ai/data/deep_dataset.py`
- `market_ai/data/event_providers.py`
- `market_ai/data/symbol_universe.py`
- `market_ai/features/deep_features.py`
- `market_ai/features/context_features.py`
- `market_ai/schemas/deep_learning.py`
- `scripts/train/train_deep_fusion_models.py`
- `configs/symbol_universe.yaml`
- `data/external/events/sample_market_events.csv`

## 6. 수정된 주요 파일

- `market_ai/forecasting/service.py`: model selection, deep fallback, additive response fields.
- `market_ai/modeling/registry.py`: `.pt` artifact scan 추가.
- `market_ai/backtesting/runner.py`: cleaned model list and availability reporting.
- `backend/app/api/routes/forecast.py`, `chart.py`, `models.py`: model query and cleanup policy.
- `frontend/index.html`, `frontend/src/main.js`: model selector and query forwarding.
- `market_ai/llm/event_encoder.py`: local encoder, optional external adapter, safety validator.

## 7. 삭제/아카이브한 파일

파일 이동/아카이브는 하지 않았다. 대신 legacy wrapper `cycle.py`, `ensemble.py`는 명확한 removal error를 반환하도록 바꿨고, live LSTM/TCN 코드와 fixed ensemble logic은 active default path에서 제거했다.

## 8. 신규 모델 구조

두 모델 모두 volatility-scaled cumulative log return quantile을 출력한다. 가격 복원은 `current_price * exp(cumulative_log_return_h)`만 사용한다. LLM context는 gating과 confidence에만 들어가며 numeric path를 직접 만들 수 없다.

## 9. 데이터 Pipeline

Deep dataset builder는 price feature, cross-asset placeholder/missing indicator, event context vector, static feature를 생성한다. Target은 `future cumulative log return / recent_realized_volatility`다. Split은 time-based이며 random split을 사용하지 않는다.

## 10. LLM Context 처리

`LocalEventContextEncoder`는 CSV/JSON event file을 deterministic하게 읽는다. `OpenAICompatibleLLMEventEncoder`는 `ENABLE_EXTERNAL_LLM_CALLS=true`이고 key가 있을 때만 호출한다. Forbidden numeric forecast field가 나오면 validator가 warning과 함께 무시한다.

## 11. 학습 방법

```bash
python scripts/train/train_deep_fusion_models.py --model both --interval 1d --universe oil_core --epochs 10 --batch-size 64
python scripts/train/train_deep_fusion_models.py --model deep_lstm_tcn_fusion --interval 1d --quick-test --epochs 1 --max-samples 256
```

Quick synthetic training으로 `deep_lstm_tcn_fusion_1d_h8`과 `llm_context_seq_moe_1d_h8` metadata가 생성되었다.

## 12. 백테스트 방법

```bash
python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 10 --models random_walk,drift,motif,pattern_mlp,deep_lstm_tcn_fusion,llm_context_seq_moe --no-plots
```

Deep artifact가 없으면 model availability에 unavailable로 기록하고 전체 backtest는 계속된다.

## 13. API 변경

`ForecastResponse`에 additive field를 추가했다: `model_paths`, `selected_models`, `primary_model`, `deprecated_models_requested`, `removed_models_requested`, `llm_context_summary`, `deep_model_info`, `feature_version`, `artifact_status`.

`/api/forecast`는 unknown/removed model에 400을 반환한다. `/api/chart`는 기존 schema를 유지하고 compatibility query를 받을 수 있다.

## 14. Frontend 변경

Model selector를 추가하고 `/api/forecast?models=...`로 전달한다. Removed 모델은 selector에 표시하지 않는다. LLM context enabled/disabled 상태는 model 표시 텍스트에 반영한다.

## 15. 테스트 실행 결과

- `.venv/bin/python -m compileall backend market_ai scripts`: 통과
- `.venv/bin/python scripts/maintenance/check_docs_i18n.py --check-legacy`: 통과
- `.venv/bin/python scripts/maintenance/smoke_test_api.py`: 통과
- `.venv/bin/python -m pytest`: 75 passed
- Deep quick training: `deep_lstm_tcn_fusion`, `llm_context_seq_moe` 각각 1 epoch synthetic 통과

## 16. 실패/스킵 사유

- `python` command가 local shell에 없어 `.venv/bin/python`으로 실행했다.
- `npm run build`는 `frontend/node_modules`가 없어 실행하지 않았다.
- `ruff`와 `mypy` 실행 파일이 없어 실행하지 않았다.
- Network-dependent yfinance backtest CLI는 optional gate로 실행하지 않았다.

## 17. 남은 위험

- Quick synthetic artifacts는 smoke용이며 production 성능을 의미하지 않는다.
- Deep 1d default horizon artifact는 별도 full training이 필요하다.
- Cross-asset feature는 현재 missing indicator 중심 fallback이다.
- Coverage calibration은 아직 model metadata에 충분히 축적되지 않았다.

## 18. 다음 추천 작업

1. `oil_core` full training을 interval별로 실행하고 `.pt` artifact를 저장한다.
2. Backtest leaderboard를 deep artifact별로 재계산한다.
3. Event data ingestion을 실제 운영 feed와 연결하되 no-lookahead 검사를 유지한다.
4. Cross-asset feature alignment를 실제 related asset matrix로 확장한다.
