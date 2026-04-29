# 모델 정리 감사

작성일: 2026-04-30

## 결정 요약

구형 standalone `cycle`, live cached `lstm`/`tcn`, fixed-weight `ensemble`은 기본 API, frontend selector, backtest 기본 목록에서 제거한다. `cycle` 정보는 deep price feature의 `cycle_strength`, `cycle_phase_sin`, `cycle_phase_cos`로 흡수한다. LSTM/TCN은 신규 `deep_lstm_tcn_fusion`과 `llm_context_seq_moe` 내부 encoder로만 사용한다.

| model_name | current_file | current_role | used_by_api | used_by_frontend | used_by_backtest | artifact_based | production_ready | decision | replacement | reason | risk | migration_action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| motif | `market_ai/modeling/forecasters/motif.py` | historical analogue | yes | yes | yes | no | partial | keep | deep primary when available | explainability and analogue panel value | short history can fall back | keep and keep documented |
| pattern_mlp | `market_ai/modeling/forecasters/neural_npz.py` | legacy pretrained fallback | yes | yes | yes | yes, `.npz` | partial | keep | deep models later | useful until deep artifacts stabilize | legacy feature set | keep as fallback |
| cycle | prior `motif.py` and `backtesting/runner.py` | standalone cycle extrapolation | yes before cleanup | yes before cleanup | yes before cleanup | no | no | remove | cycle features in `deep_features.py` | weak standalone extrapolation and duplicate signal | legacy requests fail if strict | remove from defaults; chart compatibility warns |
| lstm | prior live cached path model | live trained per request | yes before cleanup | yes before cleanup | yes before cleanup | no | no | replace | `deep_lstm_tcn_fusion` | request-time training is not stable production inference | old clients may request `lstm` | 400 on forecast; warning compatibility on chart |
| tcn | prior live cached path model | live trained per request | yes before cleanup | yes before cleanup | yes before cleanup | no | no | replace | `deep_lstm_tcn_fusion` | same issue as live LSTM | old clients may request `tcn` | 400 on forecast; warning compatibility on chart |
| ensemble | `forecasters/ensemble.py`, prior fixed mix | fixed motif/cycle/MLP mix | yes before cleanup | yes before cleanup | yes before cleanup | no | no | replace | `llm_context_seq_moe` | fixed weights duplicate learned gating | old clients may request `ensemble` | 400 on forecast; warning compatibility on chart |
| flat | `backtesting/runner.py` | no-change baseline | no default | hidden | yes | no | yes baseline | backtest_only | none | useful baseline, not user-facing | none | keep in backtest only |
| drift | `baselines.py`, `backtesting/runner.py` | drift baseline | yes | yes | yes | no | yes baseline | keep | none | required benchmark | can overfit recent trend | keep |
| random_walk | `baselines.py`, `backtesting/runner.py` | no-drift baseline | yes | yes | yes | no | yes baseline | keep | none | required benchmark | none | keep |
| seasonal_naive | `baselines.py`, `backtesting/runner.py` | seasonal repeat baseline | yes | yes | yes | no | yes baseline | keep | none | required benchmark | interval season assumptions | keep |
| volatility_scaled_naive | `baselines.py`, `backtesting/runner.py` | vol-scaled directional baseline | yes | yes | yes | no | yes baseline | keep | none | target-compatible baseline | simple momentum assumption | keep |
| simple_moving_average_path | `baselines.py`, `backtesting/runner.py` | SMA mean-reversion path | no default | hidden | yes | no | baseline only | backtest_only | none | useful diagnostic baseline | not primary forecast | keep in backtest only |
| regime_ensemble | `modeling/regimes/moe.py` | heuristic regime baseline | no default | hidden | optional | no | scaffold | backtest_only | `llm_context_seq_moe` | heuristic gating is superseded by learned MoE | may still help diagnostics | remove from defaults |
| LLM event/context encoder | `market_ai/llm/event_encoder.py` | structured context encoder | explanation and deep context | badge only | dataset context | no | partial | keep | Local/OpenAI-compatible encoders | context is useful, but not numeric forecast | external calls disabled by default | add safety validator and local fallback |
| `/api/forecast` models query | `market_ai/forecasting/service.py` | ignored selector | ignored before cleanup | query not sent before cleanup | n/a | n/a | no | replace | shared model catalog | user selection did not work | bad model errors change behavior | implement strict supported list |
| `/api/chart` compatibility payload | `forecasting/service.py` | legacy chart schema | yes | yes | n/a | n/a | yes | keep | additive fields | backward compatibility required | legacy model query ambiguity | keep schema and warn |
| frontend model selector | `frontend/index.html`, `frontend/src/main.js` | absent/static overlays | no | no | n/a | n/a | partial | replace | new selector | models query must be user-controlled | artifact missing warnings | add supported model selector |
| backtest model list | `market_ai/backtesting/runner.py` | mixed legacy/default | n/a | n/a | yes | mixed | partial | replace | cleaned list | old models made comparison noisy | missing deep artifacts | record availability |

## 영향 분석

- API import 영향: `forecast_model_comparison`은 `motif`와 `pattern_mlp`만 제공하고, baselines와 deep models는 service에서 조합한다.
- Frontend 영향: selector에서 legacy 모델을 제거하고 `/api/forecast?models=...`로 전달한다.
- Backtest 영향: `FORECASTERS`에서 removed 모델을 제거했고 요청 시 명확한 error를 반환한다.
- Artifact 영향: `.npz`는 `pattern_mlp`, `.pt`는 deep models로 registry가 함께 scan한다.
