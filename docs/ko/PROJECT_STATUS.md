# 프로젝트 현황

이 문서는 현재 저장소를 빠르게 이해하기 위한 지도입니다. 구현 범위, 폴더 역할, 예측선 생성 과정, LLM 사용 상태, 모델과 백테스트 구조, 다음 작업 순서를 한 곳에 정리합니다.

## 현재 한 줄 요약

현재 프로젝트는 유가 전용 dashboard에서 범용 시장 예측 플랫폼으로 재구성된 상태입니다. FastAPI backend, `market_ai` 도메인 로직, Vite frontend, CLI scripts, model artifacts가 분리되어 있으며, 차트는 yfinance 시장 데이터와 가격 시계열 모델이 만든 forecast quantile path를 TradingView Lightweight Charts 스타일로 표시합니다.

LLM은 숫자 예측에 사용되지 않습니다. `LocalEventContextEncoder`는 CSV/JSON event context를 deterministic하게 벡터화하고, `OpenAICompatibleLLMEventEncoder`는 `ENABLE_EXTERNAL_LLM_CALLS=true`와 `LLM_API_KEY`가 모두 있을 때만 외부 chat completions endpoint를 호출할 수 있습니다. 어떤 LLM 경로도 price, p50, p90, return path를 생성하거나 덮어쓰지 않습니다.

2026-04-30 기준 모델 taxonomy는 다음과 같습니다.

- Classical: `motif`
- Deep learning: `pattern_mlp`, `deep_lstm_tcn_fusion`, `llm_context_seq_moe`
- Baselines: `random_walk`, `drift`, `seasonal_naive`, `volatility_scaled_naive`
- Backtest-only: `flat`, `simple_moving_average_path`, optional `regime_ensemble`
- Removed/deprecated: `cycle`, `lstm`, `tcn`, `ensemble`

`/api/forecast`의 `models` query는 실제 선택에 반영됩니다. Removed model은 forecast API에서 400을 반환하고, `/api/chart` compatibility path에서는 warning과 fallback을 사용할 수 있습니다. `/api/chart` backward compatibility는 유지 대상입니다.

## 폴더별 역할

| 경로 | 역할 | 현재 상태 |
| --- | --- | --- |
| `backend/` | FastAPI app, API route, HTTP error 처리, static frontend serving | 동작 중인 API entrypoint는 `backend.app.main:app`입니다. |
| `market_ai/` | 데이터, feature, forecasting, modeling, calibration, regime, backtesting, LLM context 등 핵심 도메인 로직 | 가장 중요한 구현이 모여 있는 core package입니다. |
| `frontend/` | dashboard HTML/CSS/JS, chart rendering, API fetch logic | `/api/forecast`를 먼저 호출하고 실패하면 `/api/chart`로 fallback합니다. |
| `scripts/` | 사람이 직접 실행하는 CLI entrypoint | 학습, 백테스트, 데이터, 유지보수 명령이 있습니다. |
| `artifacts/models/` | production model artifact | `.npz` artifact와 production deep `.pt` artifact 위치입니다. |
| `artifacts/metadata/` | production model metadata JSON | model artifact와 분리된 sidecar metadata가 있습니다. |
| `artifacts/smoke/` | quick-test/smoke artifact | production registry에서 사용하지 않는 smoke artifact 위치입니다. |
| `data/` | raw/interim/processed/external/features 데이터 저장 영역 | 현재 runtime 기본 데이터 경로입니다. |
| `outputs/` | 백테스트 CSV, plot 등 생성 산출물 | 실행 결과가 쌓이는 위치입니다. |
| `docs/ko`, `docs/en` | 한국어 원본 문서와 영어 mirror | 같은 상대경로 구조를 유지해야 합니다. |
| `tests/` | unit/integration test | API, data status, feature, model registry, baseline, backtest 일부를 검증합니다. |
| `app/` | 구형 실행 명령 compatibility wrapper | 새 코드는 `backend/`를 기준으로 봅니다. |

## 주요 API

| Endpoint | 역할 |
| --- | --- |
| `GET /api/health` | 설정, model artifact 상태, data provider 상태 확인 |
| `GET /api/models` | 등록된 artifact metadata, logical model, deep artifact availability 확인 |
| `GET /api/data-status` | 특정 symbol/interval의 데이터 상태 확인 |
| `GET /api/forecast` | 신규 forecast contract. candles, quantile forecast, scenario, regime, model metadata 포함 |
| `GET /api/chart` | 기존 chart frontend 호환 payload. backward compatibility 대상 |
| `GET /api/explanation` | forecast 결과와 optional LLM context 기반 설명 |
| `GET /api/features` | feature 관련 API |
| `GET /api/backtests` | backtest output와 model availability 조회용 API |

## 예측선 생성 과정

현재 차트의 중심 예측선은 `ForecastResponse.forecast[].p50`입니다.

1. yfinance에서 최근 OHLC close price를 가져옵니다.
2. close price를 log return으로 변환합니다.
3. 선택된 forecasting path를 계산합니다.
4. 기본 primary 후보는 사용 가능한 artifact를 기준으로 구성됩니다. artifact가 없는 deep 모델은 default candidate에서 제외되고 `artifact_status`에만 기록됩니다.
5. 예측 target은 volatility-scaled cumulative log return distribution입니다.
6. 가격은 `current_price * exp(predicted_cumulative_log_return_h)`로 복원됩니다.
7. quantile band는 residual-volatility adapter로 구성됩니다.
8. rolling coverage가 충분히 측정되기 전까지 이 band를 검증된 confidence interval이라고 부르지 않습니다.

Frontend는 `/api/forecast`가 성공하면 p50, p10, p90, p05, p95와 bull/base/bear scenario overlay를 그립니다. `/api/chart` fallback도 동일한 기본 field를 유지하지만, removed model인 `cycle`, `lstm`, `tcn`, `ensemble`을 active comparison model로 다시 도입하지 않습니다.

## 현재 모델 상태

| 모델 | 분류 | 설명 | 운영 상태 |
| --- | --- | --- | --- |
| `motif` | Classical | 최근 return window와 과거 window의 shape similarity를 비교해 비슷한 과거 이후 경로를 가중 평균 | default 후보 |
| `pattern_mlp` | Deep learning artifact | `.npz` global MLP. pattern feature를 입력으로 volatility-scaled cumulative return path 예측 | interval별 artifact 있음 |
| `deep_lstm_tcn_fusion` | Deep learning artifact | LSTM+TCN sequence model. `.pt` artifact 필요 | code complete, `1d/h45` full artifact 필요 |
| `llm_context_seq_moe` | Deep learning artifact | event/context vector를 입력으로 쓰는 sequence MoE. LLM은 context encoder 역할만 수행 | code complete, `1d/h45` full artifact 필요 |
| `random_walk` | Baseline | no-drift random walk baseline | 사용 가능 |
| `drift` | Baseline | 최근 평균 return drift baseline | 사용 가능 |
| `seasonal_naive` | Baseline | interval별 최근 seasonal return pattern 반복 | 사용 가능 |
| `volatility_scaled_naive` | Baseline | 최근 volatility와 momentum 방향성을 이용한 baseline | 사용 가능 |
| `flat` | Backtest-only | 가격 변화 없음 baseline | backtest 전용 |
| `simple_moving_average_path` | Backtest-only | SMA 쪽으로 점진 수렴하는 baseline | backtest 전용 |
| `regime_ensemble` | Backtest-only optional | regime-aware baseline mixture | optional backtest 전용 |
| `cycle`, `lstm`, `tcn`, `ensemble` | Removed/deprecated | 과거 live comparison model | active model 아님 |

현재 production `.npz` artifact:

| Interval | Artifact | Lookback | Horizon | Target |
| --- | --- | ---: | ---: | --- |
| `1d` | `global_dl_1d_h45.npz` | 64 | 45 | volatility-scaled cumulative returns |
| `1h` | `global_dl_1h_h72.npz` | 96 | 72 | volatility-scaled cumulative returns |
| `30m` | `global_dl_30m_h120.npz` | 120 | 120 | volatility-scaled cumulative returns |
| `15m` | `global_dl_15m_h192.npz` | 144 | 192 | volatility-scaled cumulative returns |

Deep `.pt` artifact 상태:

- `deep_lstm_tcn_fusion_1d_h45.pt`: 아직 없음. `/api/models`에서 `artifact_missing`으로 표시됩니다.
- `llm_context_seq_moe_1d_h45.pt`: 아직 없음. `/api/models`에서 `artifact_missing`으로 표시됩니다.
- quick-test `1d/h8` artifact는 smoke 용도이며 dashboard default `1d/h45` candidate로 사용하지 않습니다.
- `status=smoke_only` 또는 `status=synthetic_only` metadata는 production available로 보지 않습니다.

## Deep 학습 명령

Production `1d/h45` 기본 artifact:

```bash
python scripts/train/train_deep_fusion_models.py --model deep_lstm_tcn_fusion --interval 1d --universe oil_core --epochs 10 --batch-size 64 --force --no-llm-context
```

Event context MoE artifact:

```bash
python scripts/train/train_deep_fusion_models.py --model llm_context_seq_moe --interval 1d --universe oil_core --epochs 10 --batch-size 64 --force --llm-context --events-path data/external/events/sample_market_events.csv
```

Quick smoke artifact:

```bash
python scripts/train/train_deep_fusion_models.py --model deep_lstm_tcn_fusion --interval 1d --quick-test --epochs 1 --max-samples 128
```

No event context 학습:

```bash
python scripts/train/train_deep_fusion_models.py --model deep_lstm_tcn_fusion --interval 1d --universe oil_core --epochs 5 --batch-size 64 --force --no-llm-context
```

Training CLI는 production 기본값에서 yfinance 실패 시 synthetic fallback을 조용히 사용하지 않습니다. Synthetic 데이터는 `--synthetic`, `--quick-test`, 또는 명시적 `--allow-synthetic-fallback`에서만 허용됩니다.

## LLM 상태

현재 LLM은 numeric price forecaster가 아닙니다.

- 기본값은 `ENABLE_LLM_CONTEXT=false`입니다.
- `LocalEventContextEncoder`는 CSV/JSON event context를 deterministic feature로 변환합니다.
- `OpenAICompatibleLLMEventEncoder`는 `ENABLE_EXTERNAL_LLM_CALLS=true`와 `LLM_API_KEY`가 있을 때만 외부 chat completions endpoint를 호출할 수 있습니다.
- Development에서 API key 없이 LLM context를 켜면 mock encoder가 structured context만 만듭니다.
- LLM output은 forecast price path, p50, p90, return path를 생성하거나 덮어쓰지 않습니다.

## Warning 정책

`ForecastResponse.warnings: list[str]`는 backward compatibility를 위해 유지됩니다. 새 응답에는 `warning_objects`가 추가될 수 있으며 각 항목은 `code`, `severity`, `message`, `action`을 가집니다.

- Artifact missing: 사용자가 deep model을 명시 선택했을 때 `warning`으로 표시하고 학습 명령을 제공합니다. 기본 dashboard load에서는 missing deep artifact를 노란 warning으로 띄우지 않습니다.
- Quantile uncalibrated: 기존 문구 `Quantile bands are residual-volatility adapters and are not validated coverage intervals yet.`를 유지하되 `info` severity로 표시합니다.
- Data stale/fallback/mock: data quality 문제이므로 `warning` severity로 표시합니다.

## 백테스트 방식

백테스트 entrypoint:

```bash
.venv/bin/python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 5 --models random_walk,drift,motif,pattern_mlp --no-plots
```

백테스트는 walk-forward rolling origin 방식입니다. 각 origin에서는 그 시점까지의 과거 close만 사용하고, 선택한 model이 cumulative log return path를 예측합니다. 결과는 summary, horizon metrics, probabilistic metrics, leaderboard, model availability CSV로 저장됩니다. Missing deep artifact는 해당 모델 unavailable로 기록되어야 하며 전체 backtest를 실패시키지 않는 것이 목표입니다.

## 구현 완료된 것

- FastAPI route가 `backend/app/api/routes`로 분리되었습니다.
- `/api/chart` backward compatibility가 유지됩니다.
- `/api/forecast` 신규 contract가 추가되었습니다.
- `warnings`를 유지하면서 `warning_objects`를 optional additive field로 추가했습니다.
- Data status가 real, stale, fallback, mock, error로 노출됩니다.
- Mock/fallback data policy가 설정과 테스트에 반영되었습니다.
- Forecast target은 volatility-scaled cumulative log return distribution 구조로 유지됩니다.
- interval별 global `.npz` artifact와 metadata sidecar가 분리되어 있습니다.
- Deep model code path, deep dataset, event context pipeline, `/api/forecast` models query, `/api/models` availability가 구현되었습니다.
- `--events-path`는 training dataset의 `FileEventProvider`로 실제 전달됩니다.
- Cross-asset feature는 아직 placeholder/missing-indicator 중심이며 full feature matrix 단계가 아닙니다.

## 아직 제한적인 부분

- `deep_lstm_tcn_fusion`과 `llm_context_seq_moe`의 full `1d/h45` production artifact가 아직 없습니다.
- Deep quick training은 smoke 검증용이며 production 성능을 의미하지 않습니다.
- Quantile band calibration은 완전 검증 상태가 아닙니다.
- `pattern_mlp` metadata는 `legacy_npz_v1` 상태이며 data hash, git commit, train_start/train_end가 비어 있습니다.
- Cross-asset feature matrix는 실제 related asset value보다 missing indicator placeholder 비중이 큽니다.
- yfinance 의존성이 강하므로 provider abstraction과 cache/storage 계층이 더 필요합니다.

## 다음 작업 제안

추천 순서:

1. deep artifact availability and training policy
2. `1d/h45` `oil_core` full training
3. deep backtest leaderboard
4. quantile coverage calibration
5. real event data ingestion
6. cross-asset feature matrix
7. frontend model diagnostics
8. provider cache/storage

## 처음 보는 팀원의 추천 읽기 순서

1. 이 문서
2. `docs/ko/ARCHITECTURE.md`
3. `docs/ko/API.md`
4. `docs/ko/MODEL_DESIGN.md`
5. `docs/ko/BACKTESTING.md`
6. `docs/ko/LLM_CONTEXT.md`
7. `docs/ko/OPERATIONS.md`
