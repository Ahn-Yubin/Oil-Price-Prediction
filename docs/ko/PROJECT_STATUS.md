# 프로젝트 현황

이 문서는 팀원이 현재 저장소를 빠르게 이해하기 위한 지도입니다. 기존 세부 문서보다 먼저 읽는 것을 목표로 하며, 구현 범위, 폴더 역할, 예측선 생성 과정, LLM 사용 상태, 모델과 백테스트 구조, 다음 작업 순서를 한 곳에 정리합니다.

## 현재 한 줄 요약

현재 프로젝트는 유가 전용 dashboard에서 범용 시장 예측 플랫폼으로 재구성된 상태입니다. FastAPI backend, `market_ai` 도메인 로직, Vite frontend, CLI scripts, model artifacts가 분리되어 있으며, 차트는 yfinance 시장 데이터와 가격 시계열 모델이 만든 forecast quantile path를 TradingView Lightweight Charts 스타일로 표시합니다.

LLM은 숫자 예측에 사용되지 않습니다. `LocalEventContextEncoder`와 optional OpenAI-compatible adapter는 structured context와 설명만 만들며, 외부 호출은 `ENABLE_EXTERNAL_LLM_CALLS=true`일 때만 허용됩니다.

2026-04-30 기준 모델 정리 결과:

- Classical: `motif`
- Deep learning: `pattern_mlp`, `deep_lstm_tcn_fusion`, `llm_context_seq_moe`
- Baselines: `random_walk`, `drift`, `seasonal_naive`, `volatility_scaled_naive`
- Backtest-only: `flat`, `simple_moving_average_path`, optional `regime_ensemble`
- Removed/deprecated: `cycle`, `lstm`, `tcn`, `ensemble`

`/api/forecast`의 `models` query는 실제 선택에 반영됩니다. Removed model은 forecast API에서 400을 반환하고, `/api/chart` compatibility path에서는 warning과 fallback을 사용할 수 있습니다.

## 폴더별 역할

| 경로 | 역할 | 현재 상태 |
| --- | --- | --- |
| `backend/` | FastAPI 앱, API route, HTTP error 처리, static frontend serving | 동작 중인 API entrypoint는 `backend.app.main:app`입니다. |
| `market_ai/` | 데이터, feature, forecasting, modeling, calibration, regime, backtesting, LLM context 등 핵심 도메인 로직 | 가장 중요한 구현이 모여 있는 core package입니다. |
| `frontend/` | dashboard HTML/CSS/JS, chart rendering, API fetch logic | `/api/forecast`를 먼저 호출하고 실패하면 `/api/chart`로 fallback합니다. |
| `scripts/` | 사람이 직접 실행하는 CLI entrypoint | 학습, 백테스트, 데이터, 유지보수 명령이 있습니다. |
| `artifacts/models/` | `.npz` model artifact | interval별 `global_dl_*` artifact가 있습니다. |
| `artifacts/metadata/` | model metadata JSON | model artifact와 분리된 sidecar metadata가 있습니다. |
| `data/` | raw/interim/processed/external/features 데이터 저장 영역 | 현재 runtime 기본 데이터 경로입니다. |
| `outputs/` | 백테스트 CSV, plot 등 생성 산출물 | 실행 결과가 쌓이는 위치입니다. |
| `docs/ko`, `docs/en` | 한국어 원본 문서와 영어 mirror | 같은 상대경로 구조를 유지해야 합니다. |
| `tests/` | unit/integration test | API, data status, feature, model registry, baseline, backtest 일부를 검증합니다. |
| `app/` | 구형 실행 명령 compatibility wrapper | 새 코드는 `backend/`를 기준으로 봅니다. |

## 실행 흐름

로컬에서 dashboard를 보는 기본 명령은 다음과 같습니다.

```bash
.venv/bin/python -m uvicorn backend.app.main:app --reload --port 8000
```

브라우저에서는 다음 주소를 엽니다.

```text
http://127.0.0.1:8000
```

서버가 뜨면 backend는 `frontend/index.html`을 `/`에서 렌더링하고, `frontend/src`를 `/static`으로 mount합니다. 차트 화면은 JavaScript가 API를 호출해서 데이터를 받아 그립니다.

## 주요 API

| Endpoint | 역할 |
| --- | --- |
| `GET /api/health` | 설정, model artifact 상태, data provider 상태 확인 |
| `GET /api/models` | 등록된 artifact metadata와 logical model 목록 확인 |
| `GET /api/data-status` | 특정 symbol/interval의 데이터 상태 확인 |
| `GET /api/forecast` | 신규 forecast contract. candles, quantile forecast, scenario, regime, model metadata 포함 |
| `GET /api/chart` | 기존 chart frontend 호환 payload. backward compatibility 대상 |
| `GET /api/explanation` | forecast 결과와 optional LLM context 기반 설명 |
| `GET /api/features` | feature 관련 API |
| `GET /api/backtests` | backtest output 조회용 API |

## 데이터 흐름

1. 사용자가 frontend에서 symbol과 interval을 선택합니다.
2. frontend는 `/api/forecast?symbol=...&interval=...`를 먼저 호출합니다.
3. backend route는 `market_ai.forecasting.service.build_forecast()`를 호출합니다.
4. `build_forecast()`는 yfinance provider로 OHLC 데이터를 가져옵니다.
5. 데이터가 real, stale, fallback, mock, error 중 어떤 상태인지 `data_status`에 기록합니다.
6. forecasting service가 여러 모델 path와 quantile band를 계산합니다.
7. API response에는 candles, forecast quantile, scenario, regime, warnings, model metadata가 포함됩니다.
8. frontend는 p50을 중심 예측선으로 그리고 p10-p90, p05-p95 band를 함께 표시합니다.

Production에서는 live data fetch가 실패해도 조용히 mock data로 대체하지 않습니다. Mock/fallback은 `APP_ENV=development` 또는 `ALLOW_MOCK_DATA=true`일 때만 허용됩니다.

## 차트에 표시되는 예측선이 나오는 과정

현재 차트의 중심 예측선은 `ForecastResponse.forecast[].p50`입니다. 이 값은 다음 과정으로 만들어집니다.

1. yfinance에서 최근 OHLC close price를 가져옵니다.
2. close price를 log return으로 변환합니다.
3. 여러 forecasting path를 계산합니다.
4. 기본 primary model은 `motif`입니다. historical motif analogue가 가능하면 `motif` path를 사용하고, 충분한 motif match가 없으면 `pattern_mlp` path로 대체합니다.
5. 예측 path는 cumulative log return 형태입니다.
6. 가격은 `current_price * exp(predicted_cumulative_log_return_h)`로 복원됩니다.
7. `pattern_mlp` artifact가 만든 residual band를 사용해 p05, p10, p25, p50, p75, p90, p95 quantile 값을 구성합니다.
8. band는 아직 실제 coverage가 완전히 검증된 interval이 아니므로 dashboard와 API warning에서 unvalidated band로 취급합니다.

Frontend 관점에서는 `/api/forecast`가 성공하면 p50, p10, p90, p05, p95와 bull/base/bear scenario overlay를 그립니다. `/api/forecast`가 실패하고 `/api/chart` fallback이 사용되면 chart compatibility payload의 `forecast_models`를 통해 motif, ensemble, pattern_mlp, cycle, lstm, tcn, drift, flat, baseline 계열 overlay가 표시될 수 있습니다.

## 현재 사용되는 모델 종류

현재 구현된 logical model은 다음과 같습니다.

| 모델 | 설명 | 사용 위치 |
| --- | --- | --- |
| `motif` | 최근 return window와 과거 window의 shape similarity를 비교해 비슷한 과거 이후 경로를 가중 평균 | 현재 primary forecast 후보 |
| `pattern_mlp` | `.npz` artifact로 저장된 global MLP. pattern feature를 입력으로 volatility-scaled cumulative return path 예측 | motif fallback, ensemble, band 계산 |
| `cycle` | 최근 return의 dominant frequency를 FFT로 추정해 cycle extrapolation | comparison/ensemble |
| `lstm` | 요청 데이터에서 짧게 학습하고 cache하는 live sequence LSTM path model | comparison overlay, backtest |
| `tcn` | 요청 데이터에서 짧게 학습하고 cache하는 temporal CNN path model | comparison overlay, backtest |
| `ensemble` | motif, cycle, pattern_mlp를 가중 결합 | comparison overlay, backtest |
| `flat` | 가격 변화 없음 baseline | baseline |
| `drift` | 최근 평균 return drift baseline | baseline |
| `random_walk` | no-drift random walk baseline | baseline |
| `seasonal_naive` | interval별 최근 seasonal return pattern 반복 | baseline |
| `volatility_scaled_naive` | 최근 volatility와 momentum 방향성을 이용한 baseline | baseline |
| `simple_moving_average_path` | SMA 쪽으로 점진 수렴하는 baseline | baseline |
| `regime_ensemble` | regime-aware baseline mixture | chart compatibility payload에서 추가 비교선 |

현재 저장소에는 interval별 `pattern_mlp` artifact가 있습니다.

| Interval | Artifact | Lookback | Horizon | Target |
| --- | --- | ---: | ---: | --- |
| `1d` | `global_dl_1d_h45.npz` | 64 | 45 | volatility-scaled cumulative returns |
| `1h` | `global_dl_1h_h72.npz` | 96 | 72 | volatility-scaled cumulative returns |
| `30m` | `global_dl_30m_h120.npz` | 120 | 120 | volatility-scaled cumulative returns |
| `15m` | `global_dl_15m_h192.npz` | 144 | 192 | volatility-scaled cumulative returns |

이 artifact의 metadata는 `artifacts/metadata`에 있습니다. 모델 가중치와 metadata는 source code와 분리되어야 합니다.

## LLM은 작동하고 있나

현재 LLM은 숫자 가격 예측에 작동하지 않습니다. 정책상 LLM은 context/event encoder로만 사용하고, numeric price forecaster로 사용하지 않습니다.

현재 코드 상태는 다음과 같습니다.

- 기본값은 `ENABLE_LLM_CONTEXT=false`입니다.
- 이 상태에서 `/api/explanation`은 `NullLLMEventEncoder`와 deterministic explanation을 사용합니다.
- `ENABLE_LLM_CONTEXT=true`이고 development 환경에서 API key가 없으면 `MockLLMEventEncoder`가 structured context만 만듭니다.
- API key가 있어도 현재 `OpenAICompatibleLLMEventEncoder`는 interface placeholder이며 외부 LLM call을 수행하지 않습니다.
- LLM output은 forecast price path를 덮어쓰지 않습니다.

따라서 현재 dashboard의 예측선은 LLM 결과가 아니라 가격 시계열 모델과 baseline에서 나온 값입니다.

## 백테스트 방식

백테스트 entrypoint는 다음입니다.

```bash
.venv/bin/python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 5 --models random_walk,drift,flat --no-plots
```

현재 백테스트는 walk-forward rolling origin 방식입니다.

1. yfinance에서 close series를 다운로드합니다.
2. interval별 horizon을 정합니다. 예: `1d=45`, `1h=72`, `30m=120`, `15m=192`.
3. 여러 origin을 선택합니다.
4. 각 origin에서는 그 시점까지의 과거 close만 사용합니다.
5. 선택한 model들이 horizon만큼 cumulative log return path를 예측합니다.
6. 실제 future close와 비교합니다.
7. point metric, horizon별 metric, probabilistic metric, regime breakdown, leaderboard를 CSV로 저장합니다.

주요 metric은 MAE, RMSE, SMAPE, MASE, median absolute error, directional accuracy, pinball loss, p10-p90 coverage, p05-p95 coverage, Winkler score입니다. Coverage metric은 측정값으로 기록되지만, dashboard band를 검증 완료된 interval이라고 부르려면 더 넓은 자산/기간에서 calibration 검증을 반복해야 합니다.

## 구현 완료된 것

- FastAPI route가 `backend/app/api/routes`로 분리되었습니다.
- 도메인 로직이 `market_ai` package로 이동했습니다.
- `/api/chart` backward compatibility가 유지됩니다.
- `/api/forecast` 신규 contract가 추가되었습니다.
- Data status가 real, stale, fallback, mock, error로 노출됩니다.
- Mock/fallback data policy가 설정과 테스트에 반영되었습니다.
- yfinance 기반 market data provider가 구현되었습니다.
- symbol normalization과 interval normalization이 구현되었습니다.
- Forecast target은 volatility-scaled cumulative log return 구조로 유지됩니다.
- interval별 global `.npz` artifact와 metadata sidecar가 분리되어 있습니다.
- motif, pattern_mlp, cycle, live LSTM/TCN, ensemble, baseline forecast가 구현되었습니다.
- chart frontend가 p50 line, quantile band, scenario overlay, data status badge, regime badge, explanation panel을 표시합니다.
- 백테스트 CLI와 주요 metric CSV 출력이 구현되었습니다.
- 한국어/영어 문서 구조와 문서 hygiene check가 있습니다.

## 아직 제한적인 부분

- LLM external call은 아직 실제 구현이 아니라 adapter placeholder입니다.
- Forecast band calibration은 완전 검증 상태가 아닙니다.
- `pattern_mlp` metadata는 `legacy_npz_v1` 상태이며 data hash, git commit, train_start/train_end가 비어 있습니다.
- Live LSTM/TCN은 요청 데이터에서 짧게 학습하고 cache하는 방식이라 production-grade pretrained model registry와는 다릅니다.
- Cross-asset feature는 설정 flag가 있지만 기본값은 비활성입니다.
- Frontend는 아직 component 구조로 완전히 분리되지 않은 vanilla JS 중심 구조입니다.
- yfinance 의존성이 강하므로 provider abstraction과 cache/storage 계층이 더 필요합니다.
- 백테스트 결과를 dashboard에서 비교/탐색하는 UX는 아직 제한적입니다.

## 다음 작업 제안

우선순위는 다음 순서가 좋습니다.

1. Forecast evaluation과 band calibration을 강화합니다. 여러 symbol, interval, 기간에 대해 rolling backtest를 정례화하고 coverage, pinball, interval width를 model metadata에 연결해야 합니다.
2. Model registry를 정리합니다. artifact version, train period, data hash, git commit, supported asset class를 metadata에 채우고 `/api/models`에서 현재 배포 가능한 모델과 실험 모델을 구분합니다.
3. LLM context encoder를 실제 adapter로 구현합니다. 단, output schema는 event score와 structured context로 제한하고 forecast 숫자는 절대 생성하지 않게 guardrail test를 유지합니다.
4. Data provider 계층을 확장합니다. yfinance 외 provider, local cache, stale 기준, provider error reporting을 정리하면 production에서 data quality를 더 명확히 볼 수 있습니다.
5. Frontend를 component 단위로 나눕니다. chart, controls, status badge, model panel, explanation panel을 분리하면 팀원이 UI 변경을 안전하게 할 수 있습니다.
6. Backtest 결과를 문서와 dashboard에 연결합니다. latest leaderboard와 model별 regime performance를 `outputs/backtests`에서 API로 읽어 보여주면 모델 선택 근거가 명확해집니다.
7. 운영 문서를 보강합니다. local run, train, backtest, smoke test, troubleshooting을 하나의 onboarding path로 연결합니다.

## 처음 보는 팀원의 추천 읽기 순서

1. 이 문서
2. `docs/ko/ARCHITECTURE.md`
3. `docs/ko/API.md`
4. `docs/ko/MODEL_DESIGN.md`
5. `docs/ko/BACKTESTING.md`
6. `docs/ko/LLM_CONTEXT.md`
7. `docs/ko/OPERATIONS.md`
