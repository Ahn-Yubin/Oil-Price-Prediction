# 백테스트

백테스트는 과거의 한 시점을 예측 기준점(origin)으로 잡고, 그 시점까지 실제로 알 수 있었던 데이터만 사용해 예측한 뒤, 이후 실제 가격과 비교하는 검증 절차입니다. 이 프로젝트에서는 단일 화면용 백테스트와 여러 origin을 반복하는 전진분석 리더보드를 구분합니다.

## 현재 백테스트 방식

차트의 백테스트 모드는 사용자가 선택한 과거 candle을 기준 시점으로 사용합니다.

1. `/api/backtests/visualization`이 전체 시장 데이터를 시간순으로 정렬합니다.
2. 선택한 `origin_time` 이하의 마지막 candle을 기준점으로 선택합니다.
3. 모델 입력에는 `origin_time`까지의 candle만 전달합니다.
4. 예측 경로는 기준점 이후 `horizon` 길이만큼 생성합니다.
5. origin이 artifact cutoff 이후이고, 이미 결과가 확정된 prior origin이 최소 8개 있으면 온라인 잔차 보정을 적용합니다. 이 보정은 과거 확정 예측 오차만 사용하며 현재 origin 이후 실제값은 사용하지 않습니다.
6. 기준점 이후 실제 candle은 `actual_future_candles`로 별도 반환되어 차트에 반투명 candle로 표시됩니다.
7. MAE, RMSE, MAPE는 예측 가격과 이후 실제 종가를 같은 step끼리 비교해 계산합니다.

이 화면은 “한 시점의 point-in-time 시각화”입니다. 모델 전체 성능을 판단하려면 아래의 전진분석 리더보드를 봐야 합니다.

## 전진분석 리더보드

전진분석은 여러 origin을 시간 순서대로 반복합니다. 각 origin에서 과거 window만 사용하고, 미래 구간은 오직 채점에만 사용합니다.

```bash
.venv/bin/python scripts/backtest/run_backtest.py \
  --symbol CL=F \
  --interval 1d \
  --horizon 30 \
  --lookback 260 \
  --step 7 \
  --max-origins 50 \
  --models oil_context_fusion,random_walk,drift,motif,pattern_mlp \
  --include-regime-breakdown \
  --no-plots
```

Batch leaderboard:

```bash
.venv/bin/python scripts/evaluate/run_model_leaderboard.py \
  --symbols CL=F \
  --interval 1d \
  --max-origins 50
```

두 실행 경로 모두 `data/processed/market_panel/{interval}/panel.csv` 또는 `panel.parquet`가 있으면 이를 우선 사용합니다. 로컬 processed panel이 없거나 충분하지 않을 때만 yfinance 다운로드로 fallback합니다. 이렇게 해야 학습과 검증에서 같은 데이터 정규화 기준을 쓰고, 네트워크 상태에 따라 결과가 바뀌는 문제를 줄일 수 있습니다.

## 데이터와 누수 방지

현재 운영 모델 `oil_context_fusion`은 다음 입력을 사용합니다.

- WTI/관련 원유 가격 panel
- EIA weekly petroleum data
- CFTC COT data
- macro panel
- public news/event context
- 최근 realized volatility와 static feature

누수 방지 규칙은 다음과 같습니다.

- 가격 candle은 origin 이하만 모델에 전달합니다.
- cross-asset, EIA, CFTC, macro feature는 `merge_asof(..., direction="backward")` 방식으로 해당 날짜 이전에 사용 가능했던 값만 붙입니다.
- event context도 origin 이전 lookback window만 집계합니다.
- 미래 candle은 예측 입력이 아니라 `actual_future_candles`와 metric 계산에만 사용합니다.
- random split은 사용하지 않습니다. 학습 데이터셋은 chronological train/validation/test split을 사용합니다.
- 온라인 잔차 보정은 `post_artifact_cutoff` origin에서만 켜지고, `origin - horizon` 이전에 이미 실제값이 확정된 prior forecast residual만 사용합니다. prior residual이 8개 미만이면 보정하지 않습니다.

주의할 점도 있습니다. 현재 deep artifact metadata의 `train_end`와 `training_cutoff`는 이름과 달리 전체 sample 범위의 끝으로 기록됩니다. 실제 학습/검증/테스트는 `n_train`, `n_val`, `n_test`에 따라 시간순으로 나뉩니다. 따라서 백테스트 결과에는 `origin_time`, `actual_window_end`, `artifact_training_cutoff`, `leakage_audit_status`를 함께 기록해 origin이 artifact sample 범위와 겹치는지 확인합니다.

`leakage_audit_status` 의미:

- `post_artifact_cutoff`: origin이 artifact cutoff 이후입니다. 과최적화 위험이 가장 낮은 해석 구간입니다.
- `overlaps_artifact_sample_window`: origin이 artifact sample 범위 안에 있습니다. 모델 학습/검증/테스트 범위와 겹칠 수 있으므로 최종 out-of-sample 성능으로 해석하면 안 됩니다.
- `benchmark_or_metadata_unavailable`: baseline 또는 metadata가 없는 모델입니다.

## 성능 지표

리더보드는 다음 지표를 기록합니다.

- `mae`: 평균 절대 가격 오차
- `rmse`: 큰 오차에 더 민감한 가격 오차
- `smape`: 가격 크기를 고려한 대칭 퍼센트 오차
- `mase`: naive 변화폭 대비 상대 오차
- `median_absolute_error`: 중앙 절대 오차
- `directional_accuracy`: 경로 방향 일치율
- `pinball_loss`: 분위수 예측 품질
- `coverage_80`, `coverage_90`: 실제값이 P10-P90, P05-P95 band 안에 들어간 비율
- `winkler_80`: band 폭과 이탈 패널티를 같이 보는 점수

Coverage가 충분히 측정되고 calibration artifact가 만들어지기 전에는 band를 검증된 confidence interval이라고 부르지 않습니다.

## 지원 모델

기본 비교 대상은 단일 운영 모델 `oil_context_fusion`과 내부 benchmark인 `random_walk`, `drift`, `motif`, `pattern_mlp`입니다.

`seasonal_naive`, `volatility_scaled_naive`, `flat`, `simple_moving_average_path`는 backtest-only baseline입니다. `cycle`, `lstm`, `tcn`, `ensemble`은 removed/deprecated 모델이며 요청 시 명확한 error를 반환합니다.

## 산출물

Backtest output은 `outputs/backtests`에 기록합니다.

- `*_leaderboard.csv`: 모델별 종합 순위
- `*_summary.csv`: 모델별 point metric 요약
- `*_horizon_metrics.csv`: horizon별 metric
- `*_probabilistic_metrics.csv`: band/quantile metric
- `*_regime_metrics.csv`: regime별 metric
- `*_details.csv`: origin-step별 예측값과 실제값
- `*_model_availability.csv`: artifact availability와 오류
- `*_meta.json`: 실행 설정과 데이터 출처

Batch leaderboard는 `outputs/backtests/leaderboards/{timestamp}` 아래에 같은 구조를 만들고 `outputs/backtests/leaderboards/latest.json`을 갱신합니다. `/api/backtests`는 latest leaderboard가 있으면 이를 우선 반환합니다.

## 현재 해석 기준

`oil_context_fusion_1d_h30` artifact는 2026-03-26까지의 sample 범위를 가진 현재 metadata를 사용합니다. 로컬 processed 1D panel은 2026-05-08까지 있으므로 30일 horizon에서 사용할 수 있는 가장 최근 origin도 2026-03-26 근처입니다. 즉 현재 30일 horizon 백테스트는 “rolling mechanics와 상대 성능 점검”에는 유용하지만, artifact cutoff 이후 충분한 기간을 가진 완전한 out-of-sample 검증이라고 보기는 어렵습니다.

2026-06-04에 로컬 processed panel로 재실행한 주요 지표는 다음과 같습니다.

| 구간 | 모델 | Origins | MAE | RMSE | MAPE | sMAPE | 해석 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 7일 | oil_context_fusion | 12 | 6.11 | 6.97 | 6.67% | 6.71% | 7개 origin은 artifact sample 범위와 겹치고 5개 origin만 cutoff 이후입니다. |
| 7일 cutoff 이후만 | oil_context_fusion | 5 | 7.59 | 8.46 | 7.77% | 7.46% | 현재 데이터만으로는 5% MAPE에 도달하지 못했습니다. |
| 30일 | oil_context_fusion | 8 | 11.05 | 13.57 | 12.21% | 13.65% | 모든 origin이 artifact sample 범위와 겹칩니다. |

최신 yfinance 데이터를 별도로 확인하면 2026-06-04까지 CL=F 1D 데이터가 존재하지만, 2026년 5월 후반 급등 구간에서 현재 artifact는 7일 MAPE가 5%를 안정적으로 달성하지 못했습니다. 온라인 잔차 보정은 일부 큰 오차 origin을 낮출 수 있지만, 이미 6~7%대인 origin을 악화시키는 경우도 있어 최소 8개 prior residual이 있을 때만 제한적으로 적용합니다. 따라서 현재 상태에서 “MAPE 5% 달성”이라고 보고하면 과최적화 또는 미래정보 누수에 가까운 해석이 됩니다.

더 엄밀한 검증을 하려면 다음 중 하나가 필요합니다.

1. 더 최신 실제 가격 데이터를 추가해 cutoff 이후 origin을 충분히 확보합니다.
2. artifact를 더 과거 cutoff로 고정하고 그 이후 기간만 walk-forward로 평가합니다.
3. rolling retrain 또는 expanding retrain을 도입해 각 origin 이전 데이터로만 새 artifact를 학습합니다.

## Calibration

Quantile calibration:

```bash
.venv/bin/python scripts/evaluate/calibrate_quantiles.py --model oil_context_fusion --symbol CL=F --interval 1d
```

Calibration artifact는 `artifacts/calibration/{model}_{symbol}_{interval}.json`에 저장합니다. Artifact가 `calibration_status=calibrated`이면 `/api/forecast`가 band를 conformal adjustment로 넓히고, 없으면 volatility-estimated band 상태를 유지합니다.
