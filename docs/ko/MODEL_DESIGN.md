# 모델 설계

숫자 예측은 시계열 모델과 baseline이 담당합니다. LLM은 숫자 가격 예측기가 아니라 context/event encoder와 explanation generator입니다.

## 최종 모델 분류

- Classical: `motif`
- Deep learning: `pattern_mlp`, `deep_lstm_tcn_fusion`, `llm_context_seq_moe`
- Baselines: `random_walk`, `drift`, `seasonal_naive`, `volatility_scaled_naive`
- Backtest-only: `flat`, `simple_moving_average_path`, optional `regime_ensemble`
- Removed/deprecated: `cycle`, `lstm`, `tcn`, `ensemble`

## Forecast Target

Forecast target은 volatility-scaled cumulative log return distribution 구조를 유지합니다. Raw price를 직접 target으로 학습하지 않습니다.

```text
price_t+h = current_price * exp(predicted_cumulative_log_return_h)
```

## DeepLstmTcnFusion

`deep_lstm_tcn_fusion`은 가격 feature와 optional cross-asset feature를 projection한 뒤 causal LSTM encoder와 causal TCN encoder를 병렬로 사용합니다. Fusion gate는 `lstm_repr`, `tcn_repr`, event context, static feature를 입력으로 받아 두 encoder를 섞습니다. Event/static context는 FiLM-style conditioning으로 fused representation에 주입됩니다.

출력 head는 volatility-scaled cumulative log return quantile 7개, `prob_up`, expected volatility, confidence를 냅니다.

## LLMContextSeqMoE

`llm_context_seq_moe`는 LSTM expert, TCN expert, baseline adapter, motif adapter를 가진 learned MoE입니다. LLM/event context는 gating network와 uncertainty/confidence에만 들어가며, 가격 path를 직접 생성하지 못합니다.

## 제거 이유

- 기존 `lstm`/`tcn`: 요청 시점 live cached 학습 모델이라 artifact 기반 재현성과 운영 안정성이 낮았습니다.
- `cycle`: standalone extrapolation은 약하지만 cycle phase/strength는 feature로 가치가 있어 deep feature로 흡수했습니다.
- `ensemble`: fixed-weight mix는 regime/context를 학습하지 못하므로 learned MoE로 대체했습니다.

## Quantile

Quantile path는 monotonic해야 합니다. Coverage가 backtest로 측정되기 전에는 probabilistic band를 검증된 confidence interval이라고 부르지 않습니다.

Rolling backtest residual로 `artifacts/calibration/{model}_{symbol}_{interval}.json` conformal artifact를 만들 수 있습니다. Artifact가 `calibration_status=calibrated`이면 API는 band를 조정하고, 없으면 unvalidated residual-volatility adapter warning을 유지합니다.

## 실전 Feature 입력

Processed-data 학습에서는 가격 window 외에 다음 입력을 사용할 수 있습니다.

- `data/processed/market_panel/{interval}/panel.parquet` 또는 CSV fallback
- `data/processed/oil_fundamentals/eia_weekly.csv`
- `data/processed/oil_fundamentals/cftc_cot_weekly.csv`
- `data/processed/oil_fundamentals/cme_curve_daily.csv`
- `data/processed/event_context/event_context_daily.csv`

EIA/CFTC/CME/event context는 `feature_available_at <= as_of_time`인 값만 `merge_asof`로 sample에 들어갑니다. 현재 artifact 호환성을 위해 fundamental/COT/CME 요약값은 `x_cross_asset`의 spread, relative strength, risk proxy slot에 들어가고, event/LLM vector는 `x_event_context`에 들어갑니다.

## Artifact와 Metadata

`.npz`와 `.pt` model artifact는 `artifacts/models`에 둡니다. Metadata JSON은 `artifacts/metadata`에 둡니다. Artifact가 없으면 API는 `artifact_status`와 warning을 반환하고 가능한 fallback을 사용합니다.
