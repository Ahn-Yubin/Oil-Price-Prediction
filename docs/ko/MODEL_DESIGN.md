# 모델 설계

숫자 예측은 시계열 모델과 baseline이 담당합니다. LLM은 context/event encoder와 explanation generator이며 가격을 직접 예측하지 않습니다.

## 모델 분류

| 분류 | 모델 |
| --- | --- |
| Classical | `motif` |
| Deep learning | `pattern_mlp`, `deep_lstm_tcn_fusion`, `llm_context_seq_moe` |
| Baseline | `random_walk`, `drift`, `seasonal_naive`, `volatility_scaled_naive` |
| Backtest-only | `flat`, `simple_moving_average_path`, optional `regime_ensemble` |
| Removed/deprecated | `cycle`, `lstm`, `tcn`, `ensemble` |

## Forecast Target

Forecast target은 volatility-scaled cumulative log return distribution입니다. Raw future price를 직접 target으로 학습하지 않습니다.

```text
scaled_target_h = cumulative_log_return_h / recent_realized_volatility
price_t+h = current_price * exp(predicted_cumulative_log_return_h)
```

이 구조를 유지해야 자산 가격대가 달라도 같은 모델/feature 구조로 확장할 수 있습니다.

## 입력 Feature

| 입력 | 내용 |
| --- | --- |
| `x_price` | log return, vol-scaled return, range, rolling volatility, momentum, drawdown, autocorr, trend, skew/kurtosis, cycle feature |
| `x_cross_asset` | related return, correlation, spread, relative strength, risk proxy, missing indicator |
| `x_event_context` | local_rules 또는 LLM이 생성한 event/context vector |
| `x_static` | current price, realized volatility, lookback, horizon |

EIA/CFTC/CME/event context는 `feature_available_at <= as_of_time` 조건을 만족할 때만 sample에 들어갑니다.

## DeepLstmTcnFusion

`deep_lstm_tcn_fusion`은 가격 feature와 optional cross-asset feature를 projection한 뒤 causal LSTM encoder와 causal TCN encoder를 병렬로 사용합니다. Fusion gate는 LSTM/TCN representation, event context, static feature를 입력으로 받아 두 encoder를 섞습니다.

출력:

- volatility-scaled cumulative log return quantiles
- `prob_up`
- expected volatility
- confidence

## LLMContextSeqMoE

`llm_context_seq_moe`는 LSTM expert, TCN expert, baseline adapter, motif adapter를 가진 learned mixture-of-experts입니다.

LLM/event context의 역할:

- gating network에 context 제공
- uncertainty/confidence 조정에 도움
- regime/event state를 보조 feature로 제공

LLM/event context가 하지 않는 일:

- 가격 path 직접 생성
- p50/p90 직접 생성
- 시계열 모델 output overwrite

## Quantile과 Calibration

Quantile path는 monotonic해야 합니다. Rolling backtest residual로 calibration artifact를 만들 수 있습니다.

```text
artifacts/calibration/{model}_{symbol}_{interval}.json
```

Calibration artifact가 충분히 검증되기 전까지 probabilistic band는 residual-volatility adapter이며 검증된 confidence interval이 아닙니다.

## Artifact와 Metadata

- `.npz` legacy/global artifact: `artifacts/models`
- `.pt` deep artifact: `artifacts/models`
- metadata JSON: `artifacts/metadata`

Metadata에는 최소한 model id, interval, horizon, lookback, training window, train/validation loss, feature source, data hash 또는 manifest reference가 들어가야 합니다.

## 현재 학습 가능한 구성

현재 processed data로 다음 구성이 가능합니다.

```bash
.venv/bin/python scripts/train/train_deep_fusion_models.py \
  --model both \
  --interval 1d \
  --horizon 8 \
  --lookback 128 \
  --universe research_core \
  --use-processed-data \
  --market-panel data/processed/market_panel/1d/panel.csv \
  --oil-fundamentals data/processed/oil_fundamentals/eia_weekly.csv \
  --cot data/processed/oil_fundamentals/cftc_cot_weekly.csv \
  --event-context data/processed/event_context/event_context_daily.csv \
  --max-samples 512 \
  --epochs 3 \
  --batch-size 64 \
  --device mps \
  --force
```

`horizon=45`도 가능하지만, 뉴스/event context history가 짧으면 LLM context 효과는 제한적으로 해석해야 합니다.

## 제거된 모델의 이유

- `lstm`/`tcn`: live cached 학습 방식이라 artifact 기반 재현성과 운영 안정성이 낮았습니다.
- `cycle`: standalone extrapolation보다 cycle phase/strength feature로 쓰는 편이 낫습니다.
- `ensemble`: fixed-weight mix는 regime/context를 학습하지 못하므로 learned MoE로 대체했습니다.
