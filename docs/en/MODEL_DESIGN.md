# Model Design

Numeric forecasts are handled by time-series models and baselines. The LLM is a context/event encoder and explanation generator, not a direct price forecaster.

## Model Taxonomy

| Class | Models |
| --- | --- |
| Classical | `motif` |
| Deep learning | `pattern_mlp`, `deep_lstm_tcn_fusion`, `llm_context_seq_moe` |
| Baseline | `random_walk`, `drift`, `seasonal_naive`, `volatility_scaled_naive` |
| Backtest-only | `flat`, `simple_moving_average_path`, optional `regime_ensemble` |
| Removed/deprecated | `cycle`, `lstm`, `tcn`, `ensemble` |

## Forecast Target

The forecast target is a volatility-scaled cumulative log return distribution. Raw future price is not used directly as the training target.

```text
scaled_target_h = cumulative_log_return_h / recent_realized_volatility
price_t+h = current_price * exp(predicted_cumulative_log_return_h)
```

This structure allows the same model/feature design to extend across assets with different price levels.

## Input Features

| Input | Contents |
| --- | --- |
| `x_price` | log returns, vol-scaled returns, range, rolling volatility, momentum, drawdown, autocorrelation, trend, skew/kurtosis, cycle features |
| `x_cross_asset` | related returns, correlation, spread, relative strength, risk proxy, missing indicators |
| `x_event_context` | event/context vectors from local_rules or LLM encoders |
| `x_static` | current price, realized volatility, lookback, horizon |

EIA/CFTC/CME/event context values enter samples only when `feature_available_at <= as_of_time`.

## DeepLstmTcnFusion

`deep_lstm_tcn_fusion` projects price features and optional cross-asset features, then runs causal LSTM and causal TCN encoders in parallel. The fusion gate mixes both encoders using LSTM/TCN representations, event context, and static features.

Outputs:

- volatility-scaled cumulative log return quantiles
- `prob_up`
- expected volatility
- confidence

## LLMContextSeqMoE

`llm_context_seq_moe` is a learned mixture-of-experts with an LSTM expert, TCN expert, baseline adapter, and motif adapter.

What LLM/event context does:

- provides context to the gating network
- helps adjust uncertainty/confidence
- adds regime/event state as auxiliary features

What LLM/event context does not do:

- directly generate price paths
- directly generate p50/p90
- overwrite time-series model outputs

## Quantiles And Calibration

Quantile paths must be monotonic. Rolling backtest residuals can produce calibration artifacts.

```text
artifacts/calibration/{model}_{symbol}_{interval}.json
```

Until calibration artifacts are sufficiently validated, probabilistic bands are residual-volatility adapters, not validated confidence intervals.

## Artifacts And Metadata

- `.npz` legacy/global artifacts: `artifacts/models`
- `.pt` deep artifacts: `artifacts/models`
- metadata JSON: `artifacts/metadata`

Metadata should include at least model id, interval, horizon, lookback, training window, train/validation loss, feature sources, and a data hash or manifest reference.

## Currently Trainable Setup

The current processed data supports this setup:

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

`horizon=45` is also possible, but LLM context impact should be interpreted cautiously if news/event context history is short.

## Why Models Were Removed

- `lstm`/`tcn`: live cached training made reproducibility and operations weak.
- `cycle`: cycle phase/strength is more useful as a feature than as standalone extrapolation.
- `ensemble`: fixed-weight mixes cannot learn regime/context behavior, so they were replaced by learned MoE.
