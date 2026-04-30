# Model Design

Numeric forecasting is handled by time-series models and baselines. The LLM is a context/event encoder and explanation generator, not a numeric price forecaster.

## Final Model Taxonomy

- Classical: `motif`
- Deep learning: `pattern_mlp`, `deep_lstm_tcn_fusion`, `llm_context_seq_moe`
- Baselines: `random_walk`, `drift`, `seasonal_naive`, `volatility_scaled_naive`
- Backtest-only: `flat`, `simple_moving_average_path`, optional `regime_ensemble`
- Removed/deprecated: `cycle`, `lstm`, `tcn`, `ensemble`

## Forecast Target

The forecast target remains a volatility-scaled cumulative log return distribution. Raw future price is not used as the training target.

```text
price_t+h = current_price * exp(predicted_cumulative_log_return_h)
```

## DeepLstmTcnFusion

`deep_lstm_tcn_fusion` projects price features and optional cross-asset features, then runs a causal LSTM encoder and causal TCN encoder in parallel. The fusion gate consumes `lstm_repr`, `tcn_repr`, event context, and static features to mix the two encoders. Event/static context is injected into the fused representation with FiLM-style conditioning.

The heads output seven volatility-scaled cumulative log return quantiles, `prob_up`, expected volatility, and confidence.

## LLMContextSeqMoE

`llm_context_seq_moe` is a learned MoE with LSTM expert, TCN expert, baseline adapter, and motif adapter. LLM/event context affects only gating and uncertainty/confidence; it cannot directly create the numeric price path.

## Removal Rationale

- Old `lstm`/`tcn`: request-time live cached training lacked artifact-based reproducibility and operational stability.
- `cycle`: standalone extrapolation was weak, but cycle phase/strength remains useful as a feature.
- `ensemble`: fixed weights could not learn regime/context-dependent routing, so it is replaced by learned MoE.

## Quantiles

Quantile paths must be monotonic. Until coverage is measured by backtests, probabilistic bands must not be described as validated confidence intervals.

Rolling backtest residuals can create conformal artifacts at `artifacts/calibration/{model}_{symbol}_{interval}.json`. If an artifact has `calibration_status=calibrated`, the API adjusts the band; otherwise it keeps the unvalidated residual-volatility adapter warning.

## Real Feature Inputs

Processed-data training can use these inputs in addition to the price window:

- `data/processed/market_panel/{interval}/panel.parquet` or CSV fallback
- `data/processed/oil_fundamentals/eia_weekly.csv`
- `data/processed/oil_fundamentals/cftc_cot_weekly.csv`
- `data/processed/oil_fundamentals/cme_curve_daily.csv`
- `data/processed/event_context/event_context_daily.csv`

EIA/CFTC/CME/event context values enter samples only when `feature_available_at <= as_of_time` through `merge_asof`. To preserve artifact compatibility, fundamental/COT/CME summaries currently map into the spread, relative strength, and risk proxy slots of `x_cross_asset`; event/LLM vectors enter `x_event_context`.

## Artifacts and Metadata

`.npz` and `.pt` model artifacts belong in `artifacts/models`. Metadata JSON belongs in `artifacts/metadata`. If an artifact is missing, the API returns `artifact_status` plus a warning and falls back when possible.
