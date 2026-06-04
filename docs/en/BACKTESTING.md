# Backtesting

Backtesting selects a historical forecast origin, runs the model with only the data that would have been available at that origin, and compares the resulting path with the realized future price path. This project separates the single-origin dashboard backtest from the multi-origin walk-forward leaderboard.

## Current Backtest Flow

The chart backtest mode uses the historical candle selected by the user as the forecast origin.

1. `/api/backtests/visualization` sorts the full market window by time.
2. The last candle at or before `origin_time` becomes the origin.
3. The model receives only candles up to that origin.
4. The forecast path is generated for the selected `horizon`.
5. If the origin is after the artifact cutoff and at least 8 prior origins already have realized outcomes, online residual calibration is applied. This correction uses only previously realized forecast errors and never uses realized values after the current origin.
6. Realized candles after the origin are returned as `actual_future_candles` and rendered as translucent candles.
7. MAE, RMSE, and MAPE compare forecast prices and realized closes step by step.

This UI is a single point-in-time visualization. Use the walk-forward leaderboard below to judge model-level performance.

## Walk-Forward Leaderboard

Walk-forward analysis repeats many origins in chronological order. Each origin uses only its historical window, and the future segment is used only for scoring.

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

Both execution paths prefer `data/processed/market_panel/{interval}/panel.csv` or `panel.parquet` when available. They fall back to yfinance only when the local processed panel is missing or too short. This keeps training and evaluation on the same normalization standard and reduces network-driven result drift.

## Data And Leakage Controls

The current operational `oil_context_fusion` model uses:

- WTI and related oil market panels
- EIA weekly petroleum data
- CFTC COT data
- macro panel
- public news/event context
- recent realized volatility and static features

Leakage controls:

- Price candles passed to the model stop at the origin.
- Cross-asset, EIA, CFTC, and macro features are joined with `merge_asof(..., direction="backward")`, so each sample receives only values available at or before that date.
- Event context is aggregated only over the lookback window ending at the origin.
- Future candles are used only as `actual_future_candles` and metric targets.
- Random splits are not used. Deep learning datasets use chronological train/validation/test splits.
- Online residual calibration is enabled only for `post_artifact_cutoff` origins. It uses only prior forecast residuals whose realized horizon ended at or before `origin - horizon`; if fewer than 8 prior residuals are available, no correction is applied.

There is one metadata caveat. Current deep artifact metadata fields named `train_end` and `training_cutoff` are recorded from the full sample range, not only the training split. The actual split is still chronological and represented by `n_train`, `n_val`, and `n_test`. Backtest outputs therefore include `origin_time`, `actual_window_end`, `artifact_training_cutoff`, and `leakage_audit_status` so the reader can see whether each origin overlaps the artifact sample range.

`leakage_audit_status` values:

- `post_artifact_cutoff`: the origin is after the artifact cutoff. This is the lowest-leakage interpretation window.
- `overlaps_artifact_sample_window`: the origin is inside the artifact sample range. Do not treat this as final out-of-sample performance.
- `benchmark_or_metadata_unavailable`: baseline model or metadata unavailable.

## Metrics

The leaderboard records:

- `mae`: mean absolute price error
- `rmse`: price error with larger misses weighted more heavily
- `smape`: symmetric percentage error
- `mase`: error relative to naive historical movement
- `median_absolute_error`: median absolute error
- `directional_accuracy`: path direction agreement
- `pinball_loss`: quantile forecast quality
- `coverage_80`, `coverage_90`: realized values inside P10-P90 and P05-P95 bands
- `winkler_80`: interval width plus miss penalty

Forecast bands must not be called validated confidence intervals until coverage has actually been measured and calibration artifacts exist.

## Supported Models

The default comparison set is the operational `oil_context_fusion` model plus internal benchmarks `random_walk`, `drift`, `motif`, and `pattern_mlp`.

`seasonal_naive`, `volatility_scaled_naive`, `flat`, and `simple_moving_average_path` are backtest-only baselines. `cycle`, `lstm`, `tcn`, and `ensemble` are removed/deprecated models; explicit requests return a clear error.

## Outputs

Backtest outputs are written to `outputs/backtests`.

- `*_leaderboard.csv`: overall model ranking
- `*_summary.csv`: model-level point metric summary
- `*_horizon_metrics.csv`: horizon-specific metrics
- `*_probabilistic_metrics.csv`: band/quantile metrics
- `*_regime_metrics.csv`: regime-specific metrics
- `*_details.csv`: origin-step forecast and actual values
- `*_model_availability.csv`: artifact availability and errors
- `*_meta.json`: run settings and data source

Batch leaderboards write the same structure under `outputs/backtests/leaderboards/{timestamp}` and update `outputs/backtests/leaderboards/latest.json`. `/api/backtests` prefers the latest leaderboard when one exists.

## Current Interpretation

The current `oil_context_fusion_1d_h30` artifact metadata has a sample range ending on 2026-03-26. The local processed 1D panel ends on 2026-05-08, so the latest origin with a full 30-day realized future is also around 2026-03-26. That means the current 30-day backtest is useful for checking rolling mechanics and relative model behavior, but it is not yet a broad post-cutoff out-of-sample validation.

Key metrics rerun on 2026-06-04 with the local processed panel:

| Window | Model | Origins | MAE | RMSE | MAPE | sMAPE | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 7 days | oil_context_fusion | 12 | 6.11 | 6.97 | 6.67% | 6.71% | 7 origins overlap the artifact sample range and only 5 origins are post-cutoff. |
| 7 days, post-cutoff only | oil_context_fusion | 5 | 7.59 | 8.46 | 7.77% | 7.46% | The current data does not reach a 5% MAPE target. |
| 30 days | oil_context_fusion | 8 | 11.05 | 13.57 | 12.21% | 13.65% | All origins overlap the artifact sample range. |

Live yfinance data was separately checked through 2026-06-04 for CL=F 1D data, but the current artifact still did not reliably reach 5% MAPE during the late-May 2026 rally. Online residual calibration can reduce some high-error origins, but it can also worsen origins that are already near 6-7%, so it is bounded and requires at least 8 prior residuals. Claiming “5% MAPE achieved” at this point would be an overfitting or leakage-prone interpretation.

For stricter validation, use one of these approaches:

1. Add newer realized price data so enough post-cutoff origins exist.
2. Freeze an artifact at an earlier cutoff and evaluate only the later period.
3. Introduce rolling retrain or expanding retrain so each origin trains a fresh artifact using only pre-origin data.

## Calibration

Quantile calibration:

```bash
.venv/bin/python scripts/evaluate/calibrate_quantiles.py --model oil_context_fusion --symbol CL=F --interval 1d
```

Calibration artifacts are stored as `artifacts/calibration/{model}_{symbol}_{interval}.json`. If an artifact has `calibration_status=calibrated`, `/api/forecast` widens the band with the conformal adjustment. If no artifact exists, the forecast remains a volatility-estimated band.
