# Backtesting

Backtesting selects a historical forecast origin, runs the model with only the data that would have been available at that origin, and compares the resulting path with the realized future price path. This project separates the single-origin dashboard backtest from the multi-origin walk-forward leaderboard.

## Current Backtest Flow

The chart backtest mode uses the historical candle selected by the user as the forecast origin.

1. `/api/backtests/visualization` sorts the full market window by time.
2. The last candle at or before `origin_time` becomes the origin.
3. The model receives only candles up to that origin.
4. The forecast path is generated for the selected `horizon`.
5. The 1D `oil_context_fusion` path passes through a path adapter that uses only origin-time price state and event/context vectors. The adapter type is recorded under `deep_model_info.oil_context_fusion.path_adapter`.
6. Online residual calibration is disabled by default. It is applied only when `ENABLE_ONLINE_RESIDUAL_CALIBRATION=true`, and even then it may use only prior forecast residuals whose outcomes were already known before the current origin.
7. Realized candles after the origin are returned as `actual_future_candles` and rendered as translucent candles.
8. MAE, RMSE, and MAPE compare forecast prices and realized closes step by step.

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
- Online residual calibration is disabled by default. If explicitly enabled, it runs only for `post_artifact_cutoff` origins and uses only prior forecast residuals whose realized horizon ended at or before `origin - horizon`; if fewer than 8 prior residuals are available, no correction is applied.

Deep artifact metadata records the full sample range (`sample_start`, `sample_end`) separately from the actual training cutoff (`train_end`, `training_cutoff`). Backtest outputs include `origin_time`, `actual_window_end`, `artifact_training_cutoff`, and `leakage_audit_status` so the reader can see how each origin relates to the artifact training window.

`leakage_audit_status` values:

- `post_artifact_cutoff`: the origin is after the artifact cutoff. This is the lowest-leakage interpretation window.
- `overlaps_artifact_sample_window`: the origin is inside the artifact sample range. Do not treat this as final out-of-sample performance.
- `benchmark_or_metadata_unavailable`: baseline model or metadata unavailable.

## Metrics

The leaderboard records:

- `mae`: mean absolute price error
- `rmse`: price error with larger misses weighted more heavily
- `mape`: mean absolute percentage error across the full forecast path, step by step. It is not an endpoint-only metric.
- `smape`: symmetric percentage error
- `mase`: error relative to naive historical movement
- `median_absolute_error`: median absolute error
- `directional_accuracy`: path direction agreement
- `final_ape_pct`: absolute percentage error at the last step
- `step_directional_accuracy`: agreement of each step-return direction
- `pred_turns`, `actual_turns`, `turn_error`: predicted/realized turning-point counts and their difference
- `range_ratio`: predicted path range relative to realized path range
- `shape_score`: path-shape score combining direction, turn count, and range ratio
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

After the 2026-06-05 update, the current reference 1D artifact is `oil_context_fusion_1d_h30`. The dashboard displays a fixed 30-day path, and the 1-week, 2-week, and 1-month labels are endpoint markers on the same h30 output. The artifact uses CL=F event context encoded by the external LLM and is saved from a chronological holdout split rather than a final-fit-all-data run, so overfitting can be inspected.

Important distinction:

- Current artifact metadata records `sample_start`, `sample_end`, `train_end`, and `training_cutoff` separately.
- Dashboard MAPE is the average step-by-step error across the full forecast path. It is not an endpoint-only error.
- Online residual calibration is disabled by default. It helped the 2026-02 jump origin but worsened other origins, so using it by default would blur model evaluation.

Key metrics from the 2026-06-05 metadata and local processed panel/LLM event context:

| Window | Origins | MAPE | Range Ratio | Shape Score | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Validation holdout | 353 | 5.45% | 1.32 | 90.5 | Average chronological holdout performance. RMSE is 5.42. |
| Test holdout | 353 | 6.93% | 1.26 | 87.6 | Average performance on the later split. RMSE is 8.02. |
| 2025-08-14, 30-day display | 1 | 2.26% | 2.01 | 73.8 | The p50 path follows this range/downside setup reasonably well. |
| 2025-11-04, 30-day display | 1 | 3.58% | 0.89 | 91.1 | The predicted range and turning behavior are close to realized movement. |
| 2026-02-17, 30-day display | 1 | 23.99% | 0.14 | 37.0 | Problem window with a future event-driven jump after the origin. |
| 2026-02-18, 30-day display | 1 | 21.77% | 0.15 | 36.8 | The p50 median path does not track the realized event jump. |

This update changes model training and evaluation rather than decorating the displayed path. `oil_context_fusion` now includes an event-shock expert and auxiliary shock/range heads. The loss function also optimizes p50 step returns, detrended path shape, path range, step volatility, curvature, and step direction. The chart's representative line is the learned p50 path without inference-time post-processing.

The 2026-02-17/18 origins need separate interpretation. At the origin, price was still in the 60s; within the next 30 days, realized price jumped into the 90s. That jump depended on events that unfolded after the origin. Without future news or future prices as input, a p50 median path cannot honestly forecast that realized path at roughly 5% MAPE.

There are two ways to make that window look like a 5% MAPE forecast: leak future prices/news, or add origin-specific post-processing. Both are overfitting risks, so they are not part of the default model or backtest. The principled next step is to represent this as event-jump probability and scenario bands, not as a p50 line pretending it knew the future.

For stricter validation, use one of these approaches:

1. Freeze an artifact at an earlier cutoff and evaluate only the later period.
2. Introduce rolling retrain or expanding retrain so each origin trains a fresh artifact using only pre-origin data.
3. Track deployable final-fit performance and walk-forward out-of-sample performance in separate tables.

## Calibration

Quantile calibration:

```bash
.venv/bin/python scripts/evaluate/calibrate_quantiles.py --model oil_context_fusion --symbol CL=F --interval 1d
```

Calibration artifacts are stored as `artifacts/calibration/{model}_{symbol}_{interval}.json`. If an artifact has `calibration_status=calibrated`, `/api/forecast` widens the band with the conformal adjustment. If no artifact exists, the forecast remains a volatility-estimated band.
