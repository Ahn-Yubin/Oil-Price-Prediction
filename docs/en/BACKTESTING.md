# Backtesting

Backtesting verifies point accuracy, quantile quality, horizon-specific performance, and regime-specific performance.

## Supported Models

The default comparison set is the single operational model `oil_context_fusion` plus internal benchmarks `random_walk`, `drift`, `motif`, and `pattern_mlp`.

`cycle`, `lstm`, `tcn`, and `ensemble` are removed/deprecated models; explicit requests return a clear error.

## Run

```bash
.venv/bin/python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 10 --models oil_context_fusion,random_walk,drift,motif,pattern_mlp --include-regime-breakdown --no-plots
```

## Code Location

Reusable backtest logic lives in `market_ai/backtesting/runner.py`. `scripts/backtest/run_backtest.py` is the human-run CLI wrapper.

## Outputs

Backtest outputs are written to `outputs/backtests`. `model_availability.csv` records models that are unavailable because deep artifacts are missing. Plots live in `outputs/backtests/plots`.

Leaderboard batch:

```bash
.venv/bin/python scripts/evaluate/run_model_leaderboard.py --symbols CL=F,BZ=F,NG=F --interval 1d --max-origins 50
```

This writes `leaderboard.csv`, `horizon_metrics.csv`, `probabilistic_metrics.csv`, `regime_metrics.csv`, `model_availability.csv`, and `summary.md` under `outputs/backtests/leaderboards/{timestamp}` and updates `latest.json`. `summary.md` is a generated output, not canonical documentation, and can be deleted when stale. `/api/backtests` prefers the latest leaderboard when present.

Quantile calibration:

```bash
.venv/bin/python scripts/evaluate/calibrate_quantiles.py --model motif --symbol CL=F --interval 1d
```

Calibration artifacts are stored as `artifacts/calibration/{model}_{symbol}_{interval}.json`. If an artifact has `calibration_status=calibrated`, `/api/forecast` widens the band with the conformal adjustment. If no artifact exists, the existing unvalidated warning is kept.

## Principles

Backtests must use rolling/expanding origins without future leakage. Deep models also use only close data and event context available at the origin. Coverage and pinball loss are recorded to evaluate probabilistic forecast quality.
