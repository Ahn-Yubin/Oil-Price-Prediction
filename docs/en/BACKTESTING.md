# Backtesting

Backtesting is the validation layer for point accuracy, quantile quality, and regime-level performance.

## Run

```bash
python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 5 --models random_walk,drift,flat --no-plots
```

## Code Location

Reusable backtest logic lives in `market_ai/backtesting/runner.py`. `scripts/backtest/run_backtest.py` is the human-run CLI wrapper.

## Outputs

Backtest outputs are written to `outputs/backtests`. Plots belong under `outputs/backtests/plots` or `outputs/plots`.

## Principles

Backtests should use rolling or expanding origins without future leakage. Forecast band coverage and pinball loss should be recorded with point metrics to evaluate probabilistic forecast quality.
