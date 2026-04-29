# Backtesting

Backtesting verifies point accuracy, quantile quality, horizon-specific performance, and regime-specific performance.

## Supported Models

The default comparison set is `random_walk`, `drift`, `seasonal_naive`, `volatility_scaled_naive`, `flat`, `motif`, `pattern_mlp`, `deep_lstm_tcn_fusion`, and `llm_context_seq_moe`.

`cycle`, `lstm`, `tcn`, and `ensemble` are removed/deprecated models; explicit requests return a clear error.

## Run

```bash
python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 10 --models random_walk,drift,motif,pattern_mlp,deep_lstm_tcn_fusion,llm_context_seq_moe --include-regime-breakdown --no-plots
```

## Code Location

Reusable backtest logic lives in `market_ai/backtesting/runner.py`. `scripts/backtest/run_backtest.py` is the human-run CLI wrapper.

## Outputs

Backtest outputs are written to `outputs/backtests`. `model_availability.csv` records models that are unavailable because deep artifacts are missing. Plots live in `outputs/backtests/plots`.

## Principles

Backtests must use rolling/expanding origins without future leakage. Deep models also use only close data and event context available at the origin. Coverage and pinball loss are recorded to evaluate probabilistic forecast quality.
