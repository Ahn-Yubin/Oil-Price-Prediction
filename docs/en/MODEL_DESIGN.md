# Model Design

Numeric forecasts are produced by time-series models and baselines. The LLM is not a numeric price forecaster.

## Forecast Target

The forecast target remains a volatility-scaled cumulative log return distribution. Forecast prices are reconstructed as:

```text
price_t+h = current_price * exp(predicted_cumulative_log_return_h)
```

This structure makes model outputs easier to compare and calibrate across different asset price scales.

## Quantiles

Quantile paths must be monotonic. Probabilistic bands should not be called validated confidence intervals until coverage is measured by backtests.

## Artifacts and Metadata

`.npz` model artifacts live in `artifacts/models`. Metadata JSON lives in `artifacts/metadata`. Model weights should not live under source code directories.
