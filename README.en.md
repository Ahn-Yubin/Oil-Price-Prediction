# Universal Market Forecasting Dashboard

This project is a **Universal Market Forecasting Dashboard, oil as first use case**. Oil forecasting is the first use case, and the long-term goal is a universal market AI platform with price data, time-series models, an LLM context encoder, and TradingView overlays.

For the Korean primary document, see [README.md](README.md).

New teammates should start with [Project Status](docs/en/PROJECT_STATUS.md). It explains folder ownership, current implementation scope, LLM/model/backtest/chart forecast flow, and recommended next work in one place.

## Run

```bash
uvicorn backend.app.main:app --reload --port 8000
python scripts/train/train_pretrained_models.py --interval 1d
python scripts/train/train_deep_fusion_models.py --model both --interval 1d --universe oil_core --epochs 10 --batch-size 64
python scripts/train/train_deep_fusion_models.py --model deep_lstm_tcn_fusion --interval 1d --quick-test --epochs 1 --max-samples 256
python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 10 --models random_walk,drift,motif,pattern_mlp,deep_lstm_tcn_fusion,llm_context_seq_moe --no-plots
python scripts/maintenance/check_docs_i18n.py
python scripts/maintenance/smoke_test_api.py
```

If `python` is not available in the local shell, use `.venv/bin/python`.

## Environment Variables

- `APP_ENV`: runtime environment. Production must not silently use mock fallback data.
- `ALLOW_MOCK_DATA`: controls development fallback usage. The example default is `false`.
- `MODEL_DIR`: `.npz` model artifact location. Default: `artifacts/models`.
- `METADATA_DIR`: model metadata JSON location. Default: `artifacts/metadata`.
- `DATA_DIR`: runtime data location. Default: `data`.
- `DEFAULT_SYMBOL`, `DEFAULT_INTERVAL`: default symbol and interval.
- `ENABLE_LLM_CONTEXT`, `LLM_API_KEY`, `LLM_MODEL`: LLM context encoder settings.
- `ENABLE_EXTERNAL_LLM_CALLS`: allows external LLM calls. Default is `false`.
- `NEWS_EVENTS_PATH`, `ECONOMIC_EVENTS_PATH`, `MARKET_EVENTS_PATH`: deterministic event context file paths.

## API

Primary endpoints:

- `GET /api/health`
- `GET /api/models`
- `GET /api/data-status?symbol=CL=F&interval=1d`
- `GET /api/forecast?symbol=CL=F&interval=1d`
- `GET /api/chart?symbol=CL=F&interval=1d`
- `GET /api/features`
- `GET /api/explanation`
- `GET /api/backtests`

`GET /api/chart` preserves backward compatibility with the existing frontend. New integrations should prefer `GET /api/forecast`.

## Model Structure

Numeric forecasts are produced by time-series models and baselines, not by an LLM. The forecast target remains a volatility-scaled cumulative log return distribution, and forecast prices are reconstructed as `current_price * exp(cumulative_log_return_h)`.

Final model classification:

- Classical: `motif`
- Deep learning: `pattern_mlp`, `deep_lstm_tcn_fusion`, `llm_context_seq_moe`
- Baselines: `random_walk`, `drift`, `seasonal_naive`, `volatility_scaled_naive`
- Backtest-only: `flat`, `simple_moving_average_path`, optional `regime_ensemble`
- Removed/deprecated: `cycle`, `lstm`, `tcn`, `ensemble`

The standalone `cycle` model was removed and its signal is now a feature. The old live LSTM/TCN paths were replaced by artifact-based encoders inside `deep_lstm_tcn_fusion`. The fixed ensemble is replaced by `llm_context_seq_moe`, where LLM/event context affects gating and uncertainty but cannot create numeric paths directly.

`.npz` and `.pt` model weights live in `artifacts/models`, and metadata JSON lives in `artifacts/metadata`. Source code and artifacts stay separate. If an artifact is missing, the API returns a warning plus `artifact_status` and falls back to an available model.

## Backtesting

The backtest CLI is `scripts/backtest/run_backtest.py`. Reusable logic lives in `market_ai/backtesting`.

```bash
python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 10 --models random_walk,drift,motif,pattern_mlp,deep_lstm_tcn_fusion,llm_context_seq_moe --no-plots
```

## LLM Context Encoder

The LLM is not a numeric price forecaster. It may only encode news, events, and macro context into structured context and generate explanations. APIs and the dashboard must run when LLM features are disabled.

## TradingView Overlay

The frontend lives in `frontend/` and provides a TradingView Lightweight Charts-style overlay. The current UI tries `/api/forecast` first and falls back to the `/api/chart` compatibility payload when needed.

## Documentation Policy

The documentation policy is **Korean source + English mirror**. `docs/ko` and `docs/en` must keep identical relative paths and are verified by `scripts/maintenance/check_docs_i18n.py`.
