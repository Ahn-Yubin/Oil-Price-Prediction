# LLM Context

The LLM is not a numeric price forecaster in this project. It is a market news and event context encoder. Deep models can use the LLM-derived context vector as one input feature, but the LLM must not create `p50`, `p90`, target prices, or future price paths.

## Current Flow

The intended flow is:

```text
news/event text
-> LLM context encoder
-> bias, impact, uncertainty, event count, explanation, event embedding
-> raw recent-news-pool volume, directional pressure, sector pressure, and source-diversity features
-> data/processed/event_context/event_context_daily.csv
-> x_event_context input for oil_context_fusion
-> time-series model predicts volatility-scaled cumulative log return distribution
-> prices are reconstructed with current_price * exp(predicted_cumulative_log_return_h)
```

So the user-facing summary is correct: news is read by the LLM, converted into scores/context, and used as one model input. Those scores are point-in-time context features, not investment advice or numeric price targets. Local rules are only for selecting date/symbol/topic candidates from the large news pool, computing raw-news-pool aggregates, and development diagnostics. If an external LLM call fails during a training-context build, the row is not silently filled with a local estimate; the failed date must be retried through the LLM before it is merged.

Training samples do not use only the origin date. With the current `lookback=128` setup, each origin aggregates event context from the prior 128-day window into `x_event_context [27]`. The first 13 features are LLM or deterministic-encoder bias/impact/uncertainty/event-embedding values. The added 14 features come from the full raw point-in-time news pool, including news volume, selection coverage, bullish/bearish pressure, energy/geopolitical/macro/supply/demand pressure, and source diversity. Event-count fields are summed, while directional/impact/quality fields are weighted by impact score and recency decay. This prevents current geopolitical or supply shocks from being diluted by older news and prevents the bounded LLM input count from becoming a model-input bottleneck.

## Live Dashboard Pipeline

When the dashboard opens a symbol or the interval/horizon changes, it runs the following sequence.

1. The frontend first calls `/api/models?interval={interval}&horizon={horizon}`.
   This response tells the UI which models are executable for the current combination. Deep models without artifacts are disabled in the visual toggles and are not sent to forecast requests.

2. The frontend calls `/api/forecast`.
   The requested models are all currently available models for that interval/horizon. Clicking a model chip does not recalculate data; it only shows or hides already returned `model_paths`.

3. The backend loads OHLCV candles from yfinance or processed providers.
   The UI and API examples use yfinance provider symbols by default. For example, crude oil is `CL=F`, Brent is `BZ=F`, and USD/KRW is `USDKRW=X`. Legacy TradingView aliases such as `NYMEX:CL1!` are still normalized for backward compatibility only.

4. The backend computes numeric forecasts from time-series models.
   `motif`, `pattern_mlp`, baselines, and deep models produce cumulative log-return or price paths. Forecast prices are restored with `current_price * exp(predicted_cumulative_log_return_h)`. The LLM does not create numeric prices in this step.

5. When the single operational model `oil_context_fusion` is selected, the backend calls `build_live_event_context()`.
   This function fetches recent public news from Google News RSS and Yahoo Finance RSS, then applies symbol-aware query and relevance filtering. Current live dashboard context requests ask for up to 60 rows, while chat asks for up to 24 rows.

6. The news DataFrame is converted into `RawNewsItem` objects and passed to the encoder.
   If external LLM calls are enabled and the rate guard allows the call, the LLM encoder is used. Otherwise `local_rules` produces the same schema deterministically. The LLM output schema is `overall_bias`, `impact_score`, `uncertainty`, `event_count`, `explanation`, and `event_embedding[13]`. The builder then appends 14 aggregate features from the recent raw-news pool, so the model input is 27-dimensional.
   `local_rules` and raw-news aggregates promote war, attacks, sanctions, and Hormuz/Red Sea terms into geopolitical supply-shock context. This prevents mixed headlines such as “prices dip, but a US-Iran war raises supply disruption risk” from being diluted into a flat neutral signal.

7. The encoder output becomes a one-row `context_frame` for deep-model inference.
   `oil_context_fusion` uses this `x_event_context` for context expert gating, uncertainty, and confidence. The LLM does not draw the forecast line directly; the news context vector becomes an input feature and the time-series model computes the forecast path.

8. `/api/market-context?live=1` returns the same live context payload to the UI.
   Chart news markers, popovers, and the “News Interpretation” panel use the same `news`, `context_points`, and `chart_context_points`. Display context deduplicates headline/source/date rows and does not show internal placeholders or blank `-` explanations.

9. `/api/dashboard-analysis` explains already-computed model paths and news context with one external LLM call.
   This endpoint asks for AI market commentary, news interpretation, and forecast report prose in one JSON response, then renders the pieces in separate panels. Standalone `/api/model-commentary`, `/api/market-context`, and `/api/report` remain compatibility/diagnostic routes, but the operating dashboard prefers the combined endpoint to reduce external LLM calls. The server enforces a 60-second external-call guard and uses a dashboard-analysis cache. If the model appends extra text after the JSON object, the parser recovers by reading the first JSON object.

10. `/api/assistant-chat` answers short user-entered questions.
    It also does not create new numeric price forecasts. The UI shows a growing-dot assistant bubble while a chat answer is pending and blocks duplicate submits.

11. `/api/scenarios/forecast` converts a user-entered bundle of future events into structured context.
    The Scenario mode title/content/event time goes to the external LLM context encoder. The LLM returns event type, directional bias, impact, uncertainty, and event embedding instead of price numbers. The backend combines that output with each event's `event_time` to build a forecast-horizon event-context schedule, then passes an `event_context_frame` with only the events active by that horizon into `oil_context_fusion`. The future event timestamp remains in `scenario_event_time`, `model_context_schedule` metadata, and the LLM input, but future prices or realized returns never enter the model input.

## Allowed Roles

- Convert news, economic events, and supply events into structured market context.
- Produce `overall_bias`, `impact_score`, `uncertainty`, and event embeddings.
- Show historical news and the context interpretation on the dashboard.
- Affect gating, confidence, and uncertainty indirectly inside `oil_context_fusion`.
- Provide historical context markers and scenario commentary through `/api/market-context`.
- Write commentary/news/report prose from already-computed forecasts and supplied news evidence through `/api/dashboard-analysis`.
- Convert user-entered future events into structured event context through `/api/scenarios/forecast`.

## Forbidden Roles

- The LLM must not create `p50`, `p90`, target prices, or future return paths.
- LLM output must not overwrite numeric forecasts from deep models or baselines.
- Trading instructions or fixed target prices from the LLM must be ignored by validation.
- Forecast bands must not be called validated confidence intervals until coverage has actually been measured.
- Backtest-origin explanations must not use live relative wording such as current, recent, now, or today. When `origin_time` is present, the first sentence includes the absolute reference date/time.

## Implementations

- `LocalEventContextEncoder`: reads CSV/JSON event files with deterministic rules. Use it for development, diagnostics, or truly empty-context days where there is no news to send to the LLM.
- `OpenAICompatibleLLMEventEncoder`: calls an OpenAI-compatible chat completions endpoint.
- `LocalHTTPLLMEventEncoder`: calls an Ollama/vLLM-compatible local HTTP endpoint.
- `OfflineFileLLMEventEncoder`: reads a precomputed JSON/JSONL context cache.
- `NullLLMEventEncoder`: disables LLM context completely.

External calls require both `--live` and `ENABLE_EXTERNAL_LLM_CALLS=true`. Training builds that use an external LLM run in strict mode by default. If the API key is missing or a call fails, the build stops instead of writing a `local_rules` row into the training CSV. Use `--allow-external-llm-fallback` only when deliberately testing fallback behavior during development.

The dashboard runtime combined LLM call also returns an explicit unavailable state when external LLM configuration is missing. The frontend marks previous dashboard-analysis responses stale on language switches, live/backtest changes, and backtest-origin changes, then clears panel/marker state before rendering the next response so old news cannot remain on a new chart.

## Google Gemma/Gemini Setup

After receiving a Google API key, call hosted Gemma through the native `generateContent` endpoint. Do not expose API keys in docs, Git, or chat.

```bash
export ENABLE_LLM_CONTEXT=true
export ENABLE_EXTERNAL_LLM_CALLS=true
export LLM_CONTEXT_MODE=google_generative
export LLM_API_KEY="YOUR_GOOGLE_API_KEY"
export LLM_API_BASE="https://generativelanguage.googleapis.com/v1beta"
export LLM_MODEL="gemma-3-27b-it"
export LLM_REQUEST_TIMEOUT_SECONDS=45
```

Use the actual model ID shown by Google AI Studio or the model list. Seeing “Gemma 4” in a UI does not imply the API model id is `gemma4`. Google's OpenAI-compatible examples are centered on Gemini model IDs, while hosted Gemma uses the native `models/{model}:generateContent` endpoint.

## Checking export

Check the current shell without printing the key value.

```bash
echo "$ENABLE_LLM_CONTEXT"
echo "$ENABLE_EXTERNAL_LLM_CALLS"
echo "$LLM_CONTEXT_MODE"
echo "$LLM_API_BASE"
echo "$LLM_MODEL"
test -n "$LLM_API_KEY" && echo "LLM_API_KEY is set" || echo "LLM_API_KEY is missing"
```

To check only the key length:

```bash
python - <<'PY'
import os
key = os.environ.get("LLM_API_KEY", "")
print("LLM_API_KEY set:", bool(key), "length:", len(key))
PY
```

`export` only affects the current shell and child processes started from that shell. It disappears when the terminal closes and is not shared with other terminals. For persistence, put the values in `~/.zshrc` or in a project `.env` file. The main server/script entrypoints now auto-load the project-root `.env`.

```bash
cp .env.example .env
```

Shell commands such as `echo "$LLM_MODEL"` may not show `.env` values. Check the values read by the project with Python:

```bash
.venv/bin/python - <<'PY'
from market_ai.config import get_settings
s = get_settings()
print("enable_llm_context:", s.enable_llm_context)
print("enable_external_llm_calls:", s.enable_external_llm_calls)
print("llm_model:", s.llm_model)
print("llm_api_base:", s.llm_api_base)
print("llm_api_key_set:", bool(s.llm_api_key))
PY
```

## Validation

Validate the adapter and safety checks without an external call:

```bash
.venv/bin/python scripts/llm/test_llm_context.py --mode openai_compatible --dry-run
```

Validate the live Google endpoint:

```bash
.venv/bin/python scripts/llm/test_llm_context.py --mode google_generative --live
```

Success means `safety_check_passed=true` and no `External LLM fallback` warning. `model not found`, `404`, or `unsupported response_format` usually means a model-id or endpoint compatibility issue.

## Building Event Context

Convert the current news file into LLM context:

```bash
.venv/bin/python scripts/data/build_event_context.py \
  --news-path data/raw/news/public_market_news.csv \
  --symbols CL=F \
  --mode google_generative \
  --live \
  --start 2016-11-01 \
  --end 2026-05-08 \
  --news-limit-per-context 8 \
  --llm-batch-size 1 \
  --llm-min-interval-seconds 5.0
```

`--news-limit-per-context` is the bounded number of news rows the LLM reads directly. The current historical cache was generated with the latest 8 news items from the prior 7 days for each date. This limit controls token cost and request size; to avoid a model-input bottleneck, the builder separately computes 14 aggregate features from the full raw point-in-time news pool over recent 1/3/7/30-day windows.

With an external LLM mode and `--live`, fallback rows are rejected immediately. If the cause is quota or rate limit, keep the cache, reset usage or increase the request interval, and retry only the failed dates. This prevents lower-quality non-LLM context from leaking into model training.

Outputs:

- `data/processed/event_context/event_context_daily.csv`
- `data/processed/event_context/llm_context_cache.jsonl`

The 2026-06-05 cleanup produced 3,476 CL=F event-context rows, a 27-dimensional event/context input, and zero external LLM fallback rows. If Google free-tier 429s occur, keep the cache and retry only the failed dates with `--llm-batch-size 1` and a sufficient `--llm-min-interval-seconds`.

## Training

`oil_context_fusion` is the operational model that consumes LLM context.

```bash
.venv/bin/python scripts/train/train_deep_fusion_models.py \
  --model oil_context_fusion \
  --interval 1d \
  --horizon 30 \
  --lookback 128 \
  --universe oil_core \
  --llm-context \
  --event-context data/processed/event_context/event_context_daily.csv \
  --events-path data/raw/news/public_market_news.csv \
  --use-processed-data \
  --market-panel data/processed/market_panel/1d/panel.csv \
  --oil-fundamentals data/processed/oil_fundamentals/eia_weekly.csv \
  --cot data/processed/oil_fundamentals/cftc_cot_weekly.csv \
  --macro-panel data/processed/macro_panel/fred_daily_wide.csv \
  --max-samples 0 \
  --epochs 10 \
  --patience 3 \
  --batch-size 128 \
  --device mps \
  --force
```

For the deployable final artifact, first record holdout performance separately, then add `--fit-final-all-data`.

The dashboard server must be started from a shell with the same environment variables. If the server is already running, restart it after changing LLM environment variables.
