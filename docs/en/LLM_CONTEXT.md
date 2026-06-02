# LLM Context

The LLM is not a numeric price forecaster in this project. It is a market news and event context encoder. Deep models can use the LLM-derived context vector as one input feature, but the LLM must not create `p50`, `p90`, target prices, or future price paths.

## Current Flow

The intended flow is:

```text
news/event text
-> LLM or local_rules encoder
-> bias, impact, uncertainty, event count, explanation, event embedding
-> data/processed/event_context/event_context_daily.csv
-> x_event_context input for oil_context_fusion
-> time-series model predicts volatility-scaled cumulative log return distribution
-> prices are reconstructed with current_price * exp(predicted_cumulative_log_return_h)
```

So the user-facing summary is correct: news is read by the LLM, converted into scores/context, and used as one model input. Those scores are point-in-time context features, not investment advice or numeric price targets.

Training samples do not use only the origin date. With the current `lookback=128` setup, each origin aggregates event context from the prior 128-day window into `x_event_context [13]`. Event-count fields are summed, while directional/impact/quality fields are impact-score weighted averages.

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
   If external LLM calls are enabled and the rate guard allows the call, the LLM encoder is used. Otherwise `local_rules` produces the same schema deterministically. The output schema is `overall_bias`, `impact_score`, `uncertainty`, `event_count`, `explanation`, and `event_embedding[13]`.

7. The encoder output becomes a one-row `context_frame` for deep-model inference.
   `oil_context_fusion` uses this `x_event_context` for context expert gating, uncertainty, and confidence. The LLM does not draw the forecast line directly; the news context vector becomes an input feature and the time-series model computes the forecast path.

8. `/api/market-context?live=1` returns the same live context payload to the UI.
   Chart news markers, popovers, and the right-side “News & Scenario Context” panel use the same `news` and `context_points`. Each visible news row shows the current context point’s input factors: bias, impact, uncertainty, and event count.

9. `/api/model-commentary` and `/api/assistant-chat` explain already-computed model paths and news context.
   They do not create new numeric price forecasts. To avoid rate-limit spikes, the server enforces an external LLM call guard over a 60-second window and uses commentary/context caches. If the guard blocks a call, deterministic explanation is returned with a warning.

## Allowed Roles

- Convert news, economic events, and supply events into structured market context.
- Produce `overall_bias`, `impact_score`, `uncertainty`, and event embeddings.
- Show historical news and the context interpretation on the dashboard.
- Affect gating, confidence, and uncertainty indirectly inside `oil_context_fusion`.
- Provide historical context markers and scenario commentary through `/api/market-context`.

## Forbidden Roles

- The LLM must not create `p50`, `p90`, target prices, or future return paths.
- LLM output must not overwrite numeric forecasts from deep models or baselines.
- Trading instructions or fixed target prices from the LLM must be ignored by validation.
- Forecast bands must not be called validated confidence intervals until coverage has actually been measured.

## Implementations

- `LocalEventContextEncoder`: reads CSV/JSON event files with deterministic rules.
- `OpenAICompatibleLLMEventEncoder`: calls an OpenAI-compatible chat completions endpoint.
- `LocalHTTPLLMEventEncoder`: calls an Ollama/vLLM-compatible local HTTP endpoint.
- `OfflineFileLLMEventEncoder`: reads a precomputed JSON/JSONL context cache.
- `NullLLMEventEncoder`: disables LLM context completely.

External calls require both `--live` and `ENABLE_EXTERNAL_LLM_CALLS=true`. Missing keys or call failures fall back to local encoders and leave the reason in warnings.

## Google Gemma/Gemini Setup

After receiving a Google API key, call hosted Gemma through the native `generateContent` endpoint. Do not expose API keys in docs, Git, or chat.

```bash
export ENABLE_LLM_CONTEXT=true
export ENABLE_EXTERNAL_LLM_CALLS=true
export LLM_CONTEXT_MODE=google_generative
export LLM_API_KEY="YOUR_GOOGLE_API_KEY"
export LLM_API_BASE="https://generativelanguage.googleapis.com/v1beta"
export LLM_MODEL="gemma-3-27b-it"
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
  --symbols CL=F,BZ=F,NG=F,RB=F,HO=F,GC=F,SI=F,HG=F,DX-Y.NYB,EURUSD=X,USDKRW=X,JPY=X,SPY,QQQ,^GSPC,^VIX,XLE,USO \
  --mode google_generative \
  --live
```

Outputs:

- `data/processed/event_context/event_context_daily.csv`
- `data/processed/event_context/llm_context_cache.jsonl`

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

The dashboard server must be started from a shell with the same environment variables. If the server is already running, restart it after changing LLM environment variables.
