# LLM Context

The LLM is not a numeric price forecaster in this project. It is a market news and event context encoder. Deep models can use the LLM-derived context vector as one input feature, but the LLM must not create `p50`, `p90`, target prices, or future price paths.

## Current Flow

The intended flow is:

```text
news/event text
-> LLM or local_rules encoder
-> bias, impact, uncertainty, event count, explanation, event embedding
-> data/processed/event_context/event_context_daily.csv
-> x_event_context input for llm_context_seq_moe
-> time-series model predicts volatility-scaled cumulative log return distribution
-> prices are reconstructed with current_price * exp(predicted_cumulative_log_return_h)
```

So the user-facing summary is correct: news is read by the LLM, converted into scores/context, and used as one model input. Those scores are point-in-time context features, not investment advice or numeric price targets.

## Allowed Roles

- Convert news, economic events, and supply events into structured market context.
- Produce `overall_bias`, `impact_score`, `uncertainty`, and event embeddings.
- Show historical news and the context interpretation on the dashboard.
- Affect gating, confidence, and uncertainty indirectly inside `llm_context_seq_moe`.
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

`llm_context_seq_moe` is the model that consumes LLM context.

```bash
.venv/bin/python scripts/train/train_deep_fusion_models.py \
  --model llm_context_seq_moe \
  --interval 1d \
  --horizon 8 \
  --lookback 128 \
  --universe research_core \
  --use-processed-data \
  --market-panel data/processed/market_panel/1d/panel.csv \
  --oil-fundamentals data/processed/oil_fundamentals/eia_weekly.csv \
  --cot data/processed/oil_fundamentals/cftc_cot_weekly.csv \
  --event-context data/processed/event_context/event_context_daily.csv \
  --max-samples 512 \
  --epochs 3 \
  --batch-size 64 \
  --device mps \
  --force
```

The dashboard server must be started from a shell with the same environment variables. If the server is already running, restart it after changing LLM environment variables.
