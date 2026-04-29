# LLM Context

The LLM is an auxiliary layer for market events and explanations. Numeric price forecasting is handled by time-series models and baselines.

## Allowed Roles

- Convert news, economic events, and market events into structured context.
- Summarize directional bias, impact strength, uncertainty, and source quality.
- Turn forecast responses into human-readable explanations.
- Affect only gating and confidence/uncertainty inside `llm_context_seq_moe`.

## Forbidden Roles

- The LLM must not create `p50`, `p90`, target prices, or direct future return paths.
- LLM output must not overwrite numeric forecasts from time-series models.
- If the LLM emits target prices or trading instructions, the validator ignores those outputs.

## Implementation

- `LocalEventContextEncoder`: deterministically reads CSV/JSON event files.
- `OpenAICompatibleLLMEventEncoder`: attempts external calls only when `ENABLE_EXTERNAL_LLM_CALLS=true` and an API key exists.
- Missing keys or call failures fall back to local/null encoders.
- Prompts live in `market_ai/llm/prompts`.

## Event Input

The event file schema is documented in `data/external/events/README.md`. Files can be wired through `NEWS_EVENTS_PATH`, `ECONOMIC_EVENTS_PATH`, and `MARKET_EVENTS_PATH`. Only events with timestamps at or before forecast `as_of_time` are used.
