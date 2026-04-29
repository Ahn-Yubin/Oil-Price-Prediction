# LLM Context

The LLM is an auxiliary layer for market events and explanations. Numeric price forecasting belongs to time-series models.

## Allowed Roles

- Convert news and events into structured market context.
- Summarize major drivers, uncertainty, and risk factors.
- Turn forecast responses into human-readable explanations.

## Forbidden Roles

- The LLM must not directly generate numeric forecasts such as `p50`, quantiles, or target prices.
- LLM output must not overwrite numeric forecasts from time-series models.

## Location

LLM logic lives in `market_ai/llm` and `market_ai/schemas/llm_context.py`. Prompts live in `market_ai/llm/prompts`.
