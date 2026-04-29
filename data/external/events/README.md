# Event Data Schema

Sample event files are deterministic inputs for local context encoding. External APIs are not required.

Required fields:
- `timestamp`: UTC event timestamp. Only events at or before `as_of_time` are used.
- `symbol`: target symbol, `ALL`, or `*`.
- `event_type`: macro, energy, geopolitical, supply, demand, policy, or other category.
- `directional_bias`: `bullish`, `bearish`, `neutral`, `mixed`, or `unknown`.
- `impact_strength`: 0 to 1 event impact score.
- `uncertainty`: 0 to 1 uncertainty score.
- `source_quality_score`: 0 to 1 source quality score.
- `summary`: short context summary.
- `source`: optional source label.
