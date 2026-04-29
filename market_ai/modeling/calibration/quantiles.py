def ensure_monotonic_quantiles(values: list[float]) -> list[float]:
    return sorted(float(value) for value in values)
