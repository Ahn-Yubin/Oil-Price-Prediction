def coverage_rate(actual: list[float], lower: list[float], upper: list[float]) -> float:
    if not actual:
        return 0.0
    hits = [lo <= y <= hi for y, lo, hi in zip(actual, lower, upper)]
    return sum(hits) / len(hits)
