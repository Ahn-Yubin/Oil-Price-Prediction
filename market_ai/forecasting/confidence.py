def clamp_probability(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
