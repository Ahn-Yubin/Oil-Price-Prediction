def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)
