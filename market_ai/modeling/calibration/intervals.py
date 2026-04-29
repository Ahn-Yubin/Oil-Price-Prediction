def symmetric_interval(center: float, width: float) -> tuple[float, float]:
    return float(center) - abs(float(width)), float(center) + abs(float(width))
