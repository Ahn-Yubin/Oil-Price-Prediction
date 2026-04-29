def conformal_interval_width(residuals: list[float], alpha: float = 0.1) -> float:
    if not residuals:
        return 0.0
    ordered = sorted(abs(float(value)) for value in residuals)
    index = min(len(ordered) - 1, max(0, int((1.0 - alpha) * len(ordered))))
    return ordered[index]
