def pinball_loss(y_true: float, y_pred: float, quantile: float) -> float:
    diff = y_true - y_pred
    return max(quantile * diff, (quantile - 1.0) * diff)
