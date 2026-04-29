def direction_hit(actual_return: float, expected_return: float) -> bool:
    return (actual_return >= 0.0) == (expected_return >= 0.0)
