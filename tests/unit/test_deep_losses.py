import torch

from market_ai.modeling.deep.losses import (
    deep_forecast_loss,
    gaussian_tail_path_loss,
    path_range_loss,
    path_shape_loss,
    pinball_loss,
    range_shortfall_tail_loss,
    quantile_monotonicity_penalty,
)


def test_deep_losses_are_finite():
    y = torch.randn(4, 6)
    q = torch.sort(torch.randn(4, 6, 7), dim=-1).values
    output = {
        "quantiles": q,
        "prob_up": torch.sigmoid(torch.randn(4, 6)),
        "expected_volatility": torch.rand(4, 6) + 1e-3,
    }
    loss = deep_forecast_loss(
        output,
        y_vol_scaled_cum_return=y,
        y_direction=(y > 0).float(),
        y_future_volatility=torch.rand(4, 6),
    )
    assert torch.isfinite(loss)
    assert pinball_loss(y, q) >= 0
    assert quantile_monotonicity_penalty(q) == 0


def test_path_shape_loss_rewards_matching_detrended_shape():
    y = torch.tensor([[0.1, 0.0, 0.2, 0.1, 0.3]], dtype=torch.float32)
    same = y.clone()
    flat = torch.linspace(0.0, float(y[0, -1]), y.shape[1]).reshape(1, -1)

    assert path_shape_loss(y, same) < path_shape_loss(y, flat)


def test_path_range_loss_penalizes_flat_path_when_actual_has_range():
    actual = torch.tensor([[0.0, 1.0, -0.5, 0.8, 0.2]], dtype=torch.float32)
    flat = torch.zeros_like(actual)
    matching = actual.clone()

    assert path_range_loss(actual, matching) < path_range_loss(actual, flat)


def test_tail_losses_heavily_penalize_flat_path_on_extreme_move():
    actual = torch.tensor([[0.0, 2.0, 5.0, 8.0, 11.0, 13.0]], dtype=torch.float32)
    matching = actual.clone()
    flat = torch.zeros_like(actual)

    assert gaussian_tail_path_loss(actual, flat) > gaussian_tail_path_loss(actual, matching) + 10.0
    assert range_shortfall_tail_loss(actual, flat) > range_shortfall_tail_loss(actual, matching) + 10.0
