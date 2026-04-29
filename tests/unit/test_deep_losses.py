import torch

from market_ai.modeling.deep.losses import deep_forecast_loss, pinball_loss, quantile_monotonicity_penalty


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
