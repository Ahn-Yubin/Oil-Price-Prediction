from __future__ import annotations

import torch
from torch.nn import functional as F

from market_ai.modeling.deep.lstm_tcn_fusion import QUANTILE_LEVELS


def pinball_loss(y_true: torch.Tensor, y_quantiles: torch.Tensor, quantile_levels: tuple[float, ...] = QUANTILE_LEVELS) -> torch.Tensor:
    target = y_true.unsqueeze(-1)
    losses = []
    for idx, level in enumerate(quantile_levels):
        error = target[..., 0] - y_quantiles[..., idx]
        q = torch.as_tensor(level, dtype=y_quantiles.dtype, device=y_quantiles.device)
        losses.append(torch.maximum(q * error, (q - 1.0) * error))
    return torch.stack(losses, dim=-1).mean()


def quantile_monotonicity_penalty(y_quantiles: torch.Tensor) -> torch.Tensor:
    diffs = y_quantiles[..., :-1] - y_quantiles[..., 1:]
    return F.relu(diffs).mean()


def deep_forecast_loss(
    output: dict[str, torch.Tensor],
    *,
    y_vol_scaled_cum_return: torch.Tensor,
    y_direction: torch.Tensor,
    y_future_volatility: torch.Tensor,
) -> torch.Tensor:
    quantiles = output["quantiles"]
    median = quantiles[..., len(QUANTILE_LEVELS) // 2]
    loss_pinball = pinball_loss(y_vol_scaled_cum_return, quantiles)
    loss_median = F.huber_loss(median, y_vol_scaled_cum_return)
    loss_direction = F.binary_cross_entropy(output["prob_up"], y_direction.float())
    loss_vol = F.huber_loss(output["expected_volatility"], y_future_volatility.float())
    loss_mono = quantile_monotonicity_penalty(quantiles)
    return loss_pinball + 0.35 * loss_median + 0.15 * loss_direction + 0.10 * loss_vol + 0.05 * loss_mono
