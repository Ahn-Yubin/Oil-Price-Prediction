from __future__ import annotations

import torch
from torch.nn import functional as F

from market_ai.modeling.deep.lstm_tcn_fusion import QUANTILE_LEVELS


def _weighted_mean(loss: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    if sample_weight is None:
        return loss.mean()
    weight = sample_weight
    while weight.ndim < loss.ndim:
        weight = weight.unsqueeze(-1)
    return (loss * weight).sum() / torch.clamp(weight.sum() * (loss.numel() / max(weight.numel(), 1)), min=1e-8)


def pinball_loss(
    y_true: torch.Tensor,
    y_quantiles: torch.Tensor,
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    target = y_true.unsqueeze(-1)
    losses = []
    for idx, level in enumerate(quantile_levels):
        error = target[..., 0] - y_quantiles[..., idx]
        q = torch.as_tensor(level, dtype=y_quantiles.dtype, device=y_quantiles.device)
        losses.append(torch.maximum(q * error, (q - 1.0) * error))
    return _weighted_mean(torch.stack(losses, dim=-1), sample_weight)


def quantile_monotonicity_penalty(y_quantiles: torch.Tensor) -> torch.Tensor:
    diffs = y_quantiles[..., :-1] - y_quantiles[..., 1:]
    return F.relu(diffs).mean()


def step_return_loss(
    y_true_cumulative: torch.Tensor,
    y_pred_cumulative: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    true_steps = cumulative_to_steps(y_true_cumulative)
    pred_steps = cumulative_to_steps(y_pred_cumulative)
    return _weighted_mean(F.huber_loss(pred_steps, true_steps, reduction="none"), sample_weight)


def cumulative_to_steps(y_cumulative: torch.Tensor) -> torch.Tensor:
    return torch.diff(torch.cat([torch.zeros_like(y_cumulative[:, :1]), y_cumulative], dim=1), dim=1)


def path_shape_loss(
    y_true_cumulative: torch.Tensor,
    y_pred_cumulative: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if y_true_cumulative.shape[1] < 3:
        return y_pred_cumulative.new_tensor(0.0)
    horizon = y_true_cumulative.shape[1]
    ramp = torch.linspace(
        1.0 / float(horizon),
        1.0,
        horizon,
        dtype=y_true_cumulative.dtype,
        device=y_true_cumulative.device,
    ).unsqueeze(0)
    true_residual = y_true_cumulative - ramp * y_true_cumulative[:, -1:]
    pred_residual = y_pred_cumulative - ramp * y_pred_cumulative[:, -1:]
    true_range = torch.amax(true_residual, dim=1) - torch.amin(true_residual, dim=1)
    scale = torch.clamp(true_range, min=1.0).unsqueeze(-1)
    return _weighted_mean(F.huber_loss(pred_residual / scale, true_residual / scale, reduction="none"), sample_weight)


def path_range_loss(
    y_true_cumulative: torch.Tensor,
    y_pred_cumulative: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    true_range = torch.amax(y_true_cumulative, dim=1) - torch.amin(y_true_cumulative, dim=1)
    pred_range = torch.amax(y_pred_cumulative, dim=1) - torch.amin(y_pred_cumulative, dim=1)
    return _weighted_mean(F.huber_loss(torch.log(pred_range + 0.25), torch.log(true_range + 0.25), reduction="none"), sample_weight)


def step_volatility_loss(
    y_true_cumulative: torch.Tensor,
    y_pred_cumulative: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    true_steps = cumulative_to_steps(y_true_cumulative)
    pred_steps = cumulative_to_steps(y_pred_cumulative)
    true_scale = torch.std(true_steps, dim=1, unbiased=False)
    pred_scale = torch.std(pred_steps, dim=1, unbiased=False)
    return _weighted_mean(F.huber_loss(torch.log(pred_scale + 0.10), torch.log(true_scale + 0.10), reduction="none"), sample_weight)


def curvature_match_loss(
    y_true_cumulative: torch.Tensor,
    y_pred_cumulative: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if y_true_cumulative.shape[1] < 3:
        return y_pred_cumulative.new_tensor(0.0)
    true_second = y_true_cumulative[:, 2:] - 2.0 * y_true_cumulative[:, 1:-1] + y_true_cumulative[:, :-2]
    pred_second = y_pred_cumulative[:, 2:] - 2.0 * y_pred_cumulative[:, 1:-1] + y_pred_cumulative[:, :-2]
    return _weighted_mean(F.huber_loss(pred_second, true_second, reduction="none"), sample_weight)


def gaussian_tail_path_loss(
    y_true_cumulative: torch.Tensor,
    y_pred_cumulative: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    true_steps = cumulative_to_steps(y_true_cumulative)
    scale = torch.clamp(torch.std(true_steps, dim=1, unbiased=False), min=0.50).unsqueeze(-1)
    z = torch.abs(y_pred_cumulative - y_true_cumulative) / scale
    gaussian_nll = 0.5 * torch.square(z)
    beyond_two_sigma = F.relu(z - 2.0)
    tail_penalty = torch.square(beyond_two_sigma) * (1.0 + 0.50 * torch.clamp(beyond_two_sigma, max=4.0))
    return _weighted_mean(gaussian_nll + tail_penalty, sample_weight)


def range_shortfall_tail_loss(
    y_true_cumulative: torch.Tensor,
    y_pred_cumulative: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    true_range = torch.amax(y_true_cumulative, dim=1) - torch.amin(y_true_cumulative, dim=1)
    pred_range = torch.amax(y_pred_cumulative, dim=1) - torch.amin(y_pred_cumulative, dim=1)
    shortfall = F.relu(true_range - pred_range)
    tail = torch.expm1(torch.clamp(shortfall / 3.0, min=0.0, max=4.0))
    return _weighted_mean(torch.square(tail), sample_weight)


def step_direction_loss(
    y_true_cumulative: torch.Tensor,
    y_pred_cumulative: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    true_steps = cumulative_to_steps(y_true_cumulative)
    pred_steps = cumulative_to_steps(y_pred_cumulative)
    labels = (true_steps > 0.0).float()
    logits = pred_steps / 0.75
    return _weighted_mean(F.binary_cross_entropy_with_logits(logits, labels, reduction="none"), sample_weight)


def deep_forecast_loss(
    output: dict[str, torch.Tensor],
    *,
    y_vol_scaled_cum_return: torch.Tensor,
    y_direction: torch.Tensor,
    y_future_volatility: torch.Tensor,
) -> torch.Tensor:
    quantiles = output["quantiles"]
    median = quantiles[..., len(QUANTILE_LEVELS) // 2]
    path_magnitude = torch.amax(torch.abs(y_vol_scaled_cum_return), dim=1)
    terminal_magnitude = torch.abs(y_vol_scaled_cum_return[:, -1])
    sample_weight = (
        1.0
        + 0.08 * torch.clamp(path_magnitude - 6.0, min=0.0, max=24.0)
        + 0.04 * torch.clamp(terminal_magnitude - 6.0, min=0.0, max=24.0)
    ).detach().clamp(1.0, 3.0)
    loss_pinball = pinball_loss(y_vol_scaled_cum_return, quantiles, sample_weight=sample_weight)
    loss_median = _weighted_mean(F.huber_loss(median, y_vol_scaled_cum_return, reduction="none"), sample_weight)
    loss_steps = step_return_loss(y_vol_scaled_cum_return, median, sample_weight=sample_weight)
    loss_terminal = _weighted_mean(F.huber_loss(median[:, -1], y_vol_scaled_cum_return[:, -1], reduction="none"), sample_weight)
    loss_shape = path_shape_loss(y_vol_scaled_cum_return, median, sample_weight=sample_weight)
    loss_range = path_range_loss(y_vol_scaled_cum_return, median, sample_weight=sample_weight)
    loss_step_volatility = step_volatility_loss(y_vol_scaled_cum_return, median, sample_weight=sample_weight)
    loss_curvature = curvature_match_loss(y_vol_scaled_cum_return, median, sample_weight=sample_weight)
    loss_gaussian_tail = gaussian_tail_path_loss(y_vol_scaled_cum_return, median, sample_weight=sample_weight)
    loss_range_tail = range_shortfall_tail_loss(y_vol_scaled_cum_return, median, sample_weight=sample_weight)
    loss_step_direction = step_direction_loss(y_vol_scaled_cum_return, median, sample_weight=sample_weight)
    loss_direction = _weighted_mean(F.binary_cross_entropy(output["prob_up"], y_direction.float(), reduction="none"), sample_weight)
    loss_vol = _weighted_mean(F.huber_loss(output["expected_volatility"], y_future_volatility.float(), reduction="none"), sample_weight)
    true_path_range = torch.amax(y_vol_scaled_cum_return, dim=1) - torch.amin(y_vol_scaled_cum_return, dim=1)
    if "expected_path_range" in output:
        expected_path_range = output["expected_path_range"].reshape(-1)
        loss_aux_range = _weighted_mean(
            F.huber_loss(torch.log(expected_path_range + 0.10), torch.log(true_path_range + 0.10), reduction="none"),
            sample_weight,
        )
    else:
        loss_aux_range = median.new_tensor(0.0)
    if "shock_probability" in output:
        shock_target = ((true_path_range > 8.0) | (torch.abs(y_vol_scaled_cum_return[:, -1]) > 8.0)).float()
        loss_shock = _weighted_mean(
            F.binary_cross_entropy(output["shock_probability"].reshape(-1), shock_target, reduction="none"),
            sample_weight,
        )
    else:
        loss_shock = median.new_tensor(0.0)
    loss_mono = quantile_monotonicity_penalty(quantiles)
    return (
        loss_pinball
        + 0.30 * loss_median
        + 0.55 * loss_steps
        + 0.10 * loss_terminal
        + 0.30 * loss_shape
        + 0.35 * loss_range
        + 0.18 * loss_step_volatility
        + 0.16 * loss_curvature
        + 0.08 * loss_gaussian_tail
        + 0.06 * loss_range_tail
        + 0.12 * loss_step_direction
        + 0.08 * loss_direction
        + 0.07 * loss_vol
        + 0.12 * loss_aux_range
        + 0.10 * loss_shock
        + 0.05 * loss_mono
    )
