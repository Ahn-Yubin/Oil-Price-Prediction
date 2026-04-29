from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from market_ai.modeling.deep.lstm_tcn_fusion import QUANTILE_LEVELS
from market_ai.modeling.deep.modules import MLP, TCNEncoder, enforce_quantile_monotonicity


class LLMContextSeqMoE(nn.Module):
    def __init__(
        self,
        *,
        price_feature_dim: int,
        cross_asset_dim: int,
        event_context_dim: int,
        static_dim: int,
        horizon: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.price_feature_dim = int(price_feature_dim)
        self.cross_asset_dim = int(cross_asset_dim)
        self.event_context_dim = int(event_context_dim)
        self.static_dim = int(static_dim)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.price_projection = nn.Linear(self.price_feature_dim, hidden_dim)
        self.cross_projection = nn.Linear(max(self.cross_asset_dim, 1), hidden_dim)
        sequence_dim = hidden_dim * 2
        self.lstm = nn.LSTM(sequence_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.tcn = TCNEncoder(sequence_dim, hidden_dim, dropout=dropout)
        q_dim = self.horizon * len(QUANTILE_LEVELS)
        self.lstm_head = MLP(hidden_dim, hidden_dim, q_dim, dropout=dropout)
        self.tcn_head = MLP(hidden_dim, hidden_dim, q_dim, dropout=dropout)
        self.baseline_adapter = MLP(self.static_dim, hidden_dim, q_dim, dropout=dropout)
        self.motif_adapter = MLP(self.static_dim, hidden_dim, q_dim, dropout=dropout)
        gate_dim = self.event_context_dim + self.static_dim + hidden_dim * 2
        self.gating_network = MLP(gate_dim, hidden_dim, 4, dropout=dropout)
        self.direction_head = MLP(hidden_dim * 2 + self.static_dim, hidden_dim, self.horizon, dropout=dropout)
        self.volatility_head = MLP(hidden_dim * 2 + self.static_dim, hidden_dim, self.horizon, dropout=dropout)
        self.confidence_head = MLP(self.event_context_dim + self.static_dim, hidden_dim, 1, dropout=dropout)

    def config_dict(self) -> dict[str, int | float]:
        return {
            "price_feature_dim": self.price_feature_dim,
            "cross_asset_dim": self.cross_asset_dim,
            "event_context_dim": self.event_context_dim,
            "static_dim": self.static_dim,
            "horizon": self.horizon,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }

    def forward(
        self,
        x_price: torch.Tensor,
        x_cross_asset: torch.Tensor | None = None,
        x_event_context: torch.Tensor | None = None,
        x_static: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, lookback, _ = x_price.shape
        if x_cross_asset is None or x_cross_asset.numel() == 0:
            x_cross_asset = x_price.new_zeros((batch, lookback, max(self.cross_asset_dim, 1)))
        if x_cross_asset.shape[-1] == 0:
            x_cross_asset = x_price.new_zeros((batch, lookback, 1))
        if x_event_context is None:
            x_event_context = x_price.new_zeros((batch, self.event_context_dim))
        if x_static is None:
            x_static = x_price.new_zeros((batch, self.static_dim))

        price_repr = self.price_projection(x_price)
        cross_repr = self.cross_projection(x_cross_asset)
        seq = torch.cat([price_repr, cross_repr], dim=-1)
        lstm_seq, _ = self.lstm(seq)
        lstm_repr = lstm_seq[:, -1, :]
        tcn_repr = self.tcn(seq)
        q_dim = len(QUANTILE_LEVELS)
        experts = torch.stack(
            [
                self.lstm_head(lstm_repr).view(batch, self.horizon, q_dim),
                self.tcn_head(tcn_repr).view(batch, self.horizon, q_dim),
                self.baseline_adapter(x_static).view(batch, self.horizon, q_dim),
                self.motif_adapter(x_static).view(batch, self.horizon, q_dim),
            ],
            dim=1,
        )
        gate_input = torch.cat([x_event_context, x_static, lstm_repr, tcn_repr], dim=-1)
        expert_weights = torch.softmax(self.gating_network(gate_input), dim=-1)
        quantiles = torch.sum(experts * expert_weights[:, :, None, None], dim=1)
        quantiles = enforce_quantile_monotonicity(quantiles)
        sequence_repr = torch.cat([lstm_repr, tcn_repr, x_static], dim=-1)
        prob_up = torch.sigmoid(self.direction_head(sequence_repr))
        expected_volatility = F.softplus(self.volatility_head(sequence_repr)) + 1e-6
        confidence = torch.sigmoid(self.confidence_head(torch.cat([x_event_context, x_static], dim=-1)))
        event_uncertainty = x_event_context[:, 2:3] if self.event_context_dim >= 3 else 0.0
        confidence = torch.clamp(confidence * (1.0 - 0.35 * event_uncertainty), 0.0, 1.0)
        return {
            "quantiles": quantiles,
            "prob_up": prob_up,
            "expected_volatility": expected_volatility,
            "confidence": confidence,
            "expert_weights": expert_weights,
        }
