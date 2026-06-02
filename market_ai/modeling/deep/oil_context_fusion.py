from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from market_ai.modeling.deep.lstm_tcn_fusion import QUANTILE_LEVELS
from market_ai.modeling.deep.modules import MLP, TCNEncoder, enforce_quantile_monotonicity


class OilContextFusion(nn.Module):
    """Unified oil forecaster combining sequence, context, pattern, and motif experts."""

    def __init__(
        self,
        *,
        price_feature_dim: int,
        cross_asset_dim: int,
        event_context_dim: int,
        static_dim: int,
        horizon: int,
        hidden_dim: int = 72,
        dropout: float = 0.12,
        expert_names: list[str] | tuple[str, ...] | None = None,
    ):
        del expert_names
        super().__init__()
        self.price_feature_dim = int(price_feature_dim)
        self.cross_asset_dim = int(cross_asset_dim)
        self.event_context_dim = int(event_context_dim)
        self.static_dim = int(static_dim)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.expert_names = ("lstm", "tcn", "attention", "context", "pattern", "motif")

        self.price_projection = nn.Linear(self.price_feature_dim, hidden_dim)
        self.cross_projection = nn.Linear(max(self.cross_asset_dim, 1), hidden_dim)
        sequence_dim = hidden_dim * 2
        self.sequence_norm = nn.LayerNorm(sequence_dim)
        self.lstm = nn.LSTM(sequence_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.tcn = TCNEncoder(sequence_dim, hidden_dim, dropout=dropout)
        self.attention_input = nn.Linear(sequence_dim, hidden_dim)
        attention_heads = 4 if hidden_dim % 4 == 0 else 3 if hidden_dim % 3 == 0 else 1
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        context_dim = self.event_context_dim + self.static_dim
        self.context_encoder = MLP(context_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.pattern_encoder = MLP(sequence_dim * 4 + context_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.motif_projection = MLP(sequence_dim, hidden_dim, hidden_dim, dropout=dropout)

        q_dim = self.horizon * len(QUANTILE_LEVELS)
        expert_input_dim = hidden_dim * len(self.expert_names)
        self.lstm_head = MLP(hidden_dim, hidden_dim, q_dim, dropout=dropout)
        self.tcn_head = MLP(hidden_dim, hidden_dim, q_dim, dropout=dropout)
        self.attention_head = MLP(hidden_dim, hidden_dim, q_dim, dropout=dropout)
        self.context_head = MLP(expert_input_dim, hidden_dim, q_dim, dropout=dropout)
        self.pattern_head = MLP(hidden_dim, hidden_dim, q_dim, dropout=dropout)
        self.motif_head = MLP(hidden_dim, hidden_dim, q_dim, dropout=dropout)
        self.gating_network = MLP(expert_input_dim + context_dim, hidden_dim, len(self.expert_names), dropout=dropout)

        shared_dim = hidden_dim * 5 + context_dim
        self.direction_head = MLP(shared_dim, hidden_dim, self.horizon, dropout=dropout)
        self.volatility_head = MLP(shared_dim, hidden_dim, self.horizon, dropout=dropout)
        self.confidence_head = MLP(hidden_dim + context_dim, hidden_dim, 1, dropout=dropout)

    def config_dict(self) -> dict[str, int | float | list[str]]:
        return {
            "price_feature_dim": self.price_feature_dim,
            "cross_asset_dim": self.cross_asset_dim,
            "event_context_dim": self.event_context_dim,
            "static_dim": self.static_dim,
            "horizon": self.horizon,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "expert_names": list(self.expert_names),
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
        seq = self.sequence_norm(torch.cat([price_repr, cross_repr], dim=-1))

        lstm_seq, _ = self.lstm(seq)
        lstm_repr = lstm_seq[:, -1, :]
        tcn_repr = self.tcn(seq)
        attention_seq = self.attention_input(seq)
        attention_out, _ = self.attention(attention_seq, attention_seq, attention_seq, need_weights=False)
        attention_repr = attention_out[:, -1, :]

        context = torch.cat([x_event_context, x_static], dim=-1)
        context_repr = self.context_encoder(context)
        seq_last = seq[:, -1, :]
        seq_mean = seq.mean(dim=1)
        seq_std = seq.std(dim=1, unbiased=False)
        seq_delta = seq_last - seq[:, 0, :]
        pattern_repr = self.pattern_encoder(torch.cat([seq_last, seq_mean, seq_std, seq_delta, context], dim=-1))

        motif_len = max(1, min(16, lookback // 4 if lookback >= 4 else 1))
        if lookback > motif_len:
            history = seq[:, :-motif_len, :]
            recent = seq[:, -motif_len:, :].mean(dim=1)
        else:
            history = seq
            recent = seq[:, -1, :]
        similarity = F.cosine_similarity(history, recent.unsqueeze(1), dim=-1)
        motif_weights = torch.softmax(similarity, dim=-1)
        motif_raw = torch.sum(history * motif_weights.unsqueeze(-1), dim=1)
        motif_repr = self.motif_projection(motif_raw)

        expert_input = torch.cat([lstm_repr, tcn_repr, attention_repr, context_repr, pattern_repr, motif_repr], dim=-1)
        gate_input = torch.cat([expert_input, context], dim=-1)

        q_dim = len(QUANTILE_LEVELS)
        experts = torch.stack(
            [
                self.lstm_head(lstm_repr).view(batch, self.horizon, q_dim),
                self.tcn_head(tcn_repr).view(batch, self.horizon, q_dim),
                self.attention_head(attention_repr).view(batch, self.horizon, q_dim),
                self.context_head(expert_input).view(batch, self.horizon, q_dim),
                self.pattern_head(pattern_repr).view(batch, self.horizon, q_dim),
                self.motif_head(motif_repr).view(batch, self.horizon, q_dim),
            ],
            dim=1,
        )
        expert_weights = torch.softmax(self.gating_network(gate_input), dim=-1)
        quantiles = torch.sum(experts * expert_weights[:, :, None, None], dim=1)
        quantiles = enforce_quantile_monotonicity(quantiles)

        shared = torch.cat([lstm_repr, tcn_repr, attention_repr, pattern_repr, motif_repr, context], dim=-1)
        prob_up = torch.sigmoid(self.direction_head(shared))
        expected_volatility = F.softplus(self.volatility_head(shared)) + 1e-6
        confidence = torch.sigmoid(self.confidence_head(torch.cat([context_repr, context], dim=-1)))
        event_uncertainty = x_event_context[:, 2:3] if self.event_context_dim >= 3 else 0.0
        confidence = torch.clamp(confidence * (1.0 - 0.30 * event_uncertainty), 0.0, 1.0)

        return {
            "quantiles": quantiles,
            "prob_up": prob_up,
            "expected_volatility": expected_volatility,
            "confidence": confidence,
            "expert_weights": expert_weights,
            "expert_names": self.expert_names,
        }
