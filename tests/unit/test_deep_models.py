import torch

from market_ai.modeling.deep.lstm_tcn_fusion import DeepLstmTcnFusion
from market_ai.modeling.deep.llm_seq_moe import LLMContextSeqMoE
from market_ai.modeling.deep.oil_context_fusion import OilContextFusion


def _inputs(batch=3, lookback=12, price_dim=6, cross_dim=4, event_dim=13, static_dim=4):
    return (
        torch.randn(batch, lookback, price_dim),
        torch.randn(batch, lookback, cross_dim),
        torch.zeros(batch, event_dim),
        torch.randn(batch, static_dim),
    )


def _assert_output(output, horizon):
    assert output["quantiles"].shape == (3, horizon, 7)
    assert output["prob_up"].shape == (3, horizon)
    assert output["expected_volatility"].shape == (3, horizon)
    assert torch.isfinite(output["quantiles"]).all()
    assert torch.all(output["quantiles"][..., :-1] <= output["quantiles"][..., 1:])


def test_deep_lstm_tcn_fusion_forward_shapes_cpu():
    x_price, x_cross, x_event, x_static = _inputs()
    model = DeepLstmTcnFusion(price_feature_dim=6, cross_asset_dim=4, event_context_dim=13, static_dim=4, horizon=5)
    _assert_output(model(x_price, x_cross, x_event, x_static), 5)


def test_oil_context_fusion_forward_shapes_cpu():
    x_price, x_cross, x_event, x_static = _inputs()
    model = OilContextFusion(price_feature_dim=6, cross_asset_dim=4, event_context_dim=13, static_dim=4, horizon=5)
    output = model(x_price, x_cross, x_event, x_static)
    _assert_output(output, 5)
    assert output["expert_weights"].shape == (3, 6)
    assert torch.allclose(output["expert_weights"].sum(dim=-1), torch.ones(3), atol=1e-5)
    assert output["expert_names"] == ("lstm", "tcn", "attention", "context", "pattern", "motif")


def test_llm_context_seq_moe_forward_zero_and_nonzero_context():
    x_price, x_cross, x_event, x_static = _inputs()
    model = LLMContextSeqMoE(price_feature_dim=6, cross_asset_dim=4, event_context_dim=13, static_dim=4, horizon=5)
    zero = model(x_price, x_cross, x_event, x_static)
    _assert_output(zero, 5)
    x_event[:, 0] = 1.0
    nonzero = model(x_price, x_cross, x_event, x_static)
    _assert_output(nonzero, 5)
    assert nonzero["expert_weights"].shape == (3, 3)
    assert torch.allclose(nonzero["expert_weights"].sum(dim=-1), torch.ones(3), atol=1e-5)
    assert not hasattr(model, "baseline_adapter")
    assert not hasattr(model, "motif_adapter")
    assert hasattr(model, "context_head")
