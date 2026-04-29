import numpy as np
import pandas as pd

from market_ai.schemas.market import AssetClass
from market_ai.data.related_assets import build_cross_asset_features, cross_asset_context_summary, get_related_assets


def _frame(scale: float = 1.0) -> pd.DataFrame:
    x = np.arange(60, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC"),
            "close": scale * (100.0 + 0.2 * x + np.sin(x / 4.0)),
        }
    )


def test_related_assets_mapping_for_oil_and_crypto():
    assert "BZ=F" in get_related_assets("CL=F", AssetClass.futures)
    assert "ETH-USD" in get_related_assets("BTC-USD", AssetClass.crypto)
    assert "^GSPC" in get_related_assets("MSFT", AssetClass.equity)


def test_cross_asset_context_disabled_is_graceful():
    summary = cross_asset_context_summary("CL=F", AssetClass.futures, enabled=False)
    assert summary["enabled"] is False
    assert summary["related_assets"]
    assert summary["warning"]


def test_cross_asset_feature_alignment_no_lookahead():
    target = _frame(1.0)
    related = {"BZ=F": _frame(1.2)}
    baseline = build_cross_asset_features(target, related).iloc[:30].reset_index(drop=True)
    modified = related["BZ=F"].copy()
    modified.loc[45:, "close"] *= 5.0
    changed = build_cross_asset_features(target, {"BZ=F": modified}).iloc[:30].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline, changed)
