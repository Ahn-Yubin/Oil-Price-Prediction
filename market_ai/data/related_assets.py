from __future__ import annotations

import numpy as np
import pandas as pd

from market_ai.schemas.market import AssetClass
from market_ai.data.symbols import infer_asset_class, normalize_symbol


DEFAULT_RELATED_ASSETS: dict[str, list[str]] = {
    "CL=F": ["BZ=F", "DX-Y.NYB", "GC=F", "NG=F", "^GSPC", "^VIX", "XLE"],
    "BZ=F": ["CL=F", "DX-Y.NYB", "GC=F", "NG=F", "^GSPC", "^VIX", "XLE"],
    "BTC-USD": ["ETH-USD", "^IXIC", "GC=F", "DX-Y.NYB", "^VIX"],
}

DEFAULT_BY_ASSET_CLASS: dict[AssetClass, list[str]] = {
    AssetClass.equity: ["^GSPC", "^IXIC", "^VIX"],
    AssetClass.etf: ["^GSPC", "^VIX"],
    AssetClass.index: ["^VIX", "DX-Y.NYB", "GC=F"],
    AssetClass.crypto: ["ETH-USD", "^IXIC", "DX-Y.NYB", "^VIX"],
    AssetClass.fx: ["DX-Y.NYB", "GC=F", "^GSPC"],
    AssetClass.futures: ["DX-Y.NYB", "^GSPC", "^VIX"],
    AssetClass.commodity: ["DX-Y.NYB", "^GSPC", "^VIX"],
    AssetClass.rates: ["DX-Y.NYB", "^GSPC", "^VIX"],
    AssetClass.unknown: ["^GSPC", "^VIX"],
}


def get_related_assets(symbol: str, asset_class: AssetClass | str | None = None) -> list[str]:
    normalized = normalize_symbol(symbol).provider_symbol
    if normalized in DEFAULT_RELATED_ASSETS:
        return DEFAULT_RELATED_ASSETS[normalized]
    inferred = asset_class or infer_asset_class(normalized)
    if isinstance(inferred, str):
        inferred = AssetClass(inferred) if inferred in AssetClass._value2member_map_ else AssetClass.unknown
    return DEFAULT_BY_ASSET_CLASS.get(inferred, DEFAULT_BY_ASSET_CLASS[AssetClass.unknown])


def _prepare_returns(frame: pd.DataFrame, name: str) -> pd.Series:
    if "date" not in frame.columns or "close" not in frame.columns:
        raise ValueError(f"Related asset frame for {name} requires date and close columns")
    data = frame[["date", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["date", "close"]).sort_values("date")
    returns = np.log(data["close"].clip(lower=1e-12) / data["close"].clip(lower=1e-12).shift(1))
    return pd.Series(returns.to_numpy(dtype=float), index=data["date"], name=f"{name}_return")


def build_cross_asset_features(
    target_frame: pd.DataFrame,
    related_frames: dict[str, pd.DataFrame],
    *,
    rolling_window: int = 20,
) -> pd.DataFrame:
    target_returns = _prepare_returns(target_frame, "target")
    out = pd.DataFrame(index=target_returns.index)
    out["target_return"] = target_returns
    for symbol, frame in related_frames.items():
        related = _prepare_returns(frame, symbol).reindex(out.index).ffill()
        safe_name = symbol.replace("^", "").replace("=", "_").replace("-", "_").replace(".", "_")
        out[f"{safe_name}_return"] = related
        out[f"{safe_name}_rolling_corr"] = target_returns.rolling(rolling_window, min_periods=5).corr(related)
        out[f"{safe_name}_relative_strength"] = (
            target_returns.rolling(rolling_window, min_periods=5).sum()
            - related.rolling(rolling_window, min_periods=5).sum()
        )
        variance = related.rolling(rolling_window, min_periods=5).var().replace(0.0, np.nan)
        covariance = target_returns.rolling(rolling_window, min_periods=5).cov(related)
        out[f"{safe_name}_beta"] = covariance / variance
        out[f"{safe_name}_spread"] = target_returns - related
    risk_cols = [col for col in out.columns if col.endswith("_return") and col != "target_return"]
    out["risk_on_off_proxy"] = out[risk_cols].mean(axis=1) if risk_cols else 0.0
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).reset_index(names="date")


def cross_asset_context_summary(symbol: str, asset_class: AssetClass | str | None, enabled: bool) -> dict:
    related = get_related_assets(symbol, asset_class)
    return {
        "enabled": bool(enabled),
        "status": "configured" if enabled else "disabled",
        "related_assets": related,
        "warning": None if enabled else "Cross-asset feature loading is disabled by configuration.",
    }
