from __future__ import annotations

import re

from market_ai.schemas.market import AssetClass, AssetMetadata, MarketSymbol


SYMBOL_ALIASES: dict[str, str] = {
    "NYMEX:CL1!": "CL=F",
    "TVC:USOIL": "CL=F",
    "ICEEUR:BRN1!": "BZ=F",
    "TVC:UKOIL": "BZ=F",
    "FX_IDC:USDKRW": "USDKRW=X",
    "TVC:DXY": "DX-Y.NYB",
    "OANDA:XAUUSD": "GC=F",
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
}

FUTURES_ALIASES: dict[str, str] = {
    "CL1!": "CL=F",
    "USOIL": "CL=F",
    "UKOIL": "BZ=F",
    "BRN1!": "BZ=F",
    "XAUUSD": "GC=F",
}

FUTURES_ROOTS = {"CL", "BZ", "NG", "RB", "HO", "GC", "SI", "HG", "ZB", "ZN", "ZF", "ZT", "ES", "NQ", "YM", "RTY"}
ETF_SYMBOLS = {"SPY", "QQQ", "DIA", "IWM", "XLE", "USO", "GLD", "SLV", "TLT", "HYG", "LQD"}
CRYPTO_ROOTS = {"BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK"}
INDEX_PREFIXES = ("^",)


def symbol_candidates(raw_symbol: str, default_symbol: str = "CL=F") -> list[str]:
    raw = (raw_symbol or default_symbol).strip()
    if not raw:
        raw = default_symbol
    upper = raw.upper()

    candidates: list[str] = []
    if upper in SYMBOL_ALIASES:
        candidates.append(SYMBOL_ALIASES[upper])

    right = upper.split(":", 1)[1] if ":" in upper else upper
    candidates.append(right)

    if right in FUTURES_ALIASES:
        candidates.append(FUTURES_ALIASES[right])

    if right.endswith("USDT") and len(right) > 4:
        candidates.append(f"{right[:-4]}-USD")

    if re.fullmatch(r"[A-Z]{6}", right):
        candidates.append(f"{right}=X")

    candidates.extend([raw, upper])

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def infer_asset_class(provider_symbol: str, requested_symbol: str | None = None) -> AssetClass:
    symbol = (provider_symbol or "").upper()
    requested = (requested_symbol or "").upper()
    root = symbol.split("=", 1)[0].split("-", 1)[0].lstrip("^")

    if symbol.startswith(INDEX_PREFIXES) or requested.startswith(("TVC:DXY", "INDEX:")):
        return AssetClass.index
    if symbol.endswith("=X") or requested.startswith(("FX:", "FX_IDC:", "OANDA:")):
        return AssetClass.fx
    if "-" in symbol and symbol.split("-", 1)[0] in CRYPTO_ROOTS:
        return AssetClass.crypto
    if symbol.endswith("=F") or root in FUTURES_ROOTS or requested.endswith("1!"):
        return AssetClass.futures
    if symbol in ETF_SYMBOLS:
        return AssetClass.etf
    if re.fullmatch(r"[A-Z]{1,5}", symbol):
        return AssetClass.equity
    return AssetClass.unknown


def normalize_symbol(raw_symbol: str, default_symbol: str = "CL=F") -> MarketSymbol:
    requested = (raw_symbol or default_symbol).strip() or default_symbol
    provider_symbol = symbol_candidates(requested, default_symbol=default_symbol)[0]
    asset_class = infer_asset_class(provider_symbol, requested)
    exchange = requested.split(":", 1)[0].upper() if ":" in requested else None
    root = provider_symbol.split("=", 1)[0].split("-", 1)[0].lstrip("^")
    return MarketSymbol(
        requested=requested,
        normalized=provider_symbol,
        provider_symbol=provider_symbol,
        asset_class=asset_class,
        exchange=exchange,
        root=root,
        description=None,
    )


def asset_metadata(symbol: MarketSymbol) -> AssetMetadata:
    quote_currency = None
    if symbol.asset_class == AssetClass.crypto and "-" in symbol.provider_symbol:
        quote_currency = symbol.provider_symbol.split("-", 1)[1]
    if symbol.asset_class == AssetClass.fx and symbol.provider_symbol.endswith("=X"):
        quote_currency = symbol.provider_symbol[3:6] if len(symbol.provider_symbol) >= 6 else None
    return AssetMetadata(
        symbol=symbol.requested,
        provider_symbol=symbol.provider_symbol,
        asset_class=symbol.asset_class,
        exchange=symbol.exchange,
        currency=None,
        quote_currency=quote_currency,
        description=symbol.description,
        roll_policy="front_contract_continuous_placeholder" if symbol.asset_class == AssetClass.futures else None,
    )
