from market_ai.config import Settings
from market_ai.schemas.market import AssetClass
from market_ai.data.symbols import infer_asset_class, normalize_symbol, symbol_candidates
from market_ai.data.timeframes import normalize_timeframe


def test_oil_aliases_resolve_to_yfinance_symbols():
    assert normalize_symbol("NYMEX:CL1!").provider_symbol == "CL=F"
    assert normalize_symbol("ICEEUR:BRN1!").provider_symbol == "BZ=F"
    assert normalize_symbol("TVC:USOIL").provider_symbol == "CL=F"
    assert normalize_symbol("TVC:UKOIL").provider_symbol == "BZ=F"
    assert "CL=F" in symbol_candidates("NYMEX:CL1!")


def test_asset_class_inference_common_assets():
    assert infer_asset_class("CL=F") == AssetClass.futures
    assert infer_asset_class("BTC-USD") == AssetClass.crypto
    assert infer_asset_class("USDKRW=X") == AssetClass.fx
    assert infer_asset_class("^GSPC") == AssetClass.index
    assert infer_asset_class("SPY") == AssetClass.etf
    assert infer_asset_class("MSFT") == AssetClass.equity


def test_btcusdt_maps_to_yfinance_crypto_pair():
    symbol = normalize_symbol("BTCUSDT")
    assert symbol.provider_symbol == "BTC-USD"
    assert symbol.asset_class == AssetClass.crypto


def test_timeframe_mapping_and_fallback():
    settings = Settings(default_interval="1d")
    assert normalize_timeframe("15m", settings).normalized == "1d"
    assert normalize_timeframe("15m", settings).warning is not None
    assert normalize_timeframe("15m", settings, fallback_to_supported=False).provider_interval == "15m"
    assert normalize_timeframe("1h", settings).seconds == 3600
    unsupported = normalize_timeframe("2h", settings)
    assert unsupported.normalized == "1d"
    assert unsupported.warning is not None
    future_supported = normalize_timeframe("4h", settings)
    assert future_supported.normalized == "1d"
    assert future_supported.is_supported is True
