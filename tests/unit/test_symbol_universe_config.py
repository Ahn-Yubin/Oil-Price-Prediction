from market_ai.data.symbol_universe import load_symbol_universe, resolve_universe


def test_symbol_universe_config_contains_required_groups():
    universes = load_symbol_universe()
    assert "oil_core" in universes
    assert "default_global" in universes
    assert "CL=F" in universes["oil_core"]
    assert "SPY" in universes["default_global"]


def test_resolve_symbol_universe():
    assert resolve_universe("oil_core")[0] == "CL=F"
