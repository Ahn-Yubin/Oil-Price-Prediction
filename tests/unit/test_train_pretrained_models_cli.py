import pandas as pd

from scripts.train.train_pretrained_models import _series_from_market_panel


def test_series_from_market_panel_filters_symbols_and_orders_by_time(tmp_path):
    rows = []
    for symbol in ["CL=F", "BZ=F"]:
        for idx in range(205):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=204 - idx),
                    "symbol": symbol,
                    "close": 70.0 + idx,
                }
            )
    path = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    series, used = _series_from_market_panel(str(path), symbols=["CL=F"])

    assert used == ["CL=F"]
    assert len(series) == 1
    assert len(series[0]) == 205
    assert series[0][0] > series[0][-1]
