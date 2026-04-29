def forecast_cycle(*args, **kwargs):
    raise RuntimeError("cycle is removed as a standalone model; use deep_lstm_tcn_fusion or cycle features.")


__all__ = ["forecast_cycle"]
