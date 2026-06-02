def forecast_cycle(*args, **kwargs):
    raise RuntimeError("cycle is removed as a standalone model; use oil_context_fusion.")


__all__ = ["forecast_cycle"]
