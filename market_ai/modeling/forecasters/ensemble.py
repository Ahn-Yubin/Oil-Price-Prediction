def forecast_ensemble(*args, **kwargs):
    raise RuntimeError("ensemble is removed as a fixed-weight model; use oil_context_fusion.")


__all__ = ["forecast_ensemble"]
