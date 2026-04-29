def forecast_ensemble(*args, **kwargs):
    raise RuntimeError("ensemble is removed as a fixed-weight model; use llm_context_seq_moe.")


__all__ = ["forecast_ensemble"]
