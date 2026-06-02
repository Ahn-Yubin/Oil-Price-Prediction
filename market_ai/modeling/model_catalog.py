from __future__ import annotations

from dataclasses import dataclass


USER_FACING_MODELS: tuple[str, ...] = (
    "oil_context_fusion",
)

BACKTEST_ONLY_MODELS: tuple[str, ...] = (
    "flat",
    "simple_moving_average_path",
    "regime_ensemble",
)

REMOVED_LEGACY_MODELS: dict[str, str] = {
    "cycle": "Standalone cycle extrapolation was removed; cycle signals are now feature inputs.",
    "lstm": "The live cached LSTM was removed; use oil_context_fusion.",
    "tcn": "The live cached TCN was removed; use oil_context_fusion.",
    "ensemble": "The fixed-weight ensemble was removed; use oil_context_fusion.",
    "motif": "Standalone motif output is now an internal benchmark; use oil_context_fusion.",
    "pattern_mlp": "Standalone pattern MLP output is now an internal benchmark; use oil_context_fusion.",
    "deep_lstm_tcn_fusion": "This legacy deep branch was merged into oil_context_fusion.",
    "llm_context_seq_moe": "This legacy context branch was merged into oil_context_fusion.",
    "random_walk": "Baseline output is now backtest-only; use oil_context_fusion.",
    "drift": "Baseline output is now backtest-only; use oil_context_fusion.",
    "seasonal_naive": "Baseline output is now backtest-only; use oil_context_fusion.",
    "volatility_scaled_naive": "Baseline output is now backtest-only; use oil_context_fusion.",
}

DEPRECATED_REPLACEMENTS: dict[str, str] = {
    name: "oil_context_fusion" for name in REMOVED_LEGACY_MODELS
}

DEEP_MODELS: tuple[str, ...] = (
    "oil_context_fusion",
)

LEGACY_DEEP_MODELS: tuple[str, ...] = (
    "deep_lstm_tcn_fusion",
    "llm_context_seq_moe",
)

BASELINE_MODELS: tuple[str, ...] = (
    "random_walk",
    "drift",
    "seasonal_naive",
    "volatility_scaled_naive",
)

CLASSICAL_MODELS: tuple[str, ...] = ("motif",)
LEGACY_ARTIFACT_MODELS: tuple[str, ...] = ("pattern_mlp",)


class InvalidModelRequest(ValueError):
    def __init__(
        self,
        message: str,
        *,
        unknown: list[str] | None = None,
        removed: list[str] | None = None,
        supported: tuple[str, ...] = USER_FACING_MODELS,
    ) -> None:
        super().__init__(message)
        self.unknown = unknown or []
        self.removed = removed or []
        self.supported = supported

    def as_detail(self) -> dict[str, object]:
        return {
            "message": str(self),
            "unknown_models": self.unknown,
            "removed_models": self.removed,
            "supported_models": list(self.supported),
            "replacement_models": {name: DEPRECATED_REPLACEMENTS.get(name) for name in self.removed},
        }


@dataclass(frozen=True)
class ModelSelection:
    selected: list[str]
    requested: list[str]
    removed_requested: list[str]
    deprecated_requested: list[str]
    warnings: list[str]


def split_model_query(models: str | None) -> list[str]:
    if not models:
        return []
    return [part.strip() for part in models.split(",") if part.strip()]


def resolve_model_selection(
    models: str | None,
    *,
    supported: tuple[str, ...] = USER_FACING_MODELS,
    default: tuple[str, ...] = USER_FACING_MODELS,
    allow_removed_as_warning: bool = False,
) -> ModelSelection:
    requested = split_model_query(models)
    if not requested:
        return ModelSelection(
            selected=list(default),
            requested=[],
            removed_requested=[],
            deprecated_requested=[],
            warnings=[],
        )

    seen: set[str] = set()
    selected: list[str] = []
    removed: list[str] = []
    unknown: list[str] = []
    warnings: list[str] = []

    for name in requested:
        if name in seen:
            continue
        seen.add(name)
        if name in REMOVED_LEGACY_MODELS:
            removed.append(name)
            replacement = DEPRECATED_REPLACEMENTS.get(name)
            if allow_removed_as_warning:
                warnings.append(f"Model '{name}' is removed/deprecated. Use '{replacement}' instead.")
                continue
            continue
        if name not in supported:
            unknown.append(name)
            continue
        selected.append(name)

    if unknown or (removed and not allow_removed_as_warning):
        parts: list[str] = []
        if unknown:
            parts.append(f"unknown model(s): {', '.join(unknown)}")
        if removed:
            parts.append(f"removed/deprecated model(s): {', '.join(removed)}")
        raise InvalidModelRequest(
            "; ".join(parts) + f". Supported models: {', '.join(supported)}.",
            unknown=unknown,
            removed=removed,
            supported=supported,
        )

    if not selected:
        selected = list(default)

    return ModelSelection(
        selected=selected,
        requested=requested,
        removed_requested=removed,
        deprecated_requested=removed,
        warnings=warnings,
    )
