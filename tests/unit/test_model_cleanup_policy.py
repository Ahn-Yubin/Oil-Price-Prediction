import pytest

from market_ai.modeling.model_catalog import (
    REMOVED_LEGACY_MODELS,
    USER_FACING_MODELS,
    InvalidModelRequest,
    resolve_model_selection,
)


def test_user_facing_models_exclude_removed_legacy_models():
    assert {"cycle", "lstm", "tcn", "ensemble"}.isdisjoint(USER_FACING_MODELS)
    assert USER_FACING_MODELS == ("oil_context_fusion",)


def test_removed_model_request_is_clear_error():
    with pytest.raises(InvalidModelRequest) as exc:
        resolve_model_selection("lstm")
    assert exc.value.removed == ["lstm"]
    assert "oil_context_fusion" in str(exc.value.as_detail())


def test_chart_compatibility_can_warn_and_continue():
    selection = resolve_model_selection("cycle,motif", allow_removed_as_warning=True)
    assert selection.selected == ["oil_context_fusion"]
    assert selection.removed_requested == ["cycle", "motif"]
    assert selection.warnings
    assert "cycle" in REMOVED_LEGACY_MODELS
