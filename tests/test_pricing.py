"""Tests for live model pricing and registry lookups."""

import pytest

from router import (
    PROVIDER_TIERS,
    ModelRouter,
    is_known_model,
    list_registry_models,
    model_pricing,
)


def test_every_tier_model_is_priceable():
    """A tier model we can't price would silently report $0.00 costs."""
    unpriceable = [
        tier.name
        for tiers in PROVIDER_TIERS.values()
        for tier in tiers
        if model_pricing(tier.name) is None
    ]
    assert unpriceable == []


def test_pricing_prefers_registry_over_static_fallback(monkeypatch):
    import router

    monkeypatch.setattr(router, "_registry_pricing", lambda m: (9.0, 99.0))
    assert router.model_pricing("claude-opus-5") == (9.0, 99.0)


def test_pricing_falls_back_to_static_tier(monkeypatch):
    import router

    monkeypatch.setattr(router, "_registry_pricing", lambda m: None)
    assert router.model_pricing("claude-opus-5") == (5.00, 25.00)


def test_pricing_none_for_unlisted_model(monkeypatch):
    import router

    monkeypatch.setattr(router, "_registry_pricing", lambda m: None)
    assert router.model_pricing("totally-made-up-model") is None


def test_is_known_model():
    assert is_known_model("claude-opus-5")
    assert not is_known_model("totally-made-up-model")


def test_estimate_cost_uses_live_pricing(monkeypatch):
    import router

    monkeypatch.setattr(router, "_registry_pricing", lambda m: (1.0, 2.0))
    cost = ModelRouter().estimate_cost("claude-sonnet-5", 1_000_000, 1_000_000)
    assert cost == pytest.approx(3.0)


def test_registry_lookup_handles_provider_prefix():
    """gemini/x should resolve even when the registry keys it as x."""
    assert model_pricing("gemini/gemini-3.6-flash") is not None


def test_list_registry_models_filters_and_sorts():
    rows = list_registry_models("claude", limit=10)
    assert rows, "expected claude models in the registry"
    assert all("claude" in name for name, _, _ in rows)
    assert rows == sorted(rows, key=lambda r: r[1])


def test_list_registry_models_respects_limit():
    assert len(list_registry_models("", limit=5)) == 5


def test_list_registry_models_hides_reseller_aliases():
    """Browsing should show claude-opus-5, not thirty bedrock/vertex copies."""
    for name, _, _ in list_registry_models("claude", limit=40):
        assert not name.startswith(("bedrock", "snowflake/", "us.anthropic")), name


def test_list_registry_models_shows_aliases_when_asked():
    rows = list_registry_models("bedrock", limit=10)
    assert rows, "expected bedrock aliases when queried explicitly"
