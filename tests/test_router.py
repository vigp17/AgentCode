"""Tests for cost-aware model routing."""

from router import ModelRouter, classify_complexity


def test_classify_light():
    assert classify_complexity("what is a decorator?") == "light"
    assert classify_complexity("hello") == "light"


def test_classify_medium():
    assert classify_complexity("fix the bug in this function") == "medium"
    assert classify_complexity("write a unit test for parse_xml") == "medium"


def test_classify_heavy():
    assert classify_complexity(
        "refactor the entire codebase architecture across multiple files"
    ) == "heavy"


def test_route_anthropic_tiers():
    router = ModelRouter(provider="anthropic", enabled=True)
    model, tier, _ = router.route("hello")
    assert tier == "light"
    assert "haiku" in model

    model, tier, _ = router.route("fix the bug in login")
    assert tier == "medium"
    assert "sonnet" in model

    model, tier, _ = router.route(
        "refactor the entire architecture across multiple files and redesign the system"
    )
    assert tier == "heavy"
    assert model == "claude-opus-4-8"


def test_route_disabled_uses_default():
    router = ModelRouter(
        provider="anthropic",
        default_model="claude-fable-5",
        enabled=False,
    )
    model, tier, reason = router.route("refactor everything")
    assert model == "claude-fable-5"
    assert tier == "default"
    assert "disabled" in reason


def test_detect_provider():
    router = ModelRouter()
    assert router.detect_provider("claude-fable-5") == "anthropic"
    assert router.detect_provider("gpt-4o") == "openai"
    assert router.detect_provider("gemini/gemini-2.5-pro") == "gemini"


def test_estimate_cost_known_model():
    router = ModelRouter()
    cost = router.estimate_cost("claude-opus-4-8", 1_000_000, 1_000_000)
    assert abs(cost - 30.0) < 0.01  # $5 in + $25 out
