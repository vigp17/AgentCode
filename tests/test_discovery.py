"""Tests for automatic model discovery (no network: provider HTTP is stubbed)."""

import json

import pytest

import model_discovery as md
from model_discovery import DiscoveredModel
from router import ModelRouter


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTCODE_MODELS_CACHE", str(tmp_path / "cache.json"))
    monkeypatch.delenv("AGENTCODE_DISABLE_DISCOVERY", raising=False)
    for envs in md.API_KEY_ENV.values():
        for e in envs:
            monkeypatch.delenv(e, raising=False)


ANTHROPIC_PAGE = {
    "data": [
        {"id": "claude-opus-6", "display_name": "Claude Opus 6", "created_at": "2026-10-01T00:00:00Z"},
        {"id": "claude-opus-5", "display_name": "Claude Opus 5", "created_at": "2026-02-01T00:00:00Z"},
        {"id": "claude-sonnet-6", "display_name": "Claude Sonnet 6", "created_at": "2026-10-01T00:00:00Z"},
        {"id": "claude-haiku-5-5", "display_name": "Claude Haiku 5.5", "created_at": "2026-06-01T00:00:00Z"},
    ],
    "has_more": False,
}

OPENAI_PAGE = {"data": [
    {"id": "gpt-7-sol", "created": 3}, {"id": "gpt-7-terra", "created": 3},
    {"id": "gpt-7-luna", "created": 3}, {"id": "gpt-7-terra-2026-11-01", "created": 3},
    {"id": "text-embedding-4", "created": 2}, {"id": "gpt-7-realtime", "created": 3},
    {"id": "whisper-2", "created": 1}, {"id": "ft:gpt-7-luna:acme:agentcode-light:abc123", "created": 4},
]}

GEMINI_PAGE = {"models": [
    {"name": "models/gemini-4-pro", "displayName": "Gemini 4 Pro",
     "supportedGenerationMethods": ["generateContent", "countTokens"]},
    {"name": "models/gemini-4-flash", "displayName": "Gemini 4 Flash",
     "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-4-flash-lite", "displayName": "Gemini 4 Flash-Lite",
     "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-embedding-2", "supportedGenerationMethods": ["embedContent"]},
    {"name": "models/imagen-5", "supportedGenerationMethods": ["predict"]},
]}


def _stub_http(monkeypatch, calls=None):
    def fake(url, headers):
        if calls is not None:
            calls.append(url)
        if "anthropic" in url:
            assert headers["x-api-key"] == "ak"
            return ANTHROPIC_PAGE
        if "openai" in url:
            assert headers["Authorization"] == "Bearer ok"
            return OPENAI_PAGE
        if "googleapis" in url:
            assert headers["x-goog-api-key"] == "gk"
            return GEMINI_PAGE
        raise AssertionError(url)
    monkeypatch.setattr(md, "_get_json", fake)


def _keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    monkeypatch.setenv("GEMINI_API_KEY", "gk")


def test_discovers_new_models_from_provider_apis(monkeypatch):
    _keys(monkeypatch)
    _stub_http(monkeypatch)
    res = md.discover_all()
    assert [m.id for m in res["anthropic"]][0] in ("claude-opus-6", "claude-sonnet-6")
    openai_ids = {m.id for m in res["openai"]}
    assert {"gpt-7-sol", "gpt-7-terra", "gpt-7-luna"} <= openai_ids
    # Non-chat models and dated snapshots of a listed alias are filtered out.
    assert not openai_ids & {"text-embedding-4", "gpt-7-realtime", "whisper-2", "gpt-7-terra-2026-11-01"}
    # The user's own fine-tunes come through, with a readable label.
    ft = next(m for m in res["openai"] if m.id.startswith("ft:"))
    assert "agentcode-light" in ft.label
    assert {m.id for m in res["gemini"]} == {
        "gemini/gemini-4-pro", "gemini/gemini-4-flash", "gemini/gemini-4-flash-lite"}


def test_results_are_cached_and_never_store_the_key(monkeypatch, tmp_path):
    _keys(monkeypatch)
    calls: list[str] = []
    _stub_http(monkeypatch, calls)
    md.discover_all()
    n = len(calls)
    md.discover_all()
    assert len(calls) == n  # second call served from cache
    md.discover_all(refresh=True)
    assert len(calls) == 2 * n
    raw = (tmp_path / "cache.json").read_text()
    assert "ak" not in json.loads(raw)["anthropic"]["key"]
    assert '"ok"' not in raw and '"gk"' not in raw


def test_cache_is_per_key(monkeypatch):
    _keys(monkeypatch)
    calls: list[str] = []
    _stub_http(monkeypatch, calls)
    md.discover_provider("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")  # same key → cached
    md.discover_provider("anthropic")
    assert len(calls) == 1


def test_no_key_falls_back_to_registry(monkeypatch):
    monkeypatch.setattr(md, "_get_json", lambda *a: pytest.fail("no key → no HTTP"))
    res = md.discover_provider("anthropic")
    assert res and all(m.source == "registry" for m in res)
    assert all(m.id.startswith("claude") for m in res)


def test_network_error_uses_stale_cache_then_registry(monkeypatch):
    import urllib.error
    _keys(monkeypatch)
    _stub_http(monkeypatch)
    md.discover_provider("anthropic")

    def boom(*a):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(md, "_get_json", boom)
    stale = md.discover_provider("anthropic", refresh=True)
    assert "claude-opus-6" in {m.id for m in stale}

    monkeypatch.setenv("ANTHROPIC_API_KEY", "other-key")
    fallback = md.discover_provider("anthropic")
    assert fallback and all(m.source == "registry" for m in fallback)


def test_disabled_by_env(monkeypatch):
    _keys(monkeypatch)
    monkeypatch.setenv("AGENTCODE_DISABLE_DISCOVERY", "1")
    monkeypatch.setattr(md, "_get_json", lambda *a: pytest.fail("disabled"))
    assert md.discover_all() == {"anthropic": [], "openai": [], "gemini": []}


# ── auto-tiering ──────────────────────────────────────────────────────────────

def _m(i, provider, created=None):
    return DiscoveredModel(id=i, provider=provider, label=i, created=created)


def test_anthropic_tiers_pick_newest_of_each_family():
    models = [_m(x["id"], "anthropic", md._parse_iso(x["created_at"])) for x in ANTHROPIC_PAGE["data"]]
    assert md.auto_tiers("anthropic", models) == {
        "light": "claude-haiku-5-5", "medium": "claude-sonnet-6", "heavy": "claude-opus-6"}


def test_gemini_tiers_by_version_and_class():
    ids = ["gemini/gemini-3.6-flash", "gemini/gemini-4-flash", "gemini/gemini-4-flash-lite",
           "gemini/gemini-3.1-pro-preview", "gemini/gemini-4-pro-preview", "gemini/gemini-4-pro"]
    assert md.auto_tiers("gemini", [_m(i, "gemini") for i in ids]) == {
        "light": "gemini/gemini-4-flash-lite", "medium": "gemini/gemini-4-flash",
        "heavy": "gemini/gemini-4-pro"}


def test_openai_tiers_by_size_word():
    ids = ["gpt-6-luna", "gpt-6.1-terra", "gpt-6.1-sol", "gpt-6-sol", "gpt-6.1-pro", "ft:gpt-6-luna:x:y:z"]
    assert md.auto_tiers("openai", [_m(i, "openai") for i in ids], price=lambda m: (1, 1)) == {
        "light": "gpt-6-luna", "medium": "gpt-6.1-terra", "heavy": "gpt-6.1-sol"}


def test_openai_tiers_price_fallback_for_unknown_names():
    prices = {"gpt-9-a": (0.1, 1), "gpt-9-b": (1, 5), "gpt-9-c": (5, 20), "gpt-8-z": (0.01, 0.1)}
    picks = md.auto_tiers("openai", [_m(i, "openai") for i in prices], price=prices.get)
    assert picks == {"light": "gpt-9-a", "medium": "gpt-9-b", "heavy": "gpt-9-c"}


def test_router_uses_discovered_tiers_but_overrides_win(monkeypatch):
    import router as router_mod
    monkeypatch.setattr(router_mod, "is_known_model", lambda m: True)
    r = ModelRouter(provider="anthropic", heavy_model="claude-fable-5")
    r.apply_discovered(
        {"anthropic": [_m("claude-haiku-6", "anthropic"), _m("claude-sonnet-6", "anthropic"),
                       _m("claude-opus-6", "anthropic")]},
        md.auto_tiers,
    )
    assert r.route("explain this")[0] == "claude-haiku-6"
    assert r.route("write a function to add numbers")[0] == "claude-sonnet-6"
    assert r.route("refactor the entire codebase")[0] == "claude-fable-5"  # explicit override


def test_router_skips_unpriceable_discovered_models(monkeypatch):
    import router as router_mod
    monkeypatch.setattr(router_mod, "is_known_model", lambda m: m != "claude-opus-7")
    r = ModelRouter(provider="anthropic")
    r.apply_discovered({"anthropic": [_m("claude-opus-7", "anthropic")]}, md.auto_tiers)
    assert r.route("refactor the entire codebase")[0] == "claude-opus-5"  # static fallback


def test_to_wire_puts_finetunes_first():
    rows = md.to_wire({"anthropic": [_m("claude-x", "anthropic")]},
                      [DiscoveredModel("agentcode-medium", "agentcode", "AgentCode Medium", source="finetuned")])
    assert [r["id"] for r in rows] == ["agentcode-medium", "claude-x"]
