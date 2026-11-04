"""server.py live model/effort switching (no LLM calls)."""


import pytest

import finetuned
import server
from agent import AgentConfig, Conversation
from router import ModelRouter


@pytest.fixture
def out(monkeypatch):
    msgs = []
    monkeypatch.setattr(server, "_write", lambda m: msgs.append(m))
    finetuned.configure({})
    yield msgs
    finetuned.configure({})


def _config():
    return AgentConfig(model="claude-sonnet-5", router=ModelRouter(default_model="claude-sonnet-5"))


def test_set_model_switches_in_place_and_disables_routing(out):
    cfg = _config()
    server._set_model(cfg, "axon-1")
    assert cfg.model == "axon-1"
    assert cfg.router.enabled is False and cfg.router.default_model == "axon-1"
    assert cfg.router.provider == "agentcode"
    assert out[-1] == {"type": "model_changed", "model": "axon-1", "effort": "medium", "routing": False}


def test_set_model_requires_name(out):
    server._set_model(_config(), "")
    assert out[-1]["type"] == "error"


def test_unconnected_ecosystem_model_explains_how_to_connect(out, monkeypatch):
    monkeypatch.setattr(server.litellm, "completion", lambda **k: pytest.fail("no LLM call"))
    cfg = _config()
    server._set_model(cfg, "axon-1")
    conv = Conversation(system="sys")
    server._server_turn("hi", conv, cfg, None)
    assert "AGENTCODE_FT_AXON_1_API_BASE" in out[-2]["message"]
    assert out[-1]["type"] == "done" and conv.messages == []


def test_image_model_does_not_chat(out, monkeypatch):
    monkeypatch.setenv("AGENTCODE_FT_CHITRA_1_API_BASE", "http://x/v1")
    finetuned.configure({})
    monkeypatch.setattr(server.litellm, "completion", lambda **k: pytest.fail("no LLM call"))
    cfg = _config()
    server._set_model(cfg, "chitra-1")
    server._server_turn("draw a cat", Conversation(system="s"), cfg, None)
    assert "generates images" in out[-2]["message"]


def test_effort_reaches_axon_prompt_and_iteration_cap(out, monkeypatch):
    monkeypatch.setenv("AGENTCODE_FT_AXON_1_API_BASE", "http://x/v1")
    finetuned.configure({})
    seen = []

    class Chunk:
        usage = None
        def __init__(self, text):
            delta = type("D", (), {"content": text, "tool_calls": None})()
            self.choices = [type("C", (), {"delta": delta})()]

    def fake_completion(**kw):
        seen.append(kw)
        return iter([Chunk("ok")])

    monkeypatch.setattr(server.litellm, "completion", fake_completion)
    cfg = _config()
    server._set_model(cfg, "axon-1")
    cfg.effort = "high"
    server._server_turn("refactor", Conversation(system="sys"), cfg, None)
    assert seen[0]["messages"][0]["content"].endswith("<effort>high</effort>")
    assert seen[0]["api_base"] == "http://x/v1"
    assert out[-1]["type"] == "done"
