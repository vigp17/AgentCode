"""Tests for conversation compaction, permissions, dispatch, and cost accounting."""

from types import SimpleNamespace

import pytest

from agent import (
    AgentConfig,
    Conversation,
    _dispatch_tool,
    _get_permission,
    _record_cost,
    _run_subagents,
)
from router import ModelRouter
from settings import PermissionsSettings, Settings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assistant_with_tools(call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
    }


def _orphan_tool_messages(messages: list[dict]) -> list[dict]:
    """Tool results whose requesting assistant turn is no longer in history."""
    known: set[str] = set()
    orphans = []
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                known.add(tc["id"])
        elif msg.get("role") == "tool" and msg.get("tool_call_id") not in known:
            orphans.append(msg)
    return orphans


# ── Compaction ────────────────────────────────────────────────────────────────

def test_compact_does_not_orphan_tool_results():
    conv = Conversation(system="sys")
    conv.messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "two"},
        _assistant_with_tools("call_1"),
        {"role": "tool", "tool_call_id": "call_1", "content": "result 1"},
        {"role": "tool", "tool_call_id": "call_1", "content": "result 2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "four"},
    ]

    conv.compact(max_tokens=0)

    assert _orphan_tool_messages(conv.messages) == []
    # The summary pair replaced everything up to the first post-tool message.
    assert conv.messages[2]["content"] == "a2"


def test_compact_skips_when_tail_is_all_tool_results():
    """Nothing can be dropped without orphaning — leave the history alone."""
    conv = Conversation(system="sys")
    conv.messages = [
        {"role": "user", "content": "one"},
        _assistant_with_tools("call_1"),
        *[
            {"role": "tool", "tool_call_id": "call_1", "content": f"r{i}"}
            for i in range(6)
        ],
    ]
    before = list(conv.messages)

    conv.compact(max_tokens=0)

    assert conv.messages == before


def test_compact_noop_below_threshold():
    conv = Conversation(system="sys")
    conv.messages = [{"role": "user", "content": "hi"}] * 10
    before = list(conv.messages)

    conv.compact(max_tokens=80_000)

    assert conv.messages == before


# ── Permissions ───────────────────────────────────────────────────────────────

def _settings(**kwargs) -> Settings:
    return Settings(permissions=PermissionsSettings(**kwargs))


def test_deny_list_wins_over_auto_approve_all():
    config = AgentConfig(
        settings=_settings(auto_approve_all=True, deny=["run_command"])
    )
    assert _get_permission("run_command", config) == "deny"


def test_auto_approve_all_allows_write_tools():
    config = AgentConfig(settings=_settings(auto_approve_all=True))
    assert _get_permission("write_file", config) == "allow"


def test_read_only_tools_auto_approved_by_default():
    config = AgentConfig(settings=_settings())
    assert _get_permission("read_file", config) == "allow"
    assert _get_permission("write_file", config) == "ask"


def test_mcp_tools_ask_by_default():
    """MCP tools are not on the default auto-approve list."""
    config = AgentConfig(settings=_settings())
    assert _get_permission("mcp__github__create_issue", config) == "ask"


def test_mcp_tools_ask_without_settings():
    """Backward-compat path: no settings loaded still gates MCP tools."""
    config = AgentConfig(settings=None)
    assert _get_permission("mcp__github__create_issue", config) == "ask"
    assert _get_permission("read_file", config) == "allow"


# ── Dispatch ──────────────────────────────────────────────────────────────────

def test_dispatch_blocks_denied_tool(tmp_path):
    config = AgentConfig(settings=_settings(deny=["read_file"]))
    result = _dispatch_tool("read_file", {"path": str(tmp_path)}, config, silent=True)
    assert "blocked by settings" in result


def test_dispatch_honours_user_denial(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("secret")
    asked: list[str] = []

    def refuse(tool_name: str, args: dict) -> bool:
        asked.append(tool_name)
        return False

    config = AgentConfig(settings=_settings(), permission_cb=refuse)
    result = _dispatch_tool("write_file", {"path": str(target), "content": "x"}, config, True)

    assert asked == ["write_file"]
    assert "denied by user" in result
    assert target.read_text() == "secret"


def test_dispatch_runs_approved_tool(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello")
    config = AgentConfig(settings=_settings(), permission_cb=lambda n, a: True)

    result = _dispatch_tool("read_file", {"path": str(target)}, config, True)

    assert "hello" in result


# ── Subagents ─────────────────────────────────────────────────────────────────

def test_subagents_stop_at_depth_limit():
    config = AgentConfig(subagent_depth=1, max_subagent_depth=1)
    result = _run_subagents(["do a thing"], config)
    assert "depth limit" in result


def test_subagents_reject_empty_task_list():
    config = AgentConfig()
    assert "no tasks" in _run_subagents([], config)


# ── Cost accounting ───────────────────────────────────────────────────────────

def test_record_cost_prices_known_model():
    router = ModelRouter(provider="anthropic")
    usage = SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000)

    _record_cost(router, "claude-opus-5", usage)

    tracker = router.cost_tracker
    assert tracker.total_input_tokens == 1_000_000
    assert tracker.total_output_tokens == 1_000_000
    assert tracker.total_cost == pytest.approx(30.0)  # $5 in + $25 out


def test_record_cost_tracks_tokens_for_unknown_model():
    router = ModelRouter(provider="anthropic")
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)

    _record_cost(router, "custom/some-unlisted-model", usage)

    assert router.cost_tracker.total_input_tokens == 100
    assert router.cost_tracker.total_output_tokens == 50


def test_begin_turn_resets_per_turn_counters():
    router = ModelRouter(provider="anthropic")
    usage = SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=0)

    _record_cost(router, "claude-opus-5", usage)
    assert router.cost_tracker.last_turn_cost == pytest.approx(5.0)

    router.cost_tracker.begin_turn()
    assert router.cost_tracker.last_turn_cost == 0.0
    assert router.cost_tracker.total_cost == pytest.approx(5.0)
