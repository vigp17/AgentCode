"""Tests for XML / bare-JSON tool-call parsing."""

import json

from xml_tool_parser import (
    looks_like_json_tool_call,
    looks_like_xml_tool_call,
    parse_json_tool_calls,
    parse_xml_tool_calls,
    strip_think,
)


def test_strip_think_removes_block():
    text = "<think>secret reasoning</think>\nHello"
    assert strip_think(text) == "Hello"


def test_strip_think_handles_missing_open_tag():
    text = "partial reasoning</think>\nDone"
    assert strip_think(text) == "Done"


def test_looks_like_xml_tool_call():
    assert looks_like_xml_tool_call("<tool_call><function=read_file>")
    assert not looks_like_xml_tool_call("just plain text")


def test_parse_xml_tool_calls():
    text = """
<think>planning</think>
I'll read the file.
<tool_call>
<function=read_file>
<parameter=path>src/main.py</parameter>
</function>
</tool_call>
"""
    cleaned, calls = parse_xml_tool_calls(text)
    assert "I'll read the file." in cleaned
    assert "<tool_call>" not in cleaned
    assert "<think>" not in cleaned
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
    assert json.loads(calls[0]["arguments"]) == {"path": "src/main.py"}
    assert calls[0]["id"].startswith("call_")


def test_looks_like_json_tool_call():
    assert looks_like_json_tool_call('{"name": "git_status", "arguments": {}}')
    assert not looks_like_json_tool_call('{"foo": 1}')


def test_parse_json_tool_calls_bare():
    text = 'Sure.\n{"name": "git_status", "arguments": {}}'
    cleaned, calls = parse_json_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "git_status"
    assert json.loads(calls[0]["arguments"]) == {}
    assert "git_status" not in cleaned or cleaned.strip() == "Sure."


def test_parse_json_tool_calls_wrapped():
    text = '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    cleaned, calls = parse_json_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
    assert json.loads(calls[0]["arguments"])["path"] == "a.py"
    assert cleaned == ""


# ── Gating: describing a tool call must not execute one ───────────────────────

def test_parse_json_rejects_unknown_tool_name():
    text = '{"name": "not_a_real_tool", "arguments": {}}'
    _, calls = parse_json_tool_calls(text, valid_names={"read_file", "git_status"})
    assert calls == []


def test_parse_json_accepts_known_tool_name():
    text = '{"name": "git_status", "arguments": {}}'
    _, calls = parse_json_tool_calls(text, valid_names={"read_file", "git_status"})
    assert len(calls) == 1


def test_parse_json_rejects_call_embedded_in_prose():
    """An explanation that happens to contain JSON is not a tool call."""
    text = (
        "To check the repository state you would send a tool call to the model "
        'with the payload {"name": "run_command", "arguments": {"command": "rm -rf /"}} '
        "and the runtime would then execute it on your behalf, which is why the "
        "permission prompt exists in the first place. Let me know if you want more detail."
    )
    _, calls = parse_json_tool_calls(text, valid_names={"run_command"})
    assert calls == []


def test_parse_json_rejects_fenced_example():
    text = (
        "Here's the shape:\n\n```json\n"
        '{"name": "run_command", "arguments": {"command": "ls"}}\n'
        "```"
    )
    _, calls = parse_json_tool_calls(text, valid_names={"run_command"})
    assert calls == []


def test_parse_json_allows_short_preamble():
    text = 'Sure, checking now.\n{"name": "git_status", "arguments": {}}'
    cleaned, calls = parse_json_tool_calls(text, valid_names={"git_status"})
    assert len(calls) == 1
    assert cleaned == "Sure, checking now."


def test_parse_json_wrapped_bypasses_prose_gate():
    """Explicit <tool_call> tags are unambiguous regardless of surrounding text."""
    text = (
        "I will now inspect the working tree, which requires running git status "
        "against the current repository so we can see what has changed since the "
        "last commit and decide what to do next with the pending modifications.\n"
        '<tool_call>{"name": "git_status", "arguments": {}}</tool_call>'
    )
    _, calls = parse_json_tool_calls(text, valid_names={"git_status"})
    assert len(calls) == 1
