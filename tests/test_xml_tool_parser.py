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
