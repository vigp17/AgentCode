"""
Parse Hermes-style XML tool calls into OpenAI-compatible tool_calls.

Some fine-tuned open-weight models emit tool calls as inline XML in the
assistant's text response instead of using the structured `tool_calls` field
that litellm/OpenAI clients expect. This module detects that shape and
converts it.

Input shape we handle:

    <think>...optional reasoning...</think>

    <tool_call>
    <function=tool_name>
    <parameter=key1>value1</parameter>
    <parameter=key2>value2</parameter>
    </function>
    </tool_call>

Output: the same payload as litellm's structured tool_calls — a list of
{"id", "name", "arguments"} dicts plus the cleaned text (think blocks stripped,
tool_call blocks removed).
"""

import json
import re
import uuid


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# Qwen 3 / similar templates often consume the opening <think> tag during
# prompt construction but leave the </think>. In that case everything from
# the start of the response up to and including the first </think> is
# reasoning that should be stripped.
_OPEN_THINK_MISSING_RE = re.compile(r"\A.*?</think>\s*", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)


def looks_like_xml_tool_call(text: str) -> bool:
    """Cheap check before doing full regex parsing."""
    return "<tool_call>" in text and "<function=" in text


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks so users don't see chain-of-thought.

    Also handles the chat-template-ate-the-opening-tag case: if </think>
    appears but no matching <think>, strip everything up to and including it.
    """
    text = _THINK_RE.sub("", text)
    if "</think>" in text and "<think>" not in text:
        text = _OPEN_THINK_MISSING_RE.sub("", text)
    return text


def parse_xml_tool_calls(text: str) -> tuple[str, list[dict]]:
    """
    Extract XML tool calls from text and return (cleaned_text, tool_calls).

    cleaned_text has the <think> and <tool_call> blocks removed so it can be
    streamed to the user as a normal assistant reply.

    tool_calls is a list of dicts shaped like litellm's accumulator:
        {"id": "call_xxx", "name": "tool_name", "arguments": '{"key": "value"}'}
    """
    cleaned = strip_think(text)
    tool_calls: list[dict] = []

    for tc_match in _TOOL_CALL_RE.finditer(cleaned):
        body = tc_match.group(1)
        for fn_match in _FUNCTION_RE.finditer(body):
            name = fn_match.group(1).strip()
            params_body = fn_match.group(2)
            args = {
                p.group(1).strip(): p.group(2).strip()
                for p in _PARAMETER_RE.finditer(params_body)
            }
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "name": name,
                "arguments": json.dumps(args),
            })

    cleaned = _TOOL_CALL_RE.sub("", cleaned).strip()
    return cleaned, tool_calls


# ── JSON-style tool calls (small open-weight fine-tunes) ──────────────────────
# Smaller fine-tunes often emit a bare JSON object instead of the XML form:
#     {"name": "git_status", "arguments": {}}
# optionally wrapped in <tool_call>...</tool_call>. litellm doesn't pick these
# up as structured tool_calls, so we detect and convert them too.
#
# Detection is deliberately conservative. An assistant *explaining* a tool call
# ("you would call {"name": "run_command", ...}") must not trigger a real
# execution, so a candidate is only accepted when it names a tool the agent
# actually has and it isn't buried in prose or a fenced code block.

# Longest lead-in ("Sure." / "Let me check.") we still treat as a real call.
MAX_PREAMBLE_CHARS = 200


def looks_like_json_tool_call(text: str) -> bool:
    """Cheap pre-filter: a JSON object mentioning both name and arguments.

    Only a hint — parse_json_tool_calls applies the real gating.
    """
    return '"name"' in text and '"arguments"' in text


def _extract_json_objects(text: str) -> list[str]:
    """Return top-level {...} substrings via brace matching (ignores braces in strings)."""
    objs: list[str] = []
    depth = 0
    start = None
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start:i + 1])
                    start = None
    return objs


def parse_json_tool_calls(
    text: str, valid_names: set[str] | None = None
) -> tuple[str, list[dict]]:
    """
    Extract bare-JSON tool calls and return (cleaned_text, tool_calls).

    Handles {"name": ..., "arguments": {...}} objects, with or without
    surrounding <tool_call> tags. Returns the same accumulator shape as
    parse_xml_tool_calls.

    valid_names, when given, restricts matches to tools the agent actually
    has. Unwrapped candidates are additionally rejected when they sit inside
    a fenced code block or a long block of prose — that shape means the model
    is describing a tool call rather than making one.
    """
    cleaned = strip_think(text)
    wrapped = "<tool_call>" in cleaned
    inner = cleaned.replace("<tool_call>", "").replace("</tool_call>", "")

    tool_calls: list[dict] = []
    matched_spans: list[str] = []
    for cand in _extract_json_objects(inner):
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if not (isinstance(obj, dict) and "name" in obj and "arguments" in obj):
            continue
        if valid_names is not None and obj["name"] not in valid_names:
            continue
        args = obj["arguments"]
        args_str = args if isinstance(args, str) else json.dumps(args)
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "name": obj["name"],
            "arguments": args_str,
        })
        matched_spans.append(cand)

    if not tool_calls:
        return cleaned, []

    residual = inner
    for span in matched_spans:
        residual = residual.replace(span, "")
    residual = residual.strip()

    if not wrapped and ("```" in residual or len(residual) > MAX_PREAMBLE_CHARS):
        return cleaned, []

    for span in matched_spans:
        cleaned = cleaned.replace(span, "")
    cleaned = cleaned.replace("<tool_call>", "").replace("</tool_call>", "").strip()
    return cleaned, tool_calls
