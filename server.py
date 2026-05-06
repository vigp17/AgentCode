"""AgentCode server mode — JSON stdio protocol for the VS Code extension."""

import sys
import json
from pathlib import Path

import litellm

from tools import TOOL_DEFINITIONS, execute_tool
from agent import (
    AgentConfig, Conversation,
    build_system_prompt, load_project_config, load_hooks,
    _run_subagents, _run_hook, _is_denied, _get_permission,
)


# ── Wire protocol ─────────────────────────────────────────────────────────────

def _write(msg: dict) -> None:
    print(json.dumps(msg), flush=True)


def _read() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    try:
        return json.loads(line.strip())
    except (json.JSONDecodeError, ValueError):
        return None


# ── Permission over stdio ─────────────────────────────────────────────────────

def _ask_permission_server(tool_name: str, args: dict) -> bool:
    """Send a permission request and block until the extension responds."""
    _write({"type": "permission_request", "tool": tool_name, "args": args})
    while True:
        msg = _read()
        if msg is None:
            return False
        if msg.get("type") == "permission_response":
            return bool(msg.get("approved", False))


# ── Agentic loop (server edition) ─────────────────────────────────────────────

def _server_turn(
    user_input: str,
    conversation: Conversation,
    config: AgentConfig,
    file_context: str | None,
) -> None:
    """Run one agentic turn, streaming all output as JSON lines."""
    content = f"{file_context}\n\n{user_input}" if file_context else user_input
    conversation.messages.append({"role": "user", "content": content})

    router = config.router
    if router and router.enabled:
        router.cost_tracker.begin_turn()
        model, _tier, _reason = router.route(user_input)
    else:
        model = config.model

    conversation.compact(max_tokens=80_000, model=model)
    hooks = load_hooks(config.project_dir, config.settings)

    mcp = config.mcp_manager
    all_tools = TOOL_DEFINITIONS + (mcp.get_tool_definitions() if mcp else [])

    for _ in range(config.max_iterations):
        stream = litellm.completion(
            model=model,
            messages=[{"role": "system", "content": conversation.system}, *conversation.messages],
            tools=all_tools,
            tool_choice="auto",
            stream=True,
            stream_options={"include_usage": True},
        )

        full_text = ""
        tool_calls_accum: dict[int, dict] = {}
        usage = None

        for chunk in stream:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                full_text += delta.content
                _write({"type": "text", "delta": delta.content})

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_accum:
                        tool_calls_accum[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_accum[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_accum[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_accum[idx]["arguments"] += tc.function.arguments

        if router and usage:
            try:
                cost = litellm.completion_cost(
                    model=model,
                    prompt_tokens=usage.prompt_tokens or 0,
                    completion_tokens=usage.completion_tokens or 0,
                )
            except Exception:
                cost = 0.0
            router.cost_tracker.record(
                model,
                usage.prompt_tokens or 0,
                usage.completion_tokens or 0,
                cost,
            )

        if not tool_calls_accum:
            conversation.messages.append({"role": "assistant", "content": full_text})
            turn_cost = router.cost_tracker.last_turn_cost if router else 0.0
            _write({"type": "done", "cost": turn_cost})
            return

        conversation.messages.append({
            "role": "assistant",
            "content": full_text or "",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls_accum.values()
            ],
        })

        for tc in tool_calls_accum.values():
            tool_name = tc["name"]
            try:
                args = json.loads(tc["arguments"])
            except json.JSONDecodeError:
                args = {}

            _write({"type": "tool_call", "name": tool_name, "args": args})

            pre_hook = hooks.get(f"pre_{tool_name}") or hooks.get("pre_tool")
            if pre_hook:
                _run_hook(pre_hook, tool_name, args)

            if tool_name == "spawn_subagents":
                if _is_denied(tool_name, config):
                    result = f"Tool '{tool_name}' is blocked by settings."
                else:
                    result = _run_subagents(args.get("tasks", []), config, True)
            elif mcp and mcp.is_mcp_tool(tool_name):
                if _is_denied(tool_name, config):
                    result = f"Tool '{tool_name}' is blocked by settings."
                else:
                    result = mcp.call_tool(tool_name, args)
            else:
                perm = _get_permission(tool_name, config)
                if perm == "deny":
                    result = f"Tool '{tool_name}' is blocked by settings."
                elif perm == "ask":
                    approved = _ask_permission_server(tool_name, args)
                    result = execute_tool(tool_name, args) if approved else f"Tool '{tool_name}' denied."
                else:
                    result = execute_tool(tool_name, args)

            post_hook = hooks.get(f"post_{tool_name}") or hooks.get("post_tool")
            if post_hook:
                _run_hook(post_hook, tool_name, args)

            _write({"type": "tool_result", "name": tool_name, "result": result[:2000]})
            conversation.messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    _write({"type": "done", "cost": 0.0})


# ── Main server loop ──────────────────────────────────────────────────────────

def run_server(config: AgentConfig) -> None:
    """
    Read JSON messages from stdin, write JSON messages to stdout.

    Client → Server messages:
      {"type": "message",             "content": "..."}
      {"type": "context",             "file_path": "...", "content": "..."}
      {"type": "permission_response", "approved": true}
      {"type": "clear"}
      {"type": "ping"}

    Server → Client messages:
      {"type": "ready",              "model": "...", "project_dir": "..."}
      {"type": "text",               "delta": "..."}
      {"type": "tool_call",          "name": "...", "args": {...}}
      {"type": "tool_result",        "name": "...", "result": "..."}
      {"type": "permission_request", "tool": "...", "args": {...}}
      {"type": "done",               "cost": 0.0}
      {"type": "error",              "message": "..."}
      {"type": "cleared"}
      {"type": "pong"}
    """
    project_config = load_project_config(config.project_dir)
    system_prompt = build_system_prompt(config.project_dir, project_config["combined"])
    conversation = Conversation(system=system_prompt)

    sess = Path(config.project_dir) / ".agentcode_session.json"
    conversation.load(sess)

    _write({"type": "ready", "model": config.model, "project_dir": config.project_dir})

    file_context: str | None = None

    while True:
        msg = _read()
        if msg is None:
            break

        msg_type = msg.get("type")

        if msg_type == "message":
            content = msg.get("content", "").strip()
            if not content:
                continue
            try:
                _server_turn(content, conversation, config, file_context)
                conversation.save(sess)
            except Exception as e:
                _write({"type": "error", "message": str(e)})

        elif msg_type == "context":
            file_path = msg.get("file_path", "")
            file_content = msg.get("content", "")
            if file_path and file_content:
                file_context = (
                    f"[Active file: {file_path}]\n```\n{file_content[:3000]}\n```"
                )
            else:
                file_context = None

        elif msg_type == "clear":
            conversation.messages.clear()
            sess.unlink(missing_ok=True)
            file_context = None
            _write({"type": "cleared"})

        elif msg_type == "ping":
            _write({"type": "pong"})
