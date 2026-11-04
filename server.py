"""AgentCode server mode — JSON stdio protocol for the VS Code extension."""

import sys
import json
import threading
from pathlib import Path


from tools import TOOL_DEFINITIONS
import model_discovery
import finetuned
import trajectories
from agent import (
    AgentConfig, Conversation,
    build_system_prompt, load_project_config, load_hooks,
    _dispatch_tool, _record_cost, _run_hook, stream_completion,
)
from xml_tool_parser import (
    looks_like_xml_tool_call, parse_xml_tool_calls, strip_think,
    looks_like_json_tool_call, parse_json_tool_calls,
)


# ── Wire protocol ─────────────────────────────────────────────────────────────

_write_lock = threading.Lock()


def _write(msg: dict) -> None:
    # Model discovery writes from a background thread; keep each JSON line whole.
    line = json.dumps(msg) + "\n"
    with _write_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def _send_models(config: AgentConfig, refresh: bool = False) -> None:
    """Discover models in the background and push them to the picker."""
    settings = config.settings
    if settings and not settings.model.auto_discover:
        _write({"type": "models", "models": model_discovery.to_wire({}, finetuned.as_discovered()),
                "current": config.model})
        return

    def done(results):
        if config.router and (not settings or settings.model.auto_tiers):
            config.router.apply_discovered(results, model_discovery.auto_tiers)
        _write({
            "type": "models",
            "models": model_discovery.to_wire(results, finetuned.as_discovered()),
            "current": config.model,
        })

    model_discovery.refresh_in_background(done, refresh=refresh)


def _read() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    try:
        return json.loads(line.strip())
    except (json.JSONDecodeError, ValueError):
        return None


# ── Live model / effort switching ─────────────────────────────────────────────

def _write_model_changed(config: AgentConfig) -> None:
    _write({
        "type": "model_changed",
        "model": config.model,
        "effort": config.effort,
        "routing": bool(config.router and config.router.enabled),
    })


def _set_model(config: AgentConfig, model: str) -> None:
    """Switch model in place — no process restart, conversation kept.

    Like the CLI's /model: an explicit pick disables cost-aware routing so
    the chosen model is the one that answers.
    """
    if not model:
        _write({"type": "error", "message": "set_model needs a model name."})
        return
    config.model = model
    if config.router:
        config.router.enabled = False
        config.router.default_model = model
        config.router.provider = config.router.detect_provider(model)
    _write_model_changed(config)


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
    user_msg = {"role": "user", "content": content}
    conversation.messages.append(user_msg)

    router = config.router
    if router and router.enabled:
        router.cost_tracker.begin_turn()
        model, _tier, _reason = router.route(user_input)
    else:
        model = config.model

    ft = finetuned.get(model)
    if ft and ft.kind != "chat":
        conversation.messages.pop()
        _write({"type": "error", "message":
                f"{model} generates images, not chat replies. Image generation with /image "
                "isn't wired up yet — pick a chat model for now."})
        _write({"type": "done", "cost": 0.0})
        return
    if ft and finetuned.is_ecosystem(model) and not ft.configured:
        conversation.messages.pop()
        env = f"AGENTCODE_FT_{finetuned._env_name(model)}_API_BASE"
        _write({"type": "error", "message":
                f"{model} isn't connected. Serve it and set {env} (e.g. in your workspace .env), "
                "or pick another model."})
        _write({"type": "done", "cost": 0.0})
        return

    system = conversation.system
    max_iterations = config.max_iterations
    if ft and ft.max_iterations:
        max_iterations = ft.max_iterations
    if ft and ft.supports_effort:
        system += finetuned.effort_system_suffix(config.effort)
        max_iterations = finetuned.EFFORT_MAX_ITERATIONS[config.effort]

    conversation.compact(max_tokens=80_000, model=model)
    hooks = load_hooks(config.project_dir, config.settings)

    mcp = config.mcp_manager
    all_tools = TOOL_DEFINITIONS + (mcp.get_tool_definitions() if mcp else [])
    tool_names = {t["function"]["name"] for t in all_tools}

    for _ in range(max_iterations):
        full_text, tool_calls_accum, usage = stream_completion(
            lambda text: _write({"type": "text", "delta": text}),
            **finetuned.completion_kwargs(model),
            messages=[{"role": "system", "content": system}, *conversation.messages],
            tools=all_tools,
            tool_choice="auto",
            stream_options={"include_usage": True},
        )

        # Open-weight fine-tunes emit tool calls in the text response instead of
        # litellm's structured tool_calls field. Some use XML (<function=...>),
        # others bare JSON ({"name":...,"arguments":...}). Detect and convert both.
        if not tool_calls_accum:
            if looks_like_xml_tool_call(full_text):
                cleaned, parsed = parse_xml_tool_calls(full_text)
            elif looks_like_json_tool_call(full_text):
                cleaned, parsed = parse_json_tool_calls(full_text, tool_names)
            else:
                cleaned, parsed = strip_think(full_text), []
            full_text = cleaned
            for i, tc in enumerate(parsed):
                tool_calls_accum[i] = tc

        if router and usage:
            _record_cost(router, model, usage)

        if not tool_calls_accum:
            conversation.messages.append({"role": "assistant", "content": full_text})
            trajectories.log_turn(
                config.settings, system,
                trajectories.turn_slice(conversation.messages, user_msg),
                all_tools, model, config.project_dir,
            )
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

            result = _dispatch_tool(tool_name, args, config, True)

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
      {"type": "list_models",         "refresh": false}
      {"type": "set_model",           "model": "axon-1"}
      {"type": "set_effort",          "effort": "low" | "medium" | "high"}
      {"type": "ping"}

    Server → Client messages:
      {"type": "ready",              "model": "...", "project_dir": "...", "effort": "..."}
      {"type": "model_changed",      "model": "...", "effort": "...", "routing": false}
      {"type": "text",               "delta": "..."}
      {"type": "tool_call",          "name": "...", "args": {...}}
      {"type": "tool_result",        "name": "...", "result": "..."}
      {"type": "permission_request", "tool": "...", "args": {...}}
      {"type": "done",               "cost": 0.0}
      {"type": "error",              "message": "..."}
      {"type": "cleared"}
      {"type": "models",             "models": [{id, provider, label, ...}], "current": "..."}
      {"type": "pong"}
    """
    # Route every approval — including ones raised by subagent threads — through
    # the stdio protocol. The terminal prompt would read the extension's own
    # stdin and print non-JSON to the protocol stream.
    config.permission_cb = _ask_permission_server

    project_config = load_project_config(config.project_dir)
    system_prompt = build_system_prompt(config.project_dir, project_config["combined"])
    conversation = Conversation(system=system_prompt)

    sess = Path(config.project_dir) / ".agentcode_session.json"
    conversation.load(sess)

    _write({"type": "ready", "model": config.model, "project_dir": config.project_dir,
            "effort": config.effort})
    _send_models(config)

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

        elif msg_type == "set_model":
            _set_model(config, str(msg.get("model", "")).strip())

        elif msg_type == "set_effort":
            effort = str(msg.get("effort", "")).lower()
            if effort in finetuned.EFFORT_LEVELS:
                config.effort = effort
                _write_model_changed(config)
            else:
                _write({"type": "error", "message": f"Unknown effort '{effort}' (low, medium, high)."})

        elif msg_type == "list_models":
            _send_models(config, refresh=bool(msg.get("refresh")))

        elif msg_type == "ping":
            _write({"type": "pong"})
