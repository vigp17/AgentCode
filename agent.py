#!/usr/bin/env python3
"""
AgentCode — Brain module.

Provides the agentic loop, conversation management, tool execution,
and AGENTCODE.md project config support.
"""

import os
import json
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path

import litellm
from rich.console import Console
from rich.markdown import Markdown

from xml_tool_parser import (
    looks_like_xml_tool_call, parse_xml_tool_calls, strip_think,
    looks_like_json_tool_call, parse_json_tool_calls,
)

from tools import TOOL_DEFINITIONS, execute_tool
from router import ModelRouter, display_routing_decision
from mcp_client import MCPManager
from settings import Settings

console = Console()

# Tools that require explicit user permission before running
WRITE_TOOLS = {"write_file", "edit_file", "run_command", "git_commit", "git_branch", "git_push"}

SYSTEM_PROMPT = """\
You are AgentCode, an expert software engineering assistant running in the terminal.

You are working in the directory: {cwd}

You have access to tools to read files, write files, edit files, run commands, \
search for files, and search for text. Use them to help the user with their coding tasks.

Guidelines:
- Always read a file before editing it.
- Prefer edit_file over write_file for modifying existing files.
- Break complex tasks into steps and explain what you're doing.
- When you encounter errors, diagnose the root cause before applying a fix.
- Be concise but thorough.
"""


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    model: str = "claude-sonnet-5"
    auto_approve: bool = False
    project_dir: str = field(default_factory=os.getcwd)
    max_iterations: int = field(
        default_factory=lambda: int(os.environ.get("AGENTCODE_MAX_ITERATIONS", "25"))
    )
    router: ModelRouter | None = None
    mcp_manager: MCPManager | None = None
    settings: Settings | None = None
    # How the user is asked to approve a tool call. Defaults to the terminal
    # prompt; server mode injects a stdio-based one so subagents don't read
    # from the same stdin the JSON protocol is using.
    permission_cb: Callable[[str, dict], bool] | None = None
    # Current nesting level; subagents run at parent depth + 1.
    subagent_depth: int = 0
    max_subagent_depth: int = field(
        default_factory=lambda: int(os.environ.get("AGENTCODE_MAX_SUBAGENT_DEPTH", "1"))
    )


# ── Conversation ──────────────────────────────────────────────────────────────

class Conversation:
    def __init__(self, system: str = ""):
        self.system = system
        self.messages: list[dict] = []

    def token_estimate(self) -> int:
        """Rough token estimate based on character count (~4 chars/token)."""
        total = len(self.system)
        for msg in self.messages:
            content = msg.get("content") or ""
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += len(str(block.get("text", "") or block.get("content", "")))
            else:
                total += len(str(content))
        return total // 4

    def save(self, path: Path):
        path.write_text(json.dumps(self.messages, indent=2))

    def load(self, path: Path):
        if path.exists():
            self.messages = json.loads(path.read_text())

    def _safe_split(self, index: int) -> int:
        """Move a split point forward past any leading tool results.

        A 'tool' message is only valid when the assistant turn that requested
        it is still in history. Slicing blindly can drop that assistant turn
        and leave dangling tool results, which providers reject with a 400.
        """
        while index < len(self.messages) and self.messages[index].get("role") == "tool":
            index += 1
        return index

    def compact(self, max_tokens: int = 80_000, model: str | None = None):
        """Summarize old messages to reduce context. Falls back to dropping if no model given."""
        if not self.messages:
            return
        if max_tokens > 0 and self.token_estimate() <= max_tokens:
            return

        keep = 6
        if len(self.messages) <= keep:
            return

        split = self._safe_split(len(self.messages) - keep)
        # Everything after the split is tool results — compacting here would
        # orphan them. Wait for the next assistant turn.
        if split <= 0 or split >= len(self.messages):
            return

        to_summarize = self.messages[:split]
        recent = self.messages[split:]
        summary = self._summarize(to_summarize, model) if model else (
            f"[{len(to_summarize)} earlier messages were compacted to save context.]"
        )
        self.messages = [
            {"role": "user", "content": summary},
            {"role": "assistant", "content": "Understood. Continuing from the recent context."},
            *recent,
        ]

    def _summarize(self, messages: list[dict], model: str) -> str:
        try:
            parts = []
            for m in messages:
                role = m.get("role", "")
                content = m.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        str(b.get("text", "")) for b in content if isinstance(b, dict)
                    )
                if role in ("user", "assistant") and content:
                    parts.append(f"{role.upper()}: {str(content)[:500]}")
            if not parts:
                return f"[{len(messages)} earlier messages were compacted.]"
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": (
                    "Summarize this conversation in 3-5 sentences, preserving key decisions, "
                    "file names, and code changes:\n\n" + "\n".join(parts[:50])
                )}],
                max_tokens=300,
            )
            return f"[Context summary: {resp.choices[0].message.content}]"
        except Exception:
            return f"[{len(messages)} earlier messages were compacted to save context.]"


# ── Hooks ─────────────────────────────────────────────────────────────────────

def load_hooks(project_dir: str, settings: Settings | None = None) -> dict:
    hooks: dict = {}
    # settings.hooks has lowest priority
    if settings and settings.hooks:
        hooks.update(settings.hooks)
    # hooks.json files override settings hooks
    for path in [Path.home() / ".agentcode" / "hooks.json",
                 Path(project_dir) / ".agentcode" / "hooks.json"]:
        if path.exists():
            try:
                hooks.update(json.loads(path.read_text()))
            except Exception:
                pass
    return hooks


def _run_hook(cmd: str, tool_name: str, args: dict) -> None:
    env = os.environ.copy()
    env["AGENTCODE_TOOL"] = tool_name
    for key, val in args.items():
        env[f"AGENTCODE_{key.upper()}"] = str(val)
    try:
        subprocess.run(cmd, shell=True, env=env, timeout=10, capture_output=True)
    except Exception:
        pass


# ── Permission prompt ─────────────────────────────────────────────────────────

def _ask_permission(tool_name: str, args: dict) -> bool:
    """Ask the user to approve a write/execute tool call. Returns True if approved."""
    console.print(f"\n[bold yellow]⚡ Tool request:[/bold yellow] [bold]{tool_name}[/bold]")
    for key, val in args.items():
        val_str = str(val)
        # Never truncate the command itself — the dangerous part is often at
        # the end, and the user is approving this exact string.
        if key != "command" and len(val_str) > 300:
            val_str = val_str[:300] + "..."
        console.print(f"   [dim]{key}:[/dim] {val_str}")
    try:
        answer = console.input("  [bold]Allow? [y/N][/bold] ").strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# Serializes permission prompts. Subagents run in parallel threads and would
# otherwise race for stdin (CLI) or interleave requests on the JSON protocol
# (server mode).
_permission_lock = threading.Lock()


def _request_permission(tool_name: str, args: dict, config: "AgentConfig") -> bool:
    """Ask the user for approval through the configured prompt, one at a time."""
    ask = config.permission_cb or _ask_permission
    with _permission_lock:
        return ask(tool_name, args)


# ── Permission helpers ────────────────────────────────────────────────────────

def _get_permission(tool_name: str, config: "AgentConfig") -> str:
    """
    Return 'allow', 'deny', or 'ask' for a tool call.

    Priority:
      deny list → auto_approve_all (CLI or settings) → auto_approve list → ask
    Falls back to the old WRITE_TOOLS behaviour when no settings are loaded.
    MCP tools are never on the default auto-approve list, so they ask.
    """
    s = config.settings

    if s and tool_name in s.permissions.deny:
        return "deny"

    if config.auto_approve or (s and s.permissions.auto_approve_all):
        return "allow"

    if s:
        return "allow" if tool_name in s.permissions.auto_approve else "ask"

    # Backward-compat fallback: write tools and MCP tools ask, everything else auto
    if tool_name in WRITE_TOOLS or tool_name.startswith("mcp__"):
        return "ask"
    return "allow"


# ── Tool dispatch ─────────────────────────────────────────────────────────────

def _dispatch_tool(tool_name: str, args: dict, config: "AgentConfig", silent: bool) -> str:
    """Resolve permissions and run one tool call — built-in, MCP, or subagent."""
    perm = _get_permission(tool_name, config)
    if perm == "deny":
        return f"Tool '{tool_name}' is blocked by settings (permissions.deny)."
    if perm == "ask" and not _request_permission(tool_name, args, config):
        return f"Tool '{tool_name}' denied by user."

    if tool_name == "spawn_subagents":
        return _run_subagents(args.get("tasks", []), config, silent)

    mcp = config.mcp_manager
    if mcp and mcp.is_mcp_tool(tool_name):
        return mcp.call_tool(tool_name, args)

    return execute_tool(tool_name, args)


# ── Cost accounting ───────────────────────────────────────────────────────────

def _record_cost(router: ModelRouter, model: str, usage) -> None:
    """Record token usage and cost for one API call.

    litellm.completion_cost() takes no prompt_tokens/completion_tokens
    arguments, so price from the router's own tier table first and only fall
    back to litellm for models we don't have pricing for.
    """
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    cost = router.estimate_cost(model, prompt_tokens, completion_tokens)
    if cost == 0.0:
        try:
            cost = litellm.completion_cost(
                completion_response={
                    "model": model,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )
        except Exception:
            cost = 0.0

    router.cost_tracker.record(model, prompt_tokens, completion_tokens, cost)


# ── Subagents ─────────────────────────────────────────────────────────────────

_print_lock = threading.Lock()


def _run_subagents(tasks: list[str], config: AgentConfig, parent_silent: bool = False) -> str:
    """Run multiple agent loops in parallel and return their combined results."""
    if not tasks:
        return "Error: no tasks provided."

    # Subagents inherit the full tool set, including spawn_subagents. Without a
    # depth cap a single call can fan out to 5^n threads.
    if config.subagent_depth >= config.max_subagent_depth:
        return (
            f"Error: subagent depth limit reached (max_subagent_depth="
            f"{config.max_subagent_depth}). Complete this task directly "
            "instead of delegating further."
        )

    child_config = replace(config, subagent_depth=config.subagent_depth + 1)
    max_workers = min(len(tasks), 5)
    results: dict[int, str] = {}

    if not parent_silent:
        console.print(f"\n[dim]  ↳ Spawning {len(tasks)} subagent(s) in parallel...[/dim]")

    def run_one(idx: int, task: str) -> tuple[int, str]:
        sub_conv = Conversation(system=build_system_prompt(config.project_dir))
        result = run_agent_loop(task, sub_conv, child_config, silent=True)
        if not parent_silent:
            with _print_lock:
                console.print(f"  [dim]✓ Subagent {idx + 1}/{len(tasks)} done[/dim]")
        return idx, result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one, i, task): i for i, task in enumerate(tasks)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    parts = []
    for i, task in enumerate(tasks):
        parts.append(f"**Subagent {i + 1}:** {task}\n\n{results[i]}")
    return "\n\n---\n\n".join(parts)


# ── Agentic loop ──────────────────────────────────────────────────────────────

def run_agent_loop(
    user_input: str,
    conversation: Conversation,
    config: AgentConfig,
    silent: bool = False,
) -> str:
    """
    Add user_input to conversation, stream the LLM response, handle tool calls
    in a loop, and return the final text response.

    silent=True suppresses all console output (used by subagents).
    """
    conversation.messages.append({"role": "user", "content": user_input})

    router = config.router
    if router:
        router.cost_tracker.begin_turn()
        model, tier, reason = router.route(user_input)
        if not silent:
            display_routing_decision(model, tier, reason, router)
    else:
        model = config.model

    conversation.compact(max_tokens=80_000, model=model)
    hooks = load_hooks(config.project_dir, config.settings)

    mcp = config.mcp_manager
    all_tools = TOOL_DEFINITIONS + (mcp.get_tool_definitions() if mcp else [])
    tool_names = {t["function"]["name"] for t in all_tools}

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
        tool_calls_accum: dict[int, dict] = {}  # index → {id, name, arguments}
        usage = None

        if not silent:
            console.print()

        for chunk in stream:
            # Capture usage from the final chunk
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Stream text tokens live
            if delta.content:
                full_text += delta.content
                if not silent:
                    console.print(delta.content, end="", markup=False, highlight=False)

            # Accumulate tool call chunks (arguments arrive in pieces)
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

        if not silent:
            console.print()

        # Record cost from usage in final chunk
        if router and usage:
            _record_cost(router, model, usage)

        # Open-weight fine-tunes emit tool calls in the text response: some use
        # XML (<function=...>), others bare JSON. Detect and convert before
        # treating this as a pure text turn.
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

        # Pure text response — done
        if not tool_calls_accum:
            conversation.messages.append({"role": "assistant", "content": full_text})
            return full_text

        # Append assistant turn with accumulated tool calls
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

        # Execute each tool call and append results
        for tc in tool_calls_accum.values():
            tool_name = tc["name"]
            try:
                args = json.loads(tc["arguments"])
            except json.JSONDecodeError:
                args = {}

            pre_hook = hooks.get(f"pre_{tool_name}") or hooks.get("pre_tool")
            if pre_hook:
                _run_hook(pre_hook, tool_name, args)

            result = _dispatch_tool(tool_name, args, config, silent)

            post_hook = hooks.get(f"post_{tool_name}") or hooks.get("post_tool")
            if post_hook:
                _run_hook(post_hook, tool_name, args)

            conversation.messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    return "Reached max iterations. The task may be incomplete."


# ── AGENTCODE.md support ──────────────────────────────────────────────────────

def load_project_config(project_dir: str) -> dict:
    """
    Load AGENTCODE.md from the project directory and/or ~/.agentcode/AGENTCODE.md.
    Returns a dict with project_path, global_path, and combined content.
    """
    result: dict = {"project_path": None, "global_path": None, "combined": ""}
    parts: list[str] = []

    project_file = Path(project_dir) / "AGENTCODE.md"
    if project_file.exists():
        result["project_path"] = str(project_file)
        parts.append(project_file.read_text())

    global_file = Path.home() / ".agentcode" / "AGENTCODE.md"
    if global_file.exists():
        result["global_path"] = str(global_file)
        parts.append(global_file.read_text())

    result["combined"] = "\n\n".join(parts)
    return result


def build_system_prompt(project_dir: str, agentcode_content: str = "") -> str:
    """Build the system prompt, appending AGENTCODE.md instructions if present."""
    base = SYSTEM_PROMPT.format(cwd=project_dir)
    if agentcode_content.strip():
        base += f"\n\n## Project Instructions (from AGENTCODE.md)\n\n{agentcode_content}"
    return base
