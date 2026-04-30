#!/usr/bin/env python3
"""
AgentCode — Brain module.

Provides the agentic loop, conversation management, tool execution,
and AGENTCODE.md project config support.
"""

import os
import json
from dataclasses import dataclass, field
from pathlib import Path

import litellm
from rich.console import Console
from rich.markdown import Markdown

from tools import TOOL_DEFINITIONS, execute_tool

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
    model: str = "claude-sonnet-4-6"
    auto_approve: bool = False
    project_dir: str = field(default_factory=os.getcwd)
    max_iterations: int = field(
        default_factory=lambda: int(os.environ.get("AGENTCODE_MAX_ITERATIONS", "25"))
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

    def compact(self, max_tokens: int = 80_000):
        """Drop old messages to reduce context. max_tokens=0 forces compaction."""
        if not self.messages:
            return
        if max_tokens > 0 and self.token_estimate() <= max_tokens:
            return

        keep = 6
        if len(self.messages) <= keep:
            return

        dropped = len(self.messages) - keep
        recent = self.messages[-keep:]
        self.messages = [
            {"role": "user", "content": f"[{dropped} earlier messages were compacted to save context.]"},
            {"role": "assistant", "content": "Understood. Continuing from the recent context."},
            *recent,
        ]


# ── Permission prompt ─────────────────────────────────────────────────────────

def _ask_permission(tool_name: str, args: dict) -> bool:
    """Ask the user to approve a write/execute tool call. Returns True if approved."""
    console.print(f"\n[bold yellow]⚡ Tool request:[/bold yellow] [bold]{tool_name}[/bold]")
    for key, val in args.items():
        val_str = str(val)
        if len(val_str) > 300:
            val_str = val_str[:300] + "..."
        console.print(f"   [dim]{key}:[/dim] {val_str}")
    try:
        answer = console.input("  [bold]Allow? [y/N][/bold] ").strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# ── Agentic loop ──────────────────────────────────────────────────────────────

def run_agent_loop(user_input: str, conversation: Conversation, config: AgentConfig) -> str:
    """
    Add user_input to conversation, call the LLM, handle tool calls in a loop,
    and return the final text response.
    """
    conversation.messages.append({"role": "user", "content": user_input})

    for _ in range(config.max_iterations):
        response = litellm.completion(
            model=config.model,
            messages=[{"role": "system", "content": conversation.system}, *conversation.messages],
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        tool_calls = msg.tool_calls

        if not tool_calls:
            content = msg.content or ""
            conversation.messages.append({"role": "assistant", "content": content})
            return content

        # Record assistant turn with tool calls
        assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
        assistant_entry["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
        conversation.messages.append(assistant_entry)

        # Execute each tool call and append results
        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if tool_name in WRITE_TOOLS and not config.auto_approve:
                approved = _ask_permission(tool_name, args)
                result = execute_tool(tool_name, args) if approved else f"Tool '{tool_name}' denied by user."
            else:
                result = execute_tool(tool_name, args)

            conversation.messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "Reached max iterations. The task may be incomplete."


# ── Display ───────────────────────────────────────────────────────────────────

def display_response(text: str):
    """Render the assistant's final response as Markdown in the terminal."""
    if not text:
        return
    console.print()
    console.print(Markdown(text))
    console.print()


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
