#!/usr/bin/env python3
"""
AgentCode — an open, multi-model agentic coding CLI.

Usage:
    agentcode                    # Interactive REPL
    agentcode "fix the bug"      # One-shot mode
    agentcode --model gpt-4o     # Use a different model

Supported models (via LiteLLM):
    claude-sonnet-4-6            # Anthropic Claude (default)
    gpt-4o                       # OpenAI
    gemini/gemini-2.5-pro        # Google
    ollama/llama3                # Local via Ollama
    ... and 100+ more: https://docs.litellm.ai/docs/providers
"""

import os
import sys
import argparse

from rich.console import Console
from rich.text import Text
from rich.theme import Theme

from agent import (
    AgentConfig, Conversation, run_agent_loop, display_response,
    load_project_config, build_system_prompt,
)

# ── Theme ─────────────────────────────────────────────────────────────────────

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "prompt": "bold magenta",
})

console = Console(theme=custom_theme)

# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = r"""
   ___                    __  ______          __
  / _ | ___ ____ ___  ___/ /_/ ___/__  ___/ /__
 / __ |/ _ `/ -_) _ \/ __/ __/ /__/ _ \/ _  / -_)
/_/ |_|\_, /\__/_//_/\__/\__/\___/\___/\_,_/\__/
      /___/
"""


def show_banner(config: AgentConfig, project_config: dict | None = None):
    console.print(Text(BANNER, style="bold cyan"))
    console.print(f"  [dim]Model:[/dim]  [bold]{config.model}[/bold]")
    console.print(f"  [dim]Dir:[/dim]    [bold]{config.project_dir}[/bold]")
    console.print(f"  [dim]Safety:[/dim] [bold]{'auto-approve' if config.auto_approve else 'ask before write/exec'}[/bold]")

    if project_config:
        if project_config.get("project_path"):
            console.print(f"  [dim]Config:[/dim] [bold green]✓ {project_config['project_path']}[/bold green]")
        if project_config.get("global_path"):
            console.print(f"  [dim]Global:[/dim] [bold green]✓ {project_config['global_path']}[/bold green]")
        if not project_config.get("project_path") and not project_config.get("global_path"):
            console.print("  [dim]Config:[/dim] [dim]no AGENTCODE.md found (optional)[/dim]")

    console.print()
    console.print("  [dim]Type your request, or:[/dim]")
    console.print("  [dim]  /model <name>  — switch model[/dim]")
    console.print("  [dim]  /clear          — reset conversation[/dim]")
    console.print("  [dim]  /compact        — force context compaction[/dim]")
    console.print("  [dim]  /init           — create AGENTCODE.md template[/dim]")
    console.print("  [dim]  /tokens         — show token usage[/dim]")
    console.print("  [dim]  /help           — show this help[/dim]")
    console.print("  [dim]  /exit           — quit[/dim]")
    console.print()


# ── Slash Commands ────────────────────────────────────────────────────────────

def handle_slash_command(
    cmd: str, config: AgentConfig, conversation: Conversation, project_config: dict | None = None
) -> bool:
    """Handle /commands. Returns True if the command was handled."""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ("/exit", "/quit"):
        console.print("[dim]Goodbye![/dim]")
        sys.exit(0)

    elif command == "/clear":
        conversation.messages.clear()
        console.print("[success]✓ Conversation cleared[/success]")
        return True

    elif command == "/model":
        if arg:
            config.model = arg
            console.print(f"[success]✓ Switched to model: {arg}[/success]")
        else:
            console.print(f"[info]Current model: {config.model}[/info]")
            console.print("[dim]Usage: /model <model_name>[/dim]")
            console.print("[dim]Examples: claude-sonnet-4-6, gpt-4o, ollama/llama3[/dim]")
        return True

    elif command == "/compact":
        conversation.compact(max_tokens=0)
        console.print("[success]✓ Context compacted[/success]")
        return True

    elif command == "/help":
        show_banner(config, project_config)
        return True

    elif command == "/init":
        _create_agentcode_template(config.project_dir)
        return True

    elif command == "/tokens":
        est = conversation.token_estimate()
        console.print(f"[info]Estimated tokens in context: ~{est:,}[/info]")
        return True

    return False


# ── AGENTCODE.md Template ─────────────────────────────────────────────────────

AGENTCODE_TEMPLATE = """\
# AGENTCODE.md — Project Configuration for AgentCode

## Project Overview
<!-- Brief description of what this project does -->
This is a [describe your project here].

## Tech Stack
<!-- Languages, frameworks, and key dependencies -->
- Language: Python 3.11+
- Framework: [your framework]
- Key dependencies: [list them]

## Coding Standards
<!-- Rules the agent should follow when writing code -->
- Use type hints for all function signatures
- Write docstrings for all public functions
- Follow PEP 8 style conventions
- Keep functions under 50 lines where possible

## File Structure
<!-- Help the agent understand your project layout -->
- `src/` — main source code
- `tests/` — test files (mirror src/ structure)
- `docs/` — documentation

## Testing
<!-- How to run tests and what framework you use -->
- Framework: pytest
- Run tests: `pytest tests/ -v`
- Always write tests for new functions

## Preferences
<!-- Your personal preferences for how the agent works -->
- Prefer descriptive variable names over abbreviations
- Use f-strings for string formatting
- Add comments for complex logic, skip obvious ones
- When in doubt, ask before making breaking changes

## Do NOT
<!-- Things the agent should avoid -->
- Do not modify configuration files without asking
- Do not install new dependencies without approval
- Do not delete files without confirmation
"""


def _create_agentcode_template(project_dir: str):
    """Create an AGENTCODE.md template in the project directory."""
    from pathlib import Path
    target = Path(project_dir) / "AGENTCODE.md"

    if target.exists():
        console.print(f"[warning]⚠ AGENTCODE.md already exists at {target}[/warning]")
        return

    target.write_text(AGENTCODE_TEMPLATE)
    console.print(f"[success]✓ Created AGENTCODE.md at {target}[/success]")
    console.print("[dim]  Edit it to define your project's coding standards and preferences.[/dim]")
    console.print("[dim]  Restart AgentCode to load the config.[/dim]")


# ── REPL ──────────────────────────────────────────────────────────────────────

def repl(config: AgentConfig):
    """Interactive read-eval-print loop."""
    project_config = load_project_config(config.project_dir)
    system_prompt = build_system_prompt(config.project_dir, project_config["combined"])
    conversation = Conversation(system=system_prompt)

    show_banner(config, project_config)

    while True:
        try:
            user_input = console.input("[prompt]❯[/prompt] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if handle_slash_command(user_input, config, conversation, project_config):
                continue

        try:
            response = run_agent_loop(user_input, conversation, config)
            display_response(response)
        except KeyboardInterrupt:
            console.print("\n[warning]⚠ Interrupted[/warning]")
            continue
        except Exception as e:
            console.print(f"[error]Error: {e}[/error]")
            continue


# ── One-Shot Mode ─────────────────────────────────────────────────────────────

def one_shot(prompt: str, config: AgentConfig):
    """Execute a single prompt and exit."""
    project_config = load_project_config(config.project_dir)
    system_prompt = build_system_prompt(config.project_dir, project_config["combined"])
    conversation = Conversation(system=system_prompt)
    response = run_agent_loop(prompt, conversation, config)
    display_response(response)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AgentCode — multi-model agentic coding CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="One-shot prompt (omit for interactive REPL)",
    )
    parser.add_argument(
        "--model", "-m",
        default=os.environ.get("AGENTCODE_MODEL", "claude-sonnet-4-6"),
        help="LLM model to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--auto-approve", "-y",
        action="store_true",
        help="Skip permission prompts for all tool calls (use with caution!)",
    )
    parser.add_argument(
        "--dir", "-d",
        default=os.getcwd(),
        help="Project directory to operate in (default: current dir)",
    )

    args = parser.parse_args()

    config = AgentConfig(
        model=args.model,
        auto_approve=args.auto_approve,
        project_dir=os.path.abspath(args.dir),
    )

    os.chdir(config.project_dir)

    if args.prompt:
        one_shot(args.prompt, config)
    else:
        repl(config)


if __name__ == "__main__":
    main()
